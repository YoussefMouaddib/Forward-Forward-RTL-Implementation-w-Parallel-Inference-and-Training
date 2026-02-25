// relu_norm.sv
// Post-MAC processing for one complete layer.
// Three sequential operations:
//   1. Add bias to each neuron's raw accumulator output
//   2. Apply ReLU (zero out negatives)
//   3. Mean-only normalization (subtract mean from all values)
//
// Normalization requires two passes over the neuron outputs:
//   Pass 1 — accumulate sum to compute mean
//   Pass 2 — subtract mean from each value
// This means relu_norm cannot be purely pipelined with mac_unit.
// It waits for mac_unit done, then does its own two-pass computation.
//
// RTL equivalent of q_relu() and q_layer_norm() in Python reference.
// Mean computed in Q16.16, division by NUM_NEURONS done as right-shift
// approximation when NUM_NEURONS is power of two (256 = 2^8, shift by 8).

module relu_norm #(
    parameter NUM_NEURONS = 256,
    parameter DATA_WIDTH  = 32,
    parameter FRAC_BITS   = 16,
    // Log2 of NUM_NEURONS — used for mean division
    // 256 neurons -> NEURON_LOG2 = 8, divide by shifting right 8
    parameter NEURON_LOG2 = 8
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Control
    input  logic                              start,    // pulse from mac_unit done
    output logic                              done,     // pulse when norm complete

    // Bias memory interface — read only
    // Bias stored in small ROM/RAM, one value per neuron
    output logic [$clog2(NUM_NEURONS)-1:0]    bias_addr,
    output logic                              bias_en,
    input  logic [DATA_WIDTH-1:0]             bias_rdata,

    // Raw MAC results — read from intermediate buffer
    // Written by mac_unit, read here before ReLU
    output logic [$clog2(NUM_NEURONS)-1:0]    raw_addr,
    output logic                              raw_en,
    input  logic [DATA_WIDTH-1:0]             raw_rdata,

    // Output to activation buffer
    output logic [$clog2(NUM_NEURONS)-1:0]    out_addr,
    output logic                              out_we,
    output logic [DATA_WIDTH-1:0]             out_wdata,

    // Clear activation buffer at start of pass
    output logic                              buf_clear
);

    // ─────────────────────────────────────────────
    // STATE MACHINE
    // ─────────────────────────────────────────────
    typedef enum logic [2:0] {
        IDLE       = 3'b000,
        BIAS_READ  = 3'b001,   // read raw MAC result + bias for one neuron
        BIAS_APPLY = 3'b010,   // add bias, apply ReLU, write to temp storage
        SUM_READ   = 3'b011,   // pass 1: read post-ReLU values to accumulate sum
        SUM_ACC    = 3'b100,   // pass 1: accumulate into sum register
        NORM_READ  = 3'b101,   // pass 2: read post-ReLU values again
        NORM_WRITE = 3'b110,   // pass 2: subtract mean, write to activation buffer
        DONE       = 3'b111
    } state_t;

    state_t state;

    // ─────────────────────────────────────────────
    // COUNTERS
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_NEURONS)-1:0] neuron_idx;

    // ─────────────────────────────────────────────
    // INTERMEDIATE STORAGE
    // Holds post-ReLU values before normalization
    // Same register file pattern as activation_buffer
    // ─────────────────────────────────────────────
    logic signed [0:NUM_NEURONS-1][DATA_WIDTH-1:0] post_relu ;

    // ─────────────────────────────────────────────
    // SUM ACCUMULATOR FOR MEAN COMPUTATION
    // Needs extra bits to hold sum of 256 x 32-bit values
    // 256 * 2^31 = 2^39, so 40 bits minimum, use 48 for safety
    // ─────────────────────────────────────────────
    logic signed [47:0] sum_acc;

    // Mean in Q16.16 — computed after pass 1
    // Divide sum by NUM_NEURONS = right-shift by NEURON_LOG2 (8 for 256 neurons)
    logic signed [DATA_WIDTH-1:0] mean_q;

    // ─────────────────────────────────────────────
    // LATCHED VALUES
    // ─────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] latched_raw;
    logic signed [DATA_WIDTH-1:0] latched_bias;
    logic signed [DATA_WIDTH-1:0] bias_applied;
    logic signed [DATA_WIDTH-1:0] after_relu;

    // Bias + raw MAC addition with saturation
    logic signed [DATA_WIDTH:0] bias_sum;   // one extra bit for overflow detection
    assign bias_sum = $signed(latched_raw) + $signed(latched_bias);

    // Saturate to 32-bit signed
    always_comb begin
        if (bias_sum > 32'sh7FFFFFFF)
            bias_applied = 32'sh7FFFFFFF;
        else if (bias_sum < -32'sh80000000)
            bias_applied = -32'sh80000000;
        else
            bias_applied = bias_sum[DATA_WIDTH-1:0];
    end

    // ReLU — zero negatives
    assign after_relu = ($signed(bias_applied) > 0) ? bias_applied : '0;

    // Normalized output — subtract mean with saturation
    logic signed [DATA_WIDTH:0] norm_sum;
    assign norm_sum = $signed(post_relu[neuron_idx]) - $signed(mean_q);

    logic signed [DATA_WIDTH-1:0] norm_result;
    always_comb begin
        if (norm_sum > 32'sh7FFFFFFF)
            norm_result = 32'sh7FFFFFFF;
        else if (norm_sum < -32'sh80000000)
            norm_result = -32'sh80000000;
        else
            norm_result = norm_sum[DATA_WIDTH-1:0];
    end

    // ─────────────────────────────────────────────
    // MAIN FSM
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            neuron_idx <= '0;
            sum_acc    <= '0;
            mean_q     <= '0;
            done       <= 1'b0;
            buf_clear  <= 1'b0;
            bias_en    <= 1'b0;
            raw_en     <= 1'b0;
            out_we     <= 1'b0;
        end
        else begin
            // Default deassert
            done      <= 1'b0;
            buf_clear <= 1'b0;
            bias_en   <= 1'b0;
            raw_en    <= 1'b0;
            out_we    <= 1'b0;

            case (state)

                IDLE: begin
                    if (start) begin
                        neuron_idx <= '0;
                        sum_acc    <= '0;
                        buf_clear  <= 1'b1;   // clear activation buffer for fresh write
                        state      <= BIAS_READ;
                    end
                end

                // ── PASS 1a: BIAS + RELU ──────────────────────────
                // Read raw MAC result and bias for current neuron
                BIAS_READ: begin
                    raw_addr   <= neuron_idx;
                    raw_en     <= 1'b1;
                    bias_addr  <= neuron_idx;
                    bias_en    <= 1'b1;
                    state      <= BIAS_APPLY;
                end

                // Apply bias and ReLU, store in post_relu[]
                // raw_rdata and bias_rdata valid this cycle
                BIAS_APPLY: begin
                    latched_raw  <= $signed(raw_rdata);
                    latched_bias <= $signed(bias_rdata);
                    // bias_applied and after_relu computed combinatorially
                    // from latched values — will be valid next cycle
                    // Store post-ReLU result
                    post_relu[neuron_idx] <= after_relu;

                    if (neuron_idx == NUM_NEURONS - 1) begin
                        // All neurons processed through ReLU
                        // Move to pass 1b: compute sum for mean
                        neuron_idx <= '0;
                        state      <= SUM_READ;
                    end
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= BIAS_READ;
                    end
                end

                // ── PASS 1b: SUM FOR MEAN ─────────────────────────
                SUM_READ: begin
                    // post_relu is a register file — zero latency read
                    // Accumulate directly
                    sum_acc <= sum_acc + {{16{post_relu[neuron_idx][DATA_WIDTH-1]}},
                                          post_relu[neuron_idx]};

                    if (neuron_idx == NUM_NEURONS - 1) begin
                        // Compute mean: divide sum by NUM_NEURONS
                        // NUM_NEURONS = 256 = 2^8, so arithmetic right-shift by 8
                        // This gives mean in Q16.16
                        mean_q     <= (sum_acc + {{16{post_relu[neuron_idx][DATA_WIDTH-1]}},
                                       post_relu[neuron_idx]}) >>> NEURON_LOG2;
                        neuron_idx <= '0;
                        state      <= NORM_READ;
                    end
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        // Stay in SUM_READ — combinatorial post_relu read
                    end
                end

                // ── PASS 2: NORMALIZE ─────────────────────────────
                // Subtract mean from each post-ReLU value
                // Write result to activation buffer
                NORM_READ: begin
                    // norm_result computed combinatorially from
                    // post_relu[neuron_idx] and mean_q
                    state <= NORM_WRITE;
                end

                NORM_WRITE: begin
                    out_addr  <= neuron_idx;
                    out_we    <= 1'b1;
                    out_wdata <= norm_result;

                    if (neuron_idx == NUM_NEURONS - 1) begin
                        state <= DONE;
                    end
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= NORM_READ;
                    end
                end

                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
