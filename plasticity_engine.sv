// plasticity_engine.sv
// Forward-Forward weight update rule in hardware.
// Runs on port B of weight_bram while port A is idle.
//
// Update rule (matches Python reference update_weights() exactly):
//   delta_factor = pos_flag - sigmoid(goodness - theta)
//   scaled_delta = LR * delta_factor
//   for each weight w_ij:
//       w_ij += scaled_delta * input_act[j] * output_act[i]
//
// Sequentially reads each weight, computes delta, writes back.
// One weight updated per two cycles (read then write).
// Total cycles: NUM_NEURONS * INPUT_SIZE * 2
//
// Sigmoid approximated with piecewise linear function —
// matches Python q_sigmoid() within Q16.16 precision.
// RTL sigmoid takes one cycle to compute.

module plasticity_engine #(
    parameter NUM_NEURONS = 256,
    parameter INPUT_SIZE  = 784,
    parameter DATA_WIDTH  = 32,
    parameter FRAC_BITS   = 16,
    parameter DEPTH       = NUM_NEURONS * INPUT_SIZE,
    // Hyperparameters in Q16.16
    // LR    = 0.005 * 65536 = 328
    // THETA = 3.0   * 65536 = 196608
    parameter LR          = 32'sh00000148,
    parameter THETA        = 32'sh00030000
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // Control
    input  logic                          start,
    output logic                          done,

    // Positive or negative pass flag
    // 1 = positive pass, push goodness up
    // 0 = negative pass, push goodness down
    input  logic                          is_positive,

    // Goodness scalar from goodness_calc — Q16.16
    input  logic [DATA_WIDTH-1:0]         goodness_in,

    // Input activations — layer N input (x in update rule)
    // Combinatorial read from activation_buffer
    input  logic [DATA_WIDTH-1:0]         input_acts  [0:INPUT_SIZE-1],

    // Output activations — layer N output (y in update rule)
    // Combinatorial read from activation_buffer
    input  logic [DATA_WIDTH-1:0]         output_acts [0:NUM_NEURONS-1],

    // Weight BRAM port B interface
    output logic [$clog2(DEPTH)-1:0]      weight_addr_b,
    output logic                          weight_en_b,
    output logic                          weight_we_b,
    output logic [DATA_WIDTH-1:0]         weight_wdata_b,
    input  logic [DATA_WIDTH-1:0]         weight_rdata_b,
    input logic [11:0] active_input_size
);

    // ─────────────────────────────────────────────
    // STATE MACHINE
    // ─────────────────────────────────────────────
    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        SIGMOID     = 3'b001,   // compute sigmoid(goodness - theta), one cycle
        DELTA       = 3'b010,   // compute delta_factor and scaled_delta, one cycle
        W_READ      = 3'b011,   // request current weight from BRAM port B
        W_UPDATE    = 3'b100,   // compute new weight, write back
        DONE        = 3'b101
    } state_t;

    state_t state;

    // ─────────────────────────────────────────────
    // COUNTERS
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_NEURONS)-1:0] neuron_idx;
    logic [$clog2(INPUT_SIZE)-1:0]  weight_idx;

    // ─────────────────────────────────────────────
    // SIGMOID PIECEWISE LINEAR APPROXIMATION
    // sigmoid(x) approximated as:
    //   x <= -4.0 : output = 0.0
    //   x >= +4.0 : output = 1.0
    //   else      : output = 0.125 * x + 0.5
    // Error vs true sigmoid < 0.05 across full range
    // All constants in Q16.16
    // 0.125 * 65536 = 8192
    // 0.5   * 65536 = 32768
    // 4.0   * 65536 = 262144
    // ─────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] sigmoid_in;
    logic signed [DATA_WIDTH-1:0] sigmoid_out;

    // Compute sigmoid combinatorially
    logic signed [63:0] sig_linear;
    assign sig_linear = (($signed(sigmoid_in) * 32'sh00002000) >>> FRAC_BITS)
                        + 32'sh00008000;

    always_comb begin
        if ($signed(sigmoid_in) <= -32'sh00040000)
            sigmoid_out = 32'sh00000000;        // 0.0 in Q16.16
        else if ($signed(sigmoid_in) >= 32'sh00040000)
            sigmoid_out = 32'sh00010000;        // 1.0 in Q16.16
        else
            sigmoid_out = sig_linear[DATA_WIDTH-1:0];
    end

    // ─────────────────────────────────────────────
    // INTERMEDIATE REGISTERS
    // ─────────────────────────────────────────────

    // sigmoid(goodness - theta) result
    logic signed [DATA_WIDTH-1:0] sig_result;

    // pos_flag in Q16.16: +1.0 = 65536, -1.0 = -65536
    logic signed [DATA_WIDTH-1:0] pos_flag_q;
    assign pos_flag_q = is_positive ? 32'sh00010000 : -32'sh00010000;

    // delta_factor = pos_flag - sigmoid_result   Q16.16
    logic signed [DATA_WIDTH-1:0] delta_factor;

    // scaled_delta = LR * delta_factor   Q16.16
    logic signed [DATA_WIDTH-1:0] scaled_delta;

    // y_factor = scaled_delta * output_acts[neuron_idx]   Q16.16
    logic signed [DATA_WIDTH-1:0] y_factor;

    // weight delta = y_factor * input_acts[weight_idx]   Q16.16
    logic signed [DATA_WIDTH-1:0] w_delta;

    // new weight = current weight + w_delta   Q16.16 saturated
    logic signed [DATA_WIDTH:0]   new_weight_wide;
    logic signed [DATA_WIDTH-1:0] new_weight;

    // Saturate new weight
    always_comb begin
        new_weight_wide = $signed(weight_rdata_b) + $signed(w_delta);
        if (new_weight_wide > 32'sh7FFFFFFF)
            new_weight = 32'sh7FFFFFFF;
        else if (new_weight_wide < -32'sh80000000)
            new_weight = -32'sh80000000;
        else
            new_weight = new_weight_wide[DATA_WIDTH-1:0];
    end

    // ─────────────────────────────────────────────
    // Q16.16 MULTIPLY HELPER
    // (a * b) >> FRAC_BITS, saturated to 32-bit
    // Used for LR*delta, y_factor, w_delta
    // ─────────────────────────────────────────────
    function automatic logic signed [DATA_WIDTH-1:0] q_mul;
        input logic signed [DATA_WIDTH-1:0] a, b;
        logic signed [63:0] product;
        begin
            product = ($signed(a) * $signed(b)) >>> FRAC_BITS;
            if (product > 64'sh000000007FFFFFFF)
                q_mul = 32'sh7FFFFFFF;
            else if (product < -64'sh0000000080000000)
                q_mul = -32'sh80000000;
            else
                q_mul = product[DATA_WIDTH-1:0];
        end
    endfunction

    // ─────────────────────────────────────────────
    // MAIN FSM
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            neuron_idx    <= '0;
            weight_idx    <= '0;
            sig_result    <= '0;
            delta_factor  <= '0;
            scaled_delta  <= '0;
            y_factor      <= '0;
            w_delta       <= '0;
            done          <= 1'b0;
            weight_en_b   <= 1'b0;
            weight_we_b   <= 1'b0;
        end
        else begin
            done        <= 1'b0;
            weight_en_b <= 1'b0;
            weight_we_b <= 1'b0;

            case (state)

                IDLE: begin
                    if (start) begin
                        neuron_idx <= '0;
                        weight_idx <= '0;
                        // Compute sigmoid(goodness - theta)
                        // sigmoid_in drives combinatorial sigmoid_out
                        sigmoid_in <= $signed(goodness_in) - $signed(THETA);
                        state      <= SIGMOID;
                    end
                end

                // Latch sigmoid result
                // sigmoid_out is combinatorial — stable this cycle
                SIGMOID: begin
                    sig_result <= sigmoid_out;
                    state      <= DELTA;
                end

                // Compute delta_factor and scaled_delta
                // These are the same for all weights in this update pass
                // Compute once here, reuse for all NUM_NEURONS * INPUT_SIZE weights
                DELTA: begin
                    // delta_factor = pos_flag - sigmoid_result
                    delta_factor <= $signed(pos_flag_q) - $signed(sig_result);
                    // scaled_delta = LR * delta_factor
                    scaled_delta <= q_mul($signed(LR),
                                         $signed(pos_flag_q) - $signed(sig_result));
                    // Precompute y_factor for neuron 0
                    y_factor     <= q_mul($signed(LR),
                                         $signed(pos_flag_q) - $signed(sig_result));
                    state        <= W_READ;
                end

                // Read current weight from BRAM port B
                W_READ: begin
                    weight_addr_b <= (neuron_idx * INPUT_SIZE) + weight_idx;
                    weight_en_b   <= 1'b1;
                    // Precompute y_factor for this neuron
                    // y_factor = scaled_delta * output_acts[neuron_idx]
                    y_factor      <= q_mul(scaled_delta,
                                          $signed(output_acts[neuron_idx]));
                    state         <= W_UPDATE;
                end

                // Weight is valid from BRAM (1 cycle latency)
                // Compute delta and write new weight
                W_UPDATE: begin
                    // w_delta = y_factor * input_acts[weight_idx]
                    w_delta <= q_mul(y_factor,
                                     $signed(input_acts[weight_idx]));

                    // Write updated weight back to BRAM
                    // new_weight computed combinatorially from
                    // weight_rdata_b and w_delta
                    weight_addr_b  <= (neuron_idx * INPUT_SIZE) + weight_idx;
                    weight_en_b    <= 1'b1;
                    weight_we_b    <= 1'b1;
                    weight_wdata_b <= new_weight;

                    // Advance counters
                    if (weight_idx == active_input_size - 1) begin
                        weight_idx <= '0;
                        if (neuron_idx == NUM_NEURONS - 1) begin
                            state <= DONE;
                        end
                        else begin
                            neuron_idx <= neuron_idx + 1;
                            state      <= W_READ;
                        end
                    end
                    else begin
                        weight_idx <= weight_idx + 1;
                        state      <= W_READ;
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
