// mac_unit.sv
// Multiply-accumulate unit for one layer.
// Iterates through all INPUT_SIZE weights for one neuron sequentially,
// then moves to the next neuron. Processes all NUM_NEURONS neurons
// before asserting done.
//
// One MAC operation per clock cycle.
// Total cycles per full layer forward pass: NUM_NEURONS * INPUT_SIZE
//
// RTL equivalent of forward_layer() inner loop in Python reference.
// Q16.16 arithmetic: multiply is 32x32->64, right-shift 16, truncate to 32.

module mac_unit #(
    parameter NUM_NEURONS  = 256,
    parameter INPUT_SIZE   = 784,
    parameter DATA_WIDTH   = 32,
    parameter FRAC_BITS    = 16,
    parameter DEPTH        = NUM_NEURONS * INPUT_SIZE
)(
    input  logic                         clk,
    input  logic                         rst_n,

    // Control
    input  logic                         start,       // pulse high to begin forward pass
    output logic                         done,        // pulses high when all neurons complete

    // Input activation buffer — from previous layer or input image
    // Addressed by weight_idx during MAC, read combinatorially
    input  logic [0:INPUT_SIZE-1][DATA_WIDTH-1:0]        act_in  ,

    // Weight BRAM port A interface
    output logic [$clog2(DEPTH)-1:0]     weight_addr,
    output logic                         weight_en,
    input  logic [DATA_WIDTH-1:0]        weight_rdata,

    // Output activation buffer write interface
    // Written once per neuron when accumulation is complete
    output logic [$clog2(NUM_NEURONS)-1:0] out_addr,
    output logic                           out_we,
    output logic [DATA_WIDTH-1:0]          out_wdata
);

    // ─────────────────────────────────────────────
    // STATE MACHINE
    // ─────────────────────────────────────────────
    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        LOAD    = 2'b01,    // request weight from BRAM, latch input activation
        MAC     = 2'b10,    // multiply and accumulate (one cycle after LOAD due to BRAM latency)
        WRITE   = 2'b11     // write neuron result to output buffer, move to next neuron
    } state_t;

    state_t state;

    // ─────────────────────────────────────────────
    // COUNTERS
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_NEURONS)-1:0] neuron_idx;   // which neuron we are computing
    logic [$clog2(INPUT_SIZE)-1:0]  weight_idx;   // which weight within that neuron

    // ─────────────────────────────────────────────
    // ACCUMULATOR
    // 64-bit to hold full Q32.32 product before shift
    // ─────────────────────────────────────────────
    logic signed [63:0] accumulator;

    // ─────────────────────────────────────────────
    // LATCHED INPUT ACTIVATION
    // Latch act_in[weight_idx] one cycle before MAC
    // so it is stable when weight_rdata arrives from BRAM
    // ─────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] latched_act;

    // ─────────────────────────────────────────────
    // Q16.16 MULTIPLY
    // product = (weight * activation) >> FRAC_BITS
    // Both inputs are signed 32-bit Q16.16
    // Product is signed 64-bit before shift
    // ─────────────────────────────────────────────
    logic signed [63:0] product;
    assign product = ($signed(weight_rdata) * $signed(latched_act)) >>> FRAC_BITS;

    // ─────────────────────────────────────────────
    // SATURATION HELPER
    // Clip 64-bit accumulator to signed 32-bit range
    // before writing to output buffer
    // ─────────────────────────────────────────────
    logic signed [DATA_WIDTH-1:0] sat_result;
    always_comb begin
        if (accumulator > 64'sh000000007FFFFFFF)
            sat_result = 32'sh7FFFFFFF;
        else if (accumulator < -64'sh0000000080000000)
            sat_result = 32'sh80000000;
        else
            sat_result = accumulator[DATA_WIDTH-1:0];
    end

    // ─────────────────────────────────────────────
    // MAIN FSM
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            neuron_idx  <= '0;
            weight_idx  <= '0;
            accumulator <= '0;
            latched_act <= '0;
            done        <= 1'b0;
            weight_en   <= 1'b0;
            out_we      <= 1'b0;
        end
        else begin
            // Default outputs — deassert every cycle unless explicitly set
            done      <= 1'b0;
            out_we    <= 1'b0;
            weight_en <= 1'b0;

            case (state)

                IDLE: begin
                    if (start) begin
                        neuron_idx  <= '0;
                        weight_idx  <= '0;
                        accumulator <= '0;
                        state       <= LOAD;
                    end
                end

                LOAD: begin
                    // Request weight from BRAM
                    // Address = neuron_idx * INPUT_SIZE + weight_idx
                    weight_addr <= (neuron_idx * INPUT_SIZE) + weight_idx;
                    weight_en   <= 1'b1;
                    // Latch the input activation for this weight index
                    latched_act <= $signed(act_in[weight_idx]);
                    state       <= MAC;
                end

                MAC: begin
                    // weight_rdata is valid now (1 cycle BRAM latency)
                    // Accumulate product into accumulator
                    accumulator <= accumulator + product;

                    if (weight_idx == INPUT_SIZE - 1) begin
                        // Last weight for this neuron — go write result
                        state <= WRITE;
                    end
                    else begin
                        // More weights — increment and go back to LOAD
                        weight_idx <= weight_idx + 1;
                        state      <= LOAD;
                    end
                end

                WRITE: begin
                    // Write saturated accumulator to output activation buffer
                    out_addr  <= neuron_idx;
                    out_we    <= 1'b1;
                    out_wdata <= sat_result;

                    // Reset accumulator for next neuron
                    accumulator <= '0;
                    weight_idx  <= '0;

                    if (neuron_idx == NUM_NEURONS - 1) begin
                        // All neurons done
                        done       <= 1'b1;
                        state      <= IDLE;
                        neuron_idx <= '0;
                    end
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= LOAD;
                    end
                end

            endcase
        end
    end

    // Weight address driven combinatorially in LOAD state
    // already assigned inside always_ff which is fine for registered output

endmodule
