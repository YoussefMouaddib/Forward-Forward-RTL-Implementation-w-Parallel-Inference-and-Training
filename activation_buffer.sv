// activation_buffer.sv
// Stores one full layer's post-ReLU post-normalization activations.
// Single write port — written by relu_norm unit one neuron at a time.
// Dual read port — read by next layer's MAC unit AND plasticity engine.
//
// Implemented as register file not BRAM because:
// 1. Small size — 256 x 32 bits = 8KB, fits in registers on Arty7
// 2. Needs simultaneous read from two masters without arbitration
// 3. Zero read latency needed for MAC unit performance
//
// RTL equivalent of pos_activations[] and neg_activations[] lists
// in Python reference train() function.

module activation_buffer #(
    parameter NUM_NEURONS = 256,
    parameter DATA_WIDTH  = 32
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Write port — from relu_norm unit
    input  logic                              we,
    input  logic [$clog2(NUM_NEURONS)-1:0]    waddr,
    input  logic [DATA_WIDTH-1:0]             wdata,

    // Read port A — next layer MAC unit
    // Combinatorial read, zero latency
    input  logic [$clog2(NUM_NEURONS)-1:0]    raddr_a,
    output logic [DATA_WIDTH-1:0]             rdata_a,

    // Read port B — plasticity engine
    // Combinatorial read, zero latency
    input  logic [$clog2(NUM_NEURONS)-1:0]    raddr_b,
    output logic [DATA_WIDTH-1:0]             rdata_b,

    // Valid flag — asserted when buffer holds complete layer output
    // Deasserted at start of new forward pass
    input  logic                              clear,
    output logic                              valid,
    output logic [DATA_WIDTH-1:0] shadow_out [0:NUM_NEURONS-1]
);

    // ─────────────────────────────────────────────
    // REGISTER FILE
    // 256 x 32-bit registers
    // Tools will implement as distributed RAM or flip-flops
    // ─────────────────────────────────────────────
    logic [DATA_WIDTH-1:0] buf [0:NUM_NEURONS-1];

    // Write port — synchronous
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all activations to zero
            integer i;
            for (i = 0; i < NUM_NEURONS; i++)
                buf[i] <= '0;
        end
        else if (we) begin
            buf[waddr] <= wdata;
        end
    end

    // Read port A — combinatorial, zero latency
    // MAC unit needs this because it reads act_in[weight_idx]
    // directly as an array — see mac_unit.sv act_in port
    assign rdata_a = buf[raddr_a];

    // Read port B — combinatorial, zero latency
    assign rdata_b = buf[raddr_b];

    assign shadow_out = buf;

    // ─────────────────────────────────────────────
    // VALID TRACKING
    // Counts writes — valid goes high when all NUM_NEURONS
    // activations have been written after a forward pass
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_NEURONS):0] write_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_count <= '0;
            valid       <= 1'b0;
        end
        else if (clear) begin
            write_count <= '0;
            valid       <= 1'b0;
        end
        else if (we) begin
            if (write_count < NUM_NEURONS)
                write_count <= write_count + 1;
            if (write_count == NUM_NEURONS - 1)
                valid <= 1'b1;
        end
    end

endmodule
