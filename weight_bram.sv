// weight_bram.sv
// Dual-port weight memory for one layer.
// Port A: read-only  — MAC unit reads weights during forward pass
// Port B: write-only — plasticity engine writes updated weights
//
// Address scheme: addr = neuron_idx * INPUT_SIZE + weight_idx
// Total depth: NUM_NEURONS * INPUT_SIZE
// Width: 32 bits (Q16.16)
//
// Both ports are synchronous. Read latency = 1 clock cycle.

module weight_bram #(
    parameter NUM_NEURONS = 256,
    parameter INPUT_SIZE  = 784,
    parameter DATA_WIDTH  = 32,
    parameter DEPTH       = NUM_NEURONS * INPUT_SIZE
)(
    // Port A — MAC unit (read only)
    input  logic                          clk_a,
    input  logic                          en_a,
    input  logic [$clog2(DEPTH)-1:0]      addr_a,
    output logic [DATA_WIDTH-1:0]         rdata_a,

    // Port B — plasticity engine (write only)
    input  logic                          clk_b,
    input  logic                          en_b,
    input  logic                          we_b,
    input  logic [$clog2(DEPTH)-1:0]      addr_b,
    input  logic [DATA_WIDTH-1:0]         wdata_b
);

    // Memory array — tools infer BRAM from this pattern
    // Xilinx: infers RAMB36E2
    // Intel:  infers M20K blocks
    logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Port A — synchronous read
    always_ff @(posedge clk_a) begin
        if (en_a)
            rdata_a <= mem[addr_a];
    end

    // Port B — synchronous write
    always_ff @(posedge clk_b) begin
        if (en_b && we_b)
            mem[addr_b] <= wdata_b;
    end

endmodule
