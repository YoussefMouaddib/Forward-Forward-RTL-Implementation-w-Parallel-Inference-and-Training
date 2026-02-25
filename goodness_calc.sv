// goodness_calc.sv
// Computes goodness scalar from one layer's normalized activations.
// goodness = sum of squared activations across all neurons
//
// Matches Python reference compute_goodness() exactly:
//   squared = q_mul(activations_q, activations_q)
//   goodness = sum(squared)
//
// Q16.16 squaring: (a * a) >> FRAC_BITS
// Sum accumulated in 64-bit register to prevent overflow.
// Output saturated to 32-bit Q16.16.
//
// Triggered by activation_buffer valid signal going high.
// Completes in NUM_NEURONS + 2 cycles.

module goodness_calc #(
    parameter NUM_NEURONS = 256,
    parameter DATA_WIDTH  = 32,
    parameter FRAC_BITS   = 16
)(
    input  logic                              clk,
    input  logic                              rst_n,

    // Control
    input  logic                              start,
    output logic                              done,

    // Activation buffer read port
    // Combinatorial read — zero latency
    input  logic [0:NUM_NEURONS-1][DATA_WIDTH-1:0]             act_data ,

    // Goodness output — valid when done pulses
    output logic [DATA_WIDTH-1:0]             goodness_out
);

    // ─────────────────────────────────────────────
    // STATE MACHINE
    // ─────────────────────────────────────────────
    typedef enum logic [1:0] {
        IDLE     = 2'b00,
        COMPUTE  = 2'b01,   // square and accumulate one neuron per cycle
        SATURATE = 2'b10,   // clip 64-bit sum to 32-bit output
        DONE     = 2'b11
    } state_t;

    state_t state;

    // ─────────────────────────────────────────────
    // COUNTER
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_NEURONS)-1:0] neuron_idx;

    // ─────────────────────────────────────────────
    // SUM ACCUMULATOR
    // Each squared term is up to (4.9 * 65536)^2 >> 16
    // = (321126)^2 >> 16 = 1.57e9 — fits in 32 bits but
    // summing 256 of these needs 64 bits
    // ─────────────────────────────────────────────
    logic signed [63:0] sum_acc;

    // ─────────────────────────────────────────────
    // Q16.16 SQUARE
    // One activation squared per cycle
    // product = (act * act) >> FRAC_BITS
    // ─────────────────────────────────────────────
    logic signed [63:0] square;
    assign square = ($signed(act_data[neuron_idx]) *
                     $signed(act_data[neuron_idx])) >>> FRAC_BITS;

    // ─────────────────────────────────────────────
    // SATURATION TO 32-BIT OUTPUT
    // ─────────────────────────────────────────────
    always_comb begin
        if (sum_acc > 64'sh000000007FFFFFFF)
            goodness_out = 32'sh7FFFFFFF;
        else if (sum_acc < -64'sh0000000080000000)
            goodness_out = 32'sh80000000;
        else
            goodness_out = sum_acc[DATA_WIDTH-1:0];
    end

    // ─────────────────────────────────────────────
    // MAIN FSM
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            neuron_idx <= '0;
            sum_acc    <= '0;
            done       <= 1'b0;
        end
        else begin
            done <= 1'b0;

            case (state)

                IDLE: begin
                    if (start) begin
                        neuron_idx <= '0;
                        sum_acc    <= '0;
                        state      <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // square is combinatorial from act_data[neuron_idx]
                    // accumulate this cycle
                    sum_acc <= sum_acc + square;

                    if (neuron_idx == NUM_NEURONS - 1) begin
                        state <= SATURATE;
                    end
                    else begin
                        neuron_idx <= neuron_idx + 1;
                    end
                end

                SATURATE: begin
                    // goodness_out driven combinatorially from sum_acc
                    // just need one cycle for it to settle before done
                    state <= DONE;
                end

                DONE: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule
