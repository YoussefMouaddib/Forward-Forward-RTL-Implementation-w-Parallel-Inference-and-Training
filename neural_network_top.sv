// neural_network_top.sv
// Top-level module instantiating and wiring all submodules.
//
// INSTANTIATES:
//   - 2x weight_bram       (layer 1 and layer 2 weights)
//   - 2x activation_buffer (layer 1 and layer 2 activations, with shadow)
//   - 1x input_buffer      (image input, with shadow for PE_L2)
//   - 2x mac_unit          (layer 1 and layer 2 forward pass)
//   - 2x relu_norm         (layer 1 and layer 2 post-MAC processing)
//   - 2x goodness_calc     (layer 1 and layer 2 goodness)
//   - 1x plasticity_engine (time-multiplexed between layers via mux)
//   - 1x training_controller (dual-FSM parallel pipeline)
//
// PARAMETER HIERARCHY:
//   Layer 1: INPUT_SIZE=784,  NUM_NEURONS=256
//   Layer 2: INPUT_SIZE=256,  NUM_NEURONS=256
//
// BRAM PORT ARBITRATION:
//   Port A — MAC unit reads during forward pass
//   Port B — PE writes during weight update
//   Scheduling by construction — never simultaneous on same layer
//
// SHADOW BUFFER MUX:
//   PE reads input_shadow when updating L2
//   PE reads l1_act_shadow when updating L1 or L2
//   pe_layer_sel controls which buffers PE sees

module neural_network_top #(
    parameter L1_INPUT_SIZE  = 784,
    parameter L1_NUM_NEURONS = 256,
    parameter L2_INPUT_SIZE  = 256,
    parameter L2_NUM_NEURONS = 256,
    parameter DATA_WIDTH     = 32,
    parameter FRAC_BITS      = 16,
    parameter NUM_SAMPLES    = 7000,
    parameter L1_DEPTH       = L1_NUM_NEURONS * L1_INPUT_SIZE,
    parameter L2_DEPTH       = L2_NUM_NEURONS * L2_INPUT_SIZE,
    parameter LR             = 32'sh00000148,   // 0.005 in Q16.16
    parameter THETA          = 32'sh00030000    // 3.0   in Q16.16
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        training_done,

    // ── EXTERNAL SAMPLE MEMORY ────────────────────
    // Holds flattened MNIST images in Q16.16
    // Provided by testbench via $readmemh
    output logic [$clog2(NUM_SAMPLES * L1_INPUT_SIZE)-1:0] sample_addr,
    output logic        sample_en,
    input  logic [DATA_WIDTH-1:0] sample_rdata,

    // ── EXTERNAL LABEL MEMORY ─────────────────────
    output logic [$clog2(NUM_SAMPLES)-1:0] label_addr,
    output logic        label_en,
    input  logic [3:0]  label_rdata
);

    // ═════════════════════════════════════════════
    // INTERNAL WIRES — TRAINING CONTROLLER OUTPUTS
    // ═════════════════════════════════════════════

    // Input buffer control
    logic [$clog2(L1_INPUT_SIZE)-1:0] inbuf_waddr;
    logic        inbuf_we;
    logic [DATA_WIDTH-1:0] inbuf_wdata;
    logic        input_shadow_capture;

    // Layer 1 control
    logic        l1_mac_start,   l1_mac_done;
    logic        l1_rn_start,    l1_rn_done;
    logic        l1_shadow_capture;
    logic [0:L1_NUM_NEURONS-1][DATA_WIDTH-1:0] l1_shadow_out ;

    // Layer 2 control
    logic        l2_mac_start,   l2_mac_done;
    logic        l2_rn_start,    l2_rn_done;
    logic        l2_shadow_capture;
    logic [0:L2_NUM_NEURONS-1][DATA_WIDTH-1:0] l2_shadow_out ;

    // Goodness control
    logic        goodness_l1_start, goodness_l1_done;
    logic        goodness_l2_start, goodness_l2_done;
    logic [DATA_WIDTH-1:0] goodness_l1_val;
    logic [DATA_WIDTH-1:0] goodness_l2_val;

    // Plasticity engine control
    logic        pe_start,    pe_done;
    logic        pe_is_positive;
    logic        pe_layer_sel;
    logic [DATA_WIDTH-1:0] pe_goodness;

    // Label outputs
    logic [3:0]  correct_label;
    logic [3:0]  injected_label;
    logic        inject_en;

    // ═════════════════════════════════════════════
    // INPUT BUFFER WITH SHADOW
    // Holds prepared image fed to L1 MAC
    // Shadow read by PE_L2 (needs input activations)
    // ═════════════════════════════════════════════

    // Live input buffer — written by training controller
    logic [0:L1_INPUT_SIZE-1][DATA_WIDTH-1:0] input_buf      ;
    logic [0:L1_INPUT_SIZE-1][DATA_WIDTH-1:0] input_shadow   ;

    // Write to live buffer
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            integer i;
            for (i = 0; i < L1_INPUT_SIZE; i++)
                input_buf[i] <= '0;
        end
        else if (inbuf_we)
            input_buf[inbuf_waddr] <= inbuf_wdata;
    end

    // Shadow capture — registered copy of full buffer
    genvar gi;
    generate
        for (gi = 0; gi < L1_INPUT_SIZE; gi++) begin : input_shadow_gen
            always_ff @(posedge clk) begin
                if (input_shadow_capture)
                    input_shadow[gi] <= input_buf[gi];
            end
        end
    endgenerate

    // ═════════════════════════════════════════════
    // LAYER 1 WEIGHT BRAM
    // Port A — L1 MAC unit reads
    // Port B — PE writes when pe_layer_sel = 0
    // ═════════════════════════════════════════════

    // Port A wires from L1 MAC
    logic [$clog2(L1_DEPTH)-1:0]  l1_weight_addr_a;
    logic                          l1_weight_en_a;
    logic [DATA_WIDTH-1:0]         l1_weight_rdata_a;

    // Port B wires — muxed between PE and idle
    logic [$clog2(L1_DEPTH)-1:0]  l1_weight_addr_b;
    logic                          l1_weight_en_b;
    logic                          l1_weight_we_b;
    logic [DATA_WIDTH-1:0]         l1_weight_wdata_b;
    logic [DATA_WIDTH-1:0]         l1_weight_rdata_b;

    weight_bram #(
        .NUM_NEURONS (L1_NUM_NEURONS),
        .INPUT_SIZE  (L1_INPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .DEPTH       (L1_DEPTH)
    ) l1_wbram (
        .clk_a   (clk),
        .en_a    (l1_weight_en_a),
        .addr_a  (l1_weight_addr_a),
        .rdata_a (l1_weight_rdata_a),
        .clk_b   (clk),
        .en_b    (l1_weight_en_b),
        .we_b    (l1_weight_we_b),
        .addr_b  (l1_weight_addr_b),
        .wdata_b (l1_weight_wdata_b)
    );

    // ═════════════════════════════════════════════
    // LAYER 2 WEIGHT BRAM
    // Port A — L2 MAC unit reads
    // Port B — PE writes when pe_layer_sel = 1
    // ═════════════════════════════════════════════

    logic [$clog2(L2_DEPTH)-1:0]  l2_weight_addr_a;
    logic                          l2_weight_en_a;
    logic [DATA_WIDTH-1:0]         l2_weight_rdata_a;

    logic [$clog2(L2_DEPTH)-1:0]  l2_weight_addr_b;
    logic                          l2_weight_en_b;
    logic                          l2_weight_we_b;
    logic [DATA_WIDTH-1:0]         l2_weight_wdata_b;
    logic [DATA_WIDTH-1:0]         l2_weight_rdata_b;

    weight_bram #(
        .NUM_NEURONS (L2_NUM_NEURONS),
        .INPUT_SIZE  (L2_INPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .DEPTH       (L2_DEPTH)
    ) l2_wbram (
        .clk_a   (clk),
        .en_a    (l2_weight_en_a),
        .addr_a  (l2_weight_addr_a),
        .rdata_a (l2_weight_rdata_a),
        .clk_b   (clk),
        .en_b    (l2_weight_en_b),
        .we_b    (l2_weight_we_b),
        .addr_b  (l2_weight_addr_b),
        .wdata_b (l2_weight_wdata_b)
    );

    // ═════════════════════════════════════════════
    // LAYER 1 ACTIVATION BUFFER WITH SHADOW
    // Live — written by L1 relu_norm
    // Port A — read by L2 MAC unit
    // Port B (shadow) — read by PE
    // ═════════════════════════════════════════════

    // L1 activation buffer write port (from relu_norm)
    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_act_waddr;
    logic        l1_act_we;
    logic [DATA_WIDTH-1:0] l1_act_wdata;
    logic        l1_act_valid;
    logic        l1_act_clear;

    // L1 activation buffer read port A (to L2 MAC)
    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_act_raddr_a;
    logic [DATA_WIDTH-1:0] l1_act_rdata_a;

    // L1 shadow read port B (to PE)
    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_act_raddr_b;
    logic [DATA_WIDTH-1:0] l1_act_rdata_b;

    activation_buffer #(
        .NUM_NEURONS (L1_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH)
    ) l1_act_buf (
        .clk            (clk),
        .rst_n          (rst_n),
        .we             (l1_act_we),
        .waddr          (l1_act_waddr),
        .wdata          (l1_act_wdata),
        .raddr_a        (l1_act_raddr_a),
        .rdata_a        (l1_act_rdata_a),
        .raddr_b        (l1_act_raddr_b),
        .rdata_b        (l1_act_rdata_b),
        .clear          (l1_act_clear),
        .valid          (l1_act_valid),
        .shadow_out (l1_shadow_out)
    );

    // ═════════════════════════════════════════════
    // LAYER 2 ACTIVATION BUFFER WITH SHADOW
    // Live — written by L2 relu_norm
    // Port A — read by goodness_calc_l2
    // Port B (shadow) — read by PE when updating L2
    // ═════════════════════════════════════════════

    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_act_waddr;
    logic        l2_act_we;
    logic [DATA_WIDTH-1:0] l2_act_wdata;
    logic        l2_act_valid;
    logic        l2_act_clear;

    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_act_raddr_a;
    logic [DATA_WIDTH-1:0] l2_act_rdata_a;

    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_act_raddr_b;
    logic [DATA_WIDTH-1:0] l2_act_rdata_b;

    activation_buffer #(
        .NUM_NEURONS (L2_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH)
    ) l2_act_buf (
        .clk            (clk),
        .rst_n          (rst_n),
        .we             (l2_act_we),
        .waddr          (l2_act_waddr),
        .wdata          (l2_act_wdata),
        .raddr_a        (l2_act_raddr_a),
        .rdata_a        (l2_act_rdata_a),
        .raddr_b        (l2_act_raddr_b),
        .rdata_b        (l2_act_rdata_b),
        .clear          (l2_act_clear),
        .valid          (l2_act_valid),
        .shadow_out (l2_shadow_out)
    );

    // ═════════════════════════════════════════════
    // LAYER 1 MAC UNIT
    // Reads from input_buf and l1_wbram port A
    // Writes raw results to intermediate buffer
    // for relu_norm to consume
    // ═════════════════════════════════════════════

    // Intermediate raw MAC output buffer
    // Written by L1 MAC, read by L1 relu_norm
    logic [0:L1_NUM_NEURONS-1][DATA_WIDTH-1:0] l1_raw_buf ;

    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_raw_waddr;
    logic        l1_raw_we;
    logic [DATA_WIDTH-1:0] l1_raw_wdata;

    always_ff @(posedge clk) begin
        if (l1_raw_we)
            l1_raw_buf[l1_raw_waddr] <= l1_raw_wdata;
    end

    mac_unit #(
        .NUM_NEURONS (L1_NUM_NEURONS),
        .INPUT_SIZE  (L1_INPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS)
    ) l1_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (l1_mac_start),
        .done        (l1_mac_done),
        .act_in      (input_buf),
        .weight_addr (l1_weight_addr_a),
        .weight_en   (l1_weight_en_a),
        .weight_rdata(l1_weight_rdata_a),
        .out_addr    (l1_raw_waddr),
        .out_we      (l1_raw_we),
        .out_wdata   (l1_raw_wdata)
    );

    // ═════════════════════════════════════════════
    // LAYER 1 RELU NORM
    // Reads raw MAC results and bias
    // Writes normalized activations to l1_act_buf
    // ═════════════════════════════════════════════

    // L1 bias ROM — initialized from b1.mem in testbench
    logic [0:L1_NUM_NEURONS-1][DATA_WIDTH-1:0] l1_bias ;

    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_rn_bias_addr;
    logic        l1_rn_bias_en;
    logic [DATA_WIDTH-1:0] l1_rn_bias_rdata;
    logic [$clog2(L1_NUM_NEURONS)-1:0] l1_rn_raw_addr;
    logic        l1_rn_raw_en;
    logic [DATA_WIDTH-1:0] l1_rn_raw_rdata;

    // Bias read — combinatorial from array
    assign l1_rn_bias_rdata = l1_bias[l1_rn_bias_addr];
    // Raw MAC read — combinatorial from array
    assign l1_rn_raw_rdata  = l1_raw_buf[l1_rn_raw_addr];

    relu_norm #(
        .NUM_NEURONS (L1_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS),
        .NEURON_LOG2 (8)
    ) l1_rn (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (l1_rn_start),
        .done        (l1_rn_done),
        .bias_addr   (l1_rn_bias_addr),
        .bias_en     (l1_rn_bias_en),
        .bias_rdata  (l1_rn_bias_rdata),
        .raw_addr    (l1_rn_raw_addr),
        .raw_en      (l1_rn_raw_en),
        .raw_rdata   (l1_rn_raw_rdata),
        .out_addr    (l1_act_waddr),
        .out_we      (l1_act_we),
        .out_wdata   (l1_act_wdata),
        .buf_clear   (l1_act_clear)
    );

    // ═════════════════════════════════════════════
    // LAYER 2 MAC UNIT
    // Reads from l1_act_buf port A and l2_wbram port A
    // Writes raw results to l2_raw_buf
    // ═════════════════════════════════════════════

    logic [0:L2_NUM_NEURONS-1][DATA_WIDTH-1:0] l2_raw_buf ;
    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_raw_waddr;
    logic        l2_raw_we;
    logic [DATA_WIDTH-1:0] l2_raw_wdata;

    always_ff @(posedge clk) begin
        if (l2_raw_we)
            l2_raw_buf[l2_raw_waddr] <= l2_raw_wdata;
    end

    // L2 MAC reads l1 activations as its input
    // l1_act_buf exposed as flat array for MAC act_in port
    // Port A address driven by L2 MAC counter internally
    // We expose the full array — MAC indexes into it
    mac_unit #(
        .NUM_NEURONS (L2_NUM_NEURONS),
        .INPUT_SIZE  (L2_INPUT_SIZE),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS)
    ) l2_mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (l2_mac_start),
        .done        (l2_mac_done),
        .act_in      (l2_shadow_out),   
        .weight_addr (l2_weight_addr_a),
        .weight_en   (l2_weight_en_a),
        .weight_rdata(l2_weight_rdata_a),
        .out_addr    (l2_raw_waddr),
        .out_we      (l2_raw_we),
        .out_wdata   (l2_raw_wdata)
    );

    // ═════════════════════════════════════════════
    // LAYER 2 RELU NORM
    // ═════════════════════════════════════════════

    logic [0:L2_NUM_NEURONS-1][DATA_WIDTH-1:0] l2_bias ;

    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_rn_bias_addr;
    logic        l2_rn_bias_en;
    logic [DATA_WIDTH-1:0] l2_rn_bias_rdata;
    logic [$clog2(L2_NUM_NEURONS)-1:0] l2_rn_raw_addr;
    logic        l2_rn_raw_en;
    logic [DATA_WIDTH-1:0] l2_rn_raw_rdata;

    assign l2_rn_bias_rdata = l2_bias[l2_rn_bias_addr];
    assign l2_rn_raw_rdata  = l2_raw_buf[l2_rn_raw_addr];

    relu_norm #(
        .NUM_NEURONS (L2_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS),
        .NEURON_LOG2 (8)
    ) l2_rn (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (l2_rn_start),
        .done        (l2_rn_done),
        .bias_addr   (l2_rn_bias_addr),
        .bias_en     (l2_rn_bias_en),
        .bias_rdata  (l2_rn_bias_rdata),
        .raw_addr    (l2_rn_raw_addr),
        .raw_en      (l2_rn_raw_en),
        .raw_rdata   (l2_rn_raw_rdata),
        .out_addr    (l2_act_waddr),
        .out_we      (l2_act_we),
        .out_wdata   (l2_act_wdata),
        .buf_clear   (l2_act_clear)
    );

    // ═════════════════════════════════════════════
    // GOODNESS CALCULATORS
    // Each takes its layer's shadow activations
    // Exposes flat array interface to goodness_calc
    // ═════════════════════════════════════════════

    // Expose l1 and l2 shadow buffers as flat arrays
    // goodness_calc reads act_data[neuron_idx] combinatorially
    // Shadow buffers are register files — direct array access fine

    // L1 goodness
    goodness_calc #(
        .NUM_NEURONS (L1_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS)
    ) goodness_l1 (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (goodness_l1_start),
        .done        (goodness_l1_done),
        .act_data    (l1_shadow_out),  // read shadow directly
        .goodness_out(goodness_l1_val)
    );

    // L2 goodness
    goodness_calc #(
        .NUM_NEURONS (L2_NUM_NEURONS),
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS)
    ) goodness_l2 (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (goodness_l2_start),
        .done        (goodness_l2_done),
        .act_data    (l2_shadow_out),
        .goodness_out(goodness_l2_val)
    );

    // ═════════════════════════════════════════════
    // PLASTICITY ENGINE MUX
    // pe_layer_sel = 0: update layer 1 weights
    // pe_layer_sel = 1: update layer 2 weights
    //
    // When updating L1:
    //   input_acts  = input_shadow (image that fed L1)
    //   output_acts = l1_shadow_out
    //   weight BRAM port B = l1_wbram
    //
    // When updating L2:
    //   input_acts  = l1_shadow_out (L1 output that fed L2)
    //   output_acts = l2_shadow_out
    //   weight BRAM port B = l2_wbram
    // ═════════════════════════════════════════════

    // PE weight BRAM port B wires
    logic [$clog2(L1_DEPTH)-1:0]  pe_weight_addr;
    logic                          pe_weight_en;
    logic                          pe_weight_we;
    logic [DATA_WIDTH-1:0]         pe_weight_wdata;
    logic [DATA_WIDTH-1:0]         pe_weight_rdata;
    logic [11:0] pe_active_input_size;

    assign pe_active_input_size = pe_layer_sel ? 
                              L2_INPUT_SIZE[($clog2(L1_INPUT_SIZE)-1):0] : 
                              L1_INPUT_SIZE[($clog2(L1_INPUT_SIZE)-1):0];
    
    // Route PE BRAM port B to correct layer
    always_comb begin
        // Default tie-off
        l1_weight_addr_b  = '0;
        l1_weight_en_b    = 1'b0;
        l1_weight_we_b    = 1'b0;
        l1_weight_wdata_b = '0;
        l2_weight_addr_b  = '0;
        l2_weight_en_b    = 1'b0;
        l2_weight_we_b    = 1'b0;
        l2_weight_wdata_b = '0;
        pe_weight_rdata   = '0;

        if (!pe_layer_sel) begin
            // PE → Layer 1 BRAM port B
            l1_weight_addr_b  = pe_weight_addr[$clog2(L1_DEPTH)-1:0];
            l1_weight_en_b    = pe_weight_en;
            l1_weight_we_b    = pe_weight_we;
            l1_weight_wdata_b = pe_weight_wdata;
            pe_weight_rdata   = l1_weight_rdata_b;
        end
        else begin
            // PE → Layer 2 BRAM port B
            l2_weight_addr_b  = pe_weight_addr[$clog2(L2_DEPTH)-1:0];
            l2_weight_en_b    = pe_weight_en;
            l2_weight_we_b    = pe_weight_we;
            l2_weight_wdata_b = pe_weight_wdata;
            pe_weight_rdata   = l2_weight_rdata_b;
        end
    end

    // PE input_acts and output_acts mux
    // PE sees flat arrays — mux selects which shadow to expose
    logic [0:L1_INPUT_SIZE-1][DATA_WIDTH-1:0] pe_input_acts  ;
    logic [0:L2_NUM_NEURONS-1][DATA_WIDTH-1:0] pe_output_acts ;

    // Input acts mux: L1 update needs image input, L2 update needs L1 acts
    generate
        for (gi = 0; gi < L1_INPUT_SIZE; gi++) begin : pe_inact_mux
            always_comb begin
                if (!pe_layer_sel)
                    pe_input_acts[gi] = input_shadow[gi];
                else
                    // L2 update: input to L2 was L1 output
                    // L1 has 256 neurons, L2 input size is 256
                    // Only index 0..255 valid here
                    pe_input_acts[gi] = (gi < L2_INPUT_SIZE) ?
                                        l1_shadow_out[gi] :
                                        '0;
            end
        end
    endgenerate

    // Output acts mux: always the updating layer's output shadow
    generate
        for (gi = 0; gi < L2_NUM_NEURONS; gi++) begin : pe_outact_mux
            always_comb begin
                if (!pe_layer_sel)
                    pe_output_acts[gi] = l1_shadow_out[gi];
                else
                    pe_output_acts[gi] = l2_shadow_out[gi];
            end
        end
    endgenerate

    // ═════════════════════════════════════════════
    // PLASTICITY ENGINE INSTANTIATION
    // ═════════════════════════════════════════════

    plasticity_engine #(
        .NUM_NEURONS (L1_NUM_NEURONS),   // max of L1/L2, same here
        .INPUT_SIZE  (L1_INPUT_SIZE),    // overridden by pe_layer_sel mux
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (FRAC_BITS),
        .LR          (LR),
        .THETA       (THETA)
    ) pe (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (pe_start),
        .done           (pe_done),
        .is_positive    (pe_is_positive),
        .goodness_in    (pe_goodness),
        .input_acts     (pe_input_acts),
        .output_acts    (pe_output_acts),
        .weight_addr_b  (pe_weight_addr),
        .weight_en_b    (pe_weight_en),
        .weight_we_b    (pe_weight_we),
        .weight_wdata_b (pe_weight_wdata),
        .weight_rdata_b (pe_weight_rdata),
        .active_input_size (pe_active_input_size)
    );

    // ═════════════════════════════════════════════
    // TRAINING CONTROLLER INSTANTIATION
    // ═════════════════════════════════════════════

    training_controller #(
        .NUM_SAMPLES  (NUM_SAMPLES),
        .INPUT_SIZE   (L1_INPUT_SIZE),
        .LABEL_PIXELS (10),
        .DATA_WIDTH   (DATA_WIDTH),
        .FRAC_BITS    (FRAC_BITS)
    ) ctrl (
        .clk                  (clk),
        .rst_n                (rst_n),
        .start                (start),
        .training_done        (training_done),
        .sample_addr          (sample_addr),
        .sample_en            (sample_en),
        .sample_rdata         (sample_rdata),
        .label_addr           (label_addr),
        .label_en             (label_en),
        .label_rdata          (label_rdata),
        .inbuf_waddr          (inbuf_waddr),
        .inbuf_we             (inbuf_we),
        .inbuf_wdata          (inbuf_wdata),
        .input_shadow_capture (input_shadow_capture),
        .l1_mac_start         (l1_mac_start),
        .l1_mac_done          (l1_mac_done),
        .l1_rn_start          (l1_rn_start),
        .l1_rn_done           (l1_rn_done),
        .l1_shadow_capture    (l1_shadow_capture),
        .l2_mac_start         (l2_mac_start),
        .l2_mac_done          (l2_mac_done),
        .l2_rn_start          (l2_rn_start),
        .l2_rn_done           (l2_rn_done),
        .l2_shadow_capture    (l2_shadow_capture),
        .goodness_l1_start    (goodness_l1_start),
        .goodness_l1_done     (goodness_l1_done),
        .goodness_l1_val      (goodness_l1_val),
        .goodness_l2_start    (goodness_l2_start),
        .goodness_l2_done     (goodness_l2_done),
        .goodness_l2_val      (goodness_l2_val),
        .pe_start             (pe_start),
        .pe_done              (pe_done),
        .pe_is_positive       (pe_is_positive),
        .pe_layer_sel         (pe_layer_sel),
        .pe_goodness          (pe_goodness),
        .correct_label        (correct_label),
        .injected_label       (injected_label),
        .inject_en            (inject_en)
    );

endmodule
