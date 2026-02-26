`timescale 1ns/1ps

module tb_top;

    // ─────────────────────────────────────────────
    // PARAMETERS
    // ─────────────────────────────────────────────
    localparam L1_INPUT_SIZE  = 784;
    localparam L1_NUM_NEURONS = 256;
    localparam L2_INPUT_SIZE  = 256;
    localparam L2_NUM_NEURONS = 256;
    localparam DATA_WIDTH     = 32;
    localparam FRAC_BITS      = 16;
    localparam NUM_SAMPLES    = 100;
    localparam L1_DEPTH       = L1_NUM_NEURONS * L1_INPUT_SIZE;
    localparam L2_DEPTH       = L2_NUM_NEURONS * L2_INPUT_SIZE;

    // ─────────────────────────────────────────────
    // CLOCK
    // ─────────────────────────────────────────────
    logic clk;
    initial clk = 0;
    always #5 clk = ~clk;

    // ─────────────────────────────────────────────
    // DUT SIGNALS
    // ─────────────────────────────────────────────
    logic rst_n;
    logic start;
    logic training_done;

    logic [$clog2(NUM_SAMPLES * L1_INPUT_SIZE)-1:0] sample_addr;
    logic        sample_en;
    logic [DATA_WIDTH-1:0] sample_rdata;
    logic [$clog2(NUM_SAMPLES)-1:0] label_addr;
    logic        label_en;
    logic [3:0]  label_rdata;

    // ─────────────────────────────────────────────
    // TESTBENCH MEMORIES
    // ─────────────────────────────────────────────
    logic [DATA_WIDTH-1:0] sample_mem [0:(NUM_SAMPLES*L1_INPUT_SIZE)-1];
    logic [3:0]            label_mem  [0:NUM_SAMPLES-1];

    // Synchronous memory read — matches DUT expectation
    always_ff @(posedge clk) begin
        if (sample_en)
            sample_rdata <= sample_mem[sample_addr];
        if (label_en)
            label_rdata <= label_mem[label_addr];
    end

    // ─────────────────────────────────────────────
    // DUT
    // ─────────────────────────────────────────────
    neural_network_top #(
        .L1_INPUT_SIZE  (L1_INPUT_SIZE),
        .L1_NUM_NEURONS (L1_NUM_NEURONS),
        .L2_INPUT_SIZE  (L2_INPUT_SIZE),
        .L2_NUM_NEURONS (L2_NUM_NEURONS),
        .DATA_WIDTH     (DATA_WIDTH),
        .FRAC_BITS      (FRAC_BITS),
        .NUM_SAMPLES    (NUM_SAMPLES)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .training_done(training_done),
        .sample_addr  (sample_addr),
        .sample_en    (sample_en),
        .sample_rdata (sample_rdata),
        .label_addr   (label_addr),
        .label_en     (label_en),
        .label_rdata  (label_rdata)
    );

    // ─────────────────────────────────────────────
    // GOODNESS MONITOR
    // ─────────────────────────────────────────────
    function real q_to_real(input logic signed [DATA_WIDTH-1:0] q);
        q_to_real = $itor($signed(q)) / 65536.0;
    endfunction

    logic prev_goodness_l2_done;
    integer sample_count;
    real pos_g;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            prev_goodness_l2_done <= 0;
            sample_count          <= 0;
            pos_g                 <= 0.0;
        end
        else begin
            prev_goodness_l2_done <= dut.goodness_l2_done;

            if (dut.goodness_l2_done && !prev_goodness_l2_done) begin
                if (dut.ctrl.is_positive_pass)
                    pos_g <= q_to_real(dut.goodness_l2_val);
                else begin
                    $display("[TB] Sample %4d | Pos: %7.3f | Neg: %7.3f",
                             sample_count,
                             pos_g,
                             q_to_real(dut.goodness_l2_val));
                    sample_count <= sample_count + 1;
                end
            end
        end
    end

    // ─────────────────────────────────────────────
    // PARALLEL DATAPATH MONITOR
    // ─────────────────────────────────────────────
    longint pe_l1_start_time;
    logic   pe_l1_running;
    logic   overlap_confirmed;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pe_l1_start_time  <= 0;
            pe_l1_running     <= 0;
            overlap_confirmed <= 0;
        end
        else begin
            // Track when PE starts on layer 1
            if (dut.pe_start && !dut.pe_layer_sel) begin
                pe_l1_start_time <= $time;
                pe_l1_running    <= 1;
            end
            if (dut.pe_done && !dut.pe_layer_sel)
                pe_l1_running <= 0;

            // If L2 MAC starts while PE_L1 is running — parallel confirmed
            if (dut.l2_mac_start && pe_l1_running && !overlap_confirmed) begin
                $display("[TB] PARALLEL CONFIRMED: L2_MAC and PE_L1 overlap at %0t ns", $time);
                overlap_confirmed <= 1;
            end
        end
    end

    // ─────────────────────────────────────────────
    // MAIN STIMULUS
    // ─────────────────────────────────────────────
    integer i;
    logic [DATA_WIDTH-1:0] w1_before;

    initial begin
        $display("================================================");
        $display("  Forward-Forward RTL Testbench");
        $display("  Samples: %0d | Clock: 100MHz", NUM_SAMPLES);
        $display("================================================");

        // Initialize control signals
        rst_n = 0;
        start = 0;

        // ── INITIALIZE WEIGHTS ────────────────────
        // Alternating pattern to break symmetry
        for (i = 0; i < L1_DEPTH; i++) begin
            case (i % 4)
                0: dut.l1_wbram.mem[i] = 32'sh00000200;
                1: dut.l1_wbram.mem[i] = -32'sh00000180;
                2: dut.l1_wbram.mem[i] = 32'sh00000100;
                3: dut.l1_wbram.mem[i] = -32'sh00000280;
            endcase
        end
        for (i = 0; i < L2_DEPTH; i++) begin
            case (i % 4)
                0: dut.l2_wbram.mem[i] = 32'sh00000180;
                1: dut.l2_wbram.mem[i] = -32'sh00000200;
                2: dut.l2_wbram.mem[i] = 32'sh00000280;
                3: dut.l2_wbram.mem[i] = -32'sh00000100;
            endcase
        end
        for (i = 0; i < L1_NUM_NEURONS; i++)
            dut.l1_bias[i] = 32'sh00000000;
        for (i = 0; i < L2_NUM_NEURONS; i++)
            dut.l2_bias[i] = 32'sh00000000;

        // ── INITIALIZE SAMPLES ────────────────────
        for (i = 0; i < NUM_SAMPLES * L1_INPUT_SIZE; i++)
            sample_mem[i] = (i % 2 == 0) ? 32'sh00008000 : 32'sh00003000;
        for (i = 0; i < NUM_SAMPLES; i++)
            label_mem[i] = i % 10;

        $display("[TB] Memories initialized.");

        // Record spot weight before training
        w1_before = dut.l1_wbram.mem[100];
        $display("[TB] L1[100] before = %h (%.4f)",
                 w1_before, q_to_real(w1_before));

        // ── RESET ─────────────────────────────────
        #20;
        rst_n = 1;
        #50;

        // ── START ─────────────────────────────────
        $display("[TB] Asserting start...");
        start = 1;
        #10;
        start = 0;

        // ── WAIT ──────────────────────────────────
        // 500ms sim time — enough for 100 samples
        // Increase if needed
        $display("[TB] Running — waiting for training_done...");
        #500_000_000;

        // ── RESULTS ───────────────────────────────
        $display("");
        $display("================================================");
        $display("  Results");
        $display("================================================");

        $display("[TB] L1[100] after  = %h (%.4f)",
                 dut.l1_wbram.mem[100],
                 q_to_real(dut.l1_wbram.mem[100]));

        if (dut.l1_wbram.mem[100] !== w1_before)
            $display("[TB] PASS: L1 weights changed — learning confirmed.");
        else
            $display("[TB] FAIL: L1 weights unchanged.");

        if (training_done)
            $display("[TB] PASS: training_done received.");
        else
            $display("[TB] FAIL: training_done never arrived — FSM stuck.");

        if (overlap_confirmed)
            $display("[TB] PASS: Dual-datapath parallelism verified.");
        else
            $display("[TB] INFO: Parallel overlap not observed — check waveform.");

        $display("================================================");
        $finish;
    end

endmodule
