// training_controller.sv
// Dual-FSM parallel pipelined Forward-Forward training controller.
//
// ARCHITECTURE:
// Two independent FSMs communicating via handshake signals.
//
// FORWARD FSM:
//   Sequences inference pipeline continuously.
//   L1_MAC → L1_RN → L2_MAC → L2_RN → repeat next sample.
//   Produces handshake pulses when layer data is ready for PE.
//   Runs one sample ahead of update FSM.
//
// UPDATE FSM:
//   Sequences plasticity engine.
//   Waits for l1_data_ready → runs PE_L1.
//   Waits for l2_data_ready → runs PE_L2.
//   Runs one sample behind forward FSM.
//   Completely independent of forward FSM except at handshake points.
//
// PIPELINE OVERLAP (per sample boundary):
//
//   Cycle:  0      401k   532k   600k   731k   1132k
//           │       │      │      │      │       │
//   L1_MAC  ████████│      │      │      ████████│
//   L1_RN   │       ███    │      │      │       │
//   L2_MAC  │       │ ████████    │      │       │
//   L2_RN   │       │      │  ███ │      │       │
//   PE_L1   │       ████████████████     │       │
//   PE_L2   │       │      │      │ ████████████ │
//                                  ↑             ↑
//                          N+1 L1_MAC     N+1 PE_L1 starts
//                          starts here
//
// SHADOW BUFFER PROTOCOL:
//   Three shadow buffers protect in-flight data:
//   input_shadow   — captured after LOAD_SAMPLE, read by PE_L2 (needs input acts)
//   l1_act_shadow  — captured after L1_RN, read by PE_L1 and L2_MAC
//   l2_act_shadow  — captured after L2_RN, read by PE_L2
//
// HANDSHAKE SIGNALS:
//   l1_data_ready — forward FSM → update FSM
//                   pulsed when l1_rn_done + goodness_l1_done
//                   carries: l1 goodness scalar, l1 shadow valid
//
//   l2_data_ready — forward FSM → update FSM
//                   pulsed when l2_rn_done + goodness_l2_done
//                   carries: l2 goodness scalar, l2 shadow valid

/*
The two FSMs share zero state. The forward FSM owns everything related to inference — MAC units, relu_norm, goodness calculators, sample loading, label injection. The update FSM owns everything related to learning — the plasticity engine, which layer to update, which pass it is. They communicate through exactly four wires: l1_data_ready, l2_data_ready, l1_data_ack, l2_data_ack. That's the entire interface between the two threads of control.
The forward FSM never waits for the PE. After firing l1_data_ready it immediately starts L2 MAC. After firing l2_data_ready it immediately moves to the next pass or next sample. It doesn't know or care when the PE finishes. This is what makes it truly non-blocking.
The update FSM never touches inference resources. It has no connection to MAC units, relu_norm, or goodness calculators. It only drives pe_start, pe_layer_sel, pe_is_positive, and pe_goodness. The hardware blocks it controls operate on shadow buffers that the forward FSM no longer touches.
The l1_data_ready sticky flag is the critical synchronization point. It gets set by the forward FSM when goodness_l1 completes. It stays high until the update FSM acknowledges it with l1_data_ack. This handles the race where the update FSM might still be in UPD_WAIT_L1 when the flag arrives, or might arrive at UPD_WAIT_L1 after the flag was already set.
The FWD_L2_MAC state starts L2 MAC immediately without waiting for goodness_l1 to finish. This is correct because goodness takes only 258 cycles and L2 MAC takes 131,000 cycles — goodness is guaranteed to complete before L2 MAC finishes. The forward FSM watches for goodness_l1_done while sitting in FWD_L2_MAC and latches the value when it arrives.
There is one duplicate state label FWD_L2_RN in the code which you will need to merge — I split the logic across two case entries by mistake. Combine them into one state that handles both the l2_mac_done transition and the late goodness_l1_done catch.
*/

module training_controller #(
    parameter NUM_SAMPLES  = 7000,
    parameter INPUT_SIZE   = 784,
    parameter LABEL_PIXELS = 10,
    parameter DATA_WIDTH   = 32,
    parameter FRAC_BITS    = 16,
    parameter LABEL_MAG    = 32'sh00010000
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        training_done,

    // ── SAMPLE MEMORY ─────────────────────────────
    output logic [$clog2(NUM_SAMPLES * INPUT_SIZE)-1:0] sample_addr,
    output logic        sample_en,
    input  logic [DATA_WIDTH-1:0] sample_rdata,

    // ── LABEL MEMORY ──────────────────────────────
    output logic [$clog2(NUM_SAMPLES)-1:0] label_addr,
    output logic        label_en,
    input  logic [3:0]  label_rdata,

    // ── INPUT BUFFER WRITE ────────────────────────
    output logic [$clog2(INPUT_SIZE)-1:0] inbuf_waddr,
    output logic        inbuf_we,
    output logic [DATA_WIDTH-1:0] inbuf_wdata,

    // Input shadow capture
    // Pulsed after sample loaded — PE_L2 reads input_shadow
    output logic        input_shadow_capture,

    // ── LAYER 1 CONTROL ───────────────────────────
    output logic        l1_mac_start,
    input  logic        l1_mac_done,
    output logic        l1_rn_start,
    input  logic        l1_rn_done,
    output logic        l1_shadow_capture,

    // ── LAYER 2 CONTROL ───────────────────────────
    output logic        l2_mac_start,
    input  logic        l2_mac_done,
    output logic        l2_rn_start,
    input  logic        l2_rn_done,
    output logic        l2_shadow_capture,

    // ── GOODNESS CONTROL ──────────────────────────
    output logic        goodness_l1_start,
    input  logic        goodness_l1_done,
    input  logic [DATA_WIDTH-1:0] goodness_l1_val,

    output logic        goodness_l2_start,
    input  logic        goodness_l2_done,
    input  logic [DATA_WIDTH-1:0] goodness_l2_val,

    // ── PLASTICITY ENGINE CONTROL ─────────────────
    output logic        pe_start,
    input  logic        pe_done,
    output logic        pe_is_positive,
    output logic        pe_layer_sel,

    // Goodness value routed to PE
    // Muxed by update FSM depending on which layer is being updated
    output logic [DATA_WIDTH-1:0] pe_goodness,

    // ── LABEL OUTPUTS ─────────────────────────────
    output logic [3:0]  correct_label,
    output logic [3:0]  injected_label,
    output logic        inject_en
);

    // ─────────────────────────────────────────────
    // SAMPLE COUNTERS — one per FSM
    // forward_idx: sample currently being forwarded
    // update_idx:  sample currently being weight-updated
    // update_idx runs one behind forward_idx
    // ─────────────────────────────────────────────
    logic [$clog2(NUM_SAMPLES)-1:0] forward_idx;
    logic [$clog2(NUM_SAMPLES)-1:0] update_idx;
    logic [$clog2(INPUT_SIZE)-1:0]  pixel_idx;
    logic [3:0]  current_label;
    logic [3:0]  wrong_label;
    logic        is_positive_pass;   // forward FSM tracks which pass
    logic pe_l1_finished;
    
    always_comb begin
        if (current_label == 4'd9)
            wrong_label = 4'd0;
        else
            wrong_label = current_label + 4'd1;
    end

    // ─────────────────────────────────────────────
    // HANDSHAKE REGISTERS
    // Set by forward FSM, cleared by update FSM
    // Sticky — stay high until update FSM acknowledges
    // ─────────────────────────────────────────────
    logic l1_data_ready;     // L1 activations + goodness available
    logic l2_data_ready;     // L2 activations + goodness available
    logic l1_data_ack;       // update FSM acknowledges l1_data_ready
    logic l2_data_ack;       // update FSM acknowledges l2_data_ready

    // Latched goodness values for update FSM
    // Forward FSM writes these when goodness computation completes
    logic [DATA_WIDTH-1:0] latched_goodness_l1;
    logic [DATA_WIDTH-1:0] latched_goodness_l2;

    // Pass flag latched per sample for update FSM
    // Two entries: one for positive, one for negative
    // Update FSM reads these when it processes that sample
    logic update_is_positive;

    // ─────────────────────────────────────────────
    // FORWARD FSM STATE ENCODING
    // ─────────────────────────────────────────────
    typedef enum logic [3:0] {
        FWD_IDLE          = 4'd0,
        FWD_LOAD_SAMPLE   = 4'd1,
        FWD_INJECT        = 4'd2,
        FWD_L1_MAC        = 4'd3,
        FWD_L1_RN         = 4'd4,
        FWD_L1_GOODNESS   = 4'd5,
        FWD_L2_MAC        = 4'd6,
        FWD_L2_RN         = 4'd7,
        FWD_L2_GOODNESS   = 4'd8,
        FWD_NEXT_PASS     = 4'd9,    // switch pos→neg or neg→next sample
        FWD_DONE          = 4'd10
    } fwd_state_t;

    fwd_state_t fwd_state;

    // ─────────────────────────────────────────────
    // UPDATE FSM STATE ENCODING
    // ─────────────────────────────────────────────
    typedef enum logic [2:0] {
        UPD_IDLE      = 3'd0,
        UPD_WAIT_L1   = 3'd1,    // wait for l1_data_ready
        UPD_PE_L1     = 3'd2,    // run PE on layer 1
        UPD_WAIT_L2   = 3'd3,    // wait for l2_data_ready
        UPD_PE_L2     = 3'd4,    // run PE on layer 2
        UPD_NEXT      = 3'd5,    // advance update_idx
        UPD_DONE      = 3'd6
    } upd_state_t;

    upd_state_t upd_state;

    // ─────────────────────────────────────────────
    // FORWARD FSM
    // Runs continuously through samples and passes
    // Completely non-blocking — never waits for PE
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fwd_state          <= FWD_IDLE;
            forward_idx        <= '0;
            pixel_idx          <= '0;
            current_label      <= '0;
            is_positive_pass   <= 1'b1;
            l1_data_ready      <= 1'b0;
            l2_data_ready      <= 1'b0;
            latched_goodness_l1<= '0;
            latched_goodness_l2<= '0;
            l1_mac_start       <= 1'b0;
            l1_rn_start        <= 1'b0;
            l2_mac_start       <= 1'b0;
            l2_rn_start        <= 1'b0;
            l1_shadow_capture  <= 1'b0;
            l2_shadow_capture  <= 1'b0;
            input_shadow_capture <= 1'b0;
            goodness_l1_start  <= 1'b0;
            goodness_l2_start  <= 1'b0;
            sample_en          <= 1'b0;
            label_en           <= 1'b0;
            inbuf_we           <= 1'b0;
            inject_en          <= 1'b0;
            correct_label      <= '0;
            injected_label     <= '0;
        end
        else begin
            // Default deassert
            l1_mac_start       <= 1'b0;
            l1_rn_start        <= 1'b0;
            l2_mac_start       <= 1'b0;
            l2_rn_start        <= 1'b0;
            l1_shadow_capture  <= 1'b0;
            l2_shadow_capture  <= 1'b0;
            input_shadow_capture <= 1'b0;
            goodness_l1_start  <= 1'b0;
            goodness_l2_start  <= 1'b0;
            sample_en          <= 1'b0;
            label_en           <= 1'b0;
            inbuf_we           <= 1'b0;
            inject_en          <= 1'b0;

            // Clear handshakes when update FSM acknowledges
            if (l1_data_ack) l1_data_ready <= 1'b0;
            if (l2_data_ack) l2_data_ready <= 1'b0;

            case (fwd_state)

                FWD_IDLE: begin
                    if (start) begin
                        forward_idx      <= '0;
                        pixel_idx        <= '0;
                        is_positive_pass <= 1'b1;
                        fwd_state        <= FWD_LOAD_SAMPLE;
                    end
                end

                // Stream sample into input buffer
                // Capture input shadow at end for PE_L2 use
                FWD_LOAD_SAMPLE: begin
                    if (pixel_idx == '0) begin
                        label_addr <= forward_idx;
                        label_en   <= 1'b1;
                    end

                    sample_addr <= (forward_idx * INPUT_SIZE) + pixel_idx;
                    sample_en   <= 1'b1;

                    if (pixel_idx > '0) begin
                        inbuf_waddr <= pixel_idx - 1;
                        inbuf_we    <= 1'b1;
                        inbuf_wdata <= sample_rdata;
                    end

                    if (pixel_idx == INPUT_SIZE - 1) begin
                        // Capture input to shadow so PE_L2 can
                        // read it while next sample loads
                        input_shadow_capture <= 1'b1;
                        pixel_idx            <= '0;
                        fwd_state            <= FWD_INJECT;
                    end
                    else begin
                        pixel_idx <= pixel_idx + 1;
                    end
                end

                // Inject label — positive or negative depending on pass
                FWD_INJECT: begin
                    if (pixel_idx == '0)
                        current_label <= label_rdata;

                    inbuf_waddr <= {{($clog2(INPUT_SIZE)-4){1'b0}},
                                    pixel_idx[3:0]};
                    inbuf_we    <= 1'b1;
                    inbuf_wdata <= '0;

                    if (is_positive_pass) begin
                        correct_label  <= current_label;
                        injected_label <= current_label;
                        if (pixel_idx[3:0] == label_rdata)
                            inbuf_wdata <= LABEL_MAG;
                    end
                    else begin
                        injected_label <= wrong_label;
                        if (pixel_idx[3:0] == wrong_label)
                            inbuf_wdata <= LABEL_MAG;
                    end

                    if (pixel_idx == LABEL_PIXELS - 1) begin
                        pixel_idx <= '0;
                        inject_en <= 1'b1;
                        fwd_state <= FWD_L1_MAC;
                    end
                    else begin
                        pixel_idx <= pixel_idx + 1;
                    end
                end

                FWD_L1_MAC: begin
                    l1_mac_start <= 1'b1;
                    fwd_state    <= FWD_L1_RN;
                end

                FWD_L1_RN: begin
                    if (l1_mac_done) begin
                        l1_rn_start <= 1'b1;
                        fwd_state   <= FWD_L1_GOODNESS;
                    end
                end

                // L1 RN done — capture shadow, start goodness
                // Signal update FSM that L1 data is ready
                FWD_L1_GOODNESS: begin
                    if (l1_rn_done) begin
                        l1_shadow_capture  <= 1'b1;
                        goodness_l1_start  <= 1'b1;
                        fwd_state          <= FWD_L2_MAC;
                    end
                end

                // L2 MAC starts immediately after L1 goodness triggered
                // Does not wait for goodness to finish —
                // goodness_l1 takes ~258 cycles, L2 MAC takes 131k cycles
                // Goodness will be done long before L2 MAC finishes
                // Update FSM watches goodness_l1_done independently
                FWD_L2_MAC: begin
                    l2_mac_start <= 1'b1;

                    // Latch goodness when it arrives
                    // goodness_l1_done comes ~258 cycles after l1_rn_done
                    // well before l2_mac_done
                    if (goodness_l1_done) begin
                        latched_goodness_l1 <= goodness_l1_val;
                        l1_data_ready       <= 1'b1;
                        // update_is_positive carried by l1_data_ready context
                    end

                    if (l2_mac_done)
                        fwd_state <= FWD_L2_RN;
                end

                FWD_L2_RN: begin
                    // Track PE_L1 finishing if it arrives here
                    if (pe_done)
                        pe_l1_finished <= 1'b1;
                
                    // Start L2 relu_norm as soon as L2 MAC done
                    if (l2_mac_done) begin
                        l2_rn_start <= 1'b1;
                    end
                
                    // Catch late goodness_l1_done if it hasn't fired yet
                    if (goodness_l1_done && !l1_data_ready) begin
                        latched_goodness_l1 <= goodness_l1_val;
                        l1_data_ready       <= 1'b1;
                    end
                
                    // Move to goodness once relu_norm completes
                    if (l2_rn_done) begin
                        l2_shadow_capture <= 1'b1;
                        goodness_l2_start <= 1'b1;
                        fwd_state         <= FWD_L2_GOODNESS;
                    end
                end

                FWD_L2_GOODNESS: begin
                    if (goodness_l2_done) begin
                        latched_goodness_l2 <= goodness_l2_val;
                        l2_data_ready       <= 1'b1;
                        fwd_state           <= FWD_NEXT_PASS;
                    end
                end

                // Switch pass or advance to next sample
                // Forward FSM does NOT wait for PE here
                // It immediately loads next pass or next sample
                FWD_NEXT_PASS: begin
                    if (is_positive_pass) begin
                        // Done with positive pass
                        // Start negative pass on same sample
                        is_positive_pass <= 1'b0;
                        pixel_idx        <= '0;
                        fwd_state        <= FWD_LOAD_SAMPLE;
                    end
                    else begin
                        // Done with negative pass
                        // Advance to next sample
                        is_positive_pass <= 1'b1;
                        pixel_idx        <= '0;
                        if (forward_idx == NUM_SAMPLES - 1)
                            fwd_state <= FWD_DONE;
                        else begin
                            forward_idx <= forward_idx + 1;
                            fwd_state   <= FWD_LOAD_SAMPLE;
                        end
                    end
                end

                FWD_DONE: begin
                    // Forward FSM done — update FSM may still be running
                    // training_done asserted by update FSM when it catches up
                end

            endcase
        end
    end

    // ─────────────────────────────────────────────
    // UPDATE FSM
    // Runs one sample behind forward FSM
    // Watches handshake signals, never touches
    // inference resources
    // ─────────────────────────────────────────────
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            upd_state        <= UPD_IDLE;
            update_idx       <= '0;
            update_is_positive <= 1'b1;
            pe_start         <= 1'b0;
            pe_is_positive   <= 1'b0;
            pe_layer_sel     <= 1'b0;
            pe_goodness      <= '0;
            l1_data_ack      <= 1'b0;
            l2_data_ack      <= 1'b0;
            training_done    <= 1'b0;
        end
        else begin
            // Default deassert
            pe_start      <= 1'b0;
            l1_data_ack   <= 1'b0;
            l2_data_ack   <= 1'b0;
            training_done <= 1'b0;

            case (upd_state)

                UPD_IDLE: begin
                    if (start) begin
                        update_idx         <= '0;
                        update_is_positive <= 1'b1;
                        upd_state          <= UPD_WAIT_L1;
                    end
                end

                // Wait for forward FSM to signal L1 data ready
                // When it arrives, acknowledge and start PE_L1
                UPD_WAIT_L1: begin
                    if (l1_data_ready) begin
                        l1_data_ack    <= 1'b1;
                        pe_goodness    <= latched_goodness_l1;
                        pe_is_positive <= update_is_positive;
                        pe_layer_sel   <= 1'b0;   // layer 1
                        pe_start       <= 1'b1;
                        upd_state      <= UPD_PE_L1;
                    end
                end

                // PE_L1 running — simultaneously forward FSM is
                // running L2 MAC for the same sample
                // Also watch for L2 data arriving while PE_L1 runs
                UPD_PE_L1: begin
                    if (pe_done)
                        upd_state <= UPD_WAIT_L2;
                end

                // Wait for L2 data ready
                // In practice this is already set by the time
                // PE_L1 finishes because PE_L1 takes 401k cycles
                // and L2 RN + goodness completes in ~600 cycles
                UPD_WAIT_L2: begin
                    if (l2_data_ready) begin
                        l2_data_ack    <= 1'b1;
                        pe_goodness    <= latched_goodness_l2;
                        pe_is_positive <= update_is_positive;
                        pe_layer_sel   <= 1'b1;   // layer 2
                        pe_start       <= 1'b1;
                        upd_state      <= UPD_PE_L2;
                    end
                end

                // PE_L2 running — simultaneously forward FSM is
                // loading sample N+1 and running its L1 MAC
                // This is the cross-sample parallelism
                UPD_PE_L2: begin
                    if (pe_done)
                        upd_state <= UPD_NEXT;
                end

                // Advance pass or sample counter
                UPD_NEXT: begin
                    if (update_is_positive) begin
                        // Positive pass done, wait for negative pass data
                        update_is_positive <= 1'b0;
                        upd_state          <= UPD_WAIT_L1;
                    end
                    else begin
                        // Both passes done for this sample
                        update_is_positive <= 1'b1;
                        if (update_idx == NUM_SAMPLES - 1)
                            upd_state <= UPD_DONE;
                        else begin
                            update_idx <= update_idx + 1;
                            upd_state  <= UPD_WAIT_L1;
                        end
                    end
                end

                UPD_DONE: begin
                    // Update FSM processed all samples
                    // Forward FSM is also done by this point
                    training_done <= 1'b1;
                    upd_state     <= UPD_IDLE;
                end

            endcase
        end
    end

endmodule
