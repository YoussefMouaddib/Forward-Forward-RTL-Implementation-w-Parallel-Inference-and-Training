"""
Forward-Forward Algorithm - Golden Reference Implementation
Architecture: 784 -> 256 -> 256 -> 10
Fixed-Point: Q16.16 (32-bit signed, 16 fractional bits)

This is the software reference that all RTL modules will be verified against.
Every operation here has a direct RTL equivalent. Keep them in sync.

Author: Youssef Mouaddib 2026
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ─────────────────────────────────────────────
# Q16.16 FIXED-POINT ARITHMETIC
# ─────────────────────────────────────────────
# Representation: 32-bit signed integer
# Real value = stored_integer / 2^16
# Range: approximately -32768.0 to +32767.9999847

FRAC_BITS = 16
SCALE     = 1 << FRAC_BITS          # 65536
Q_MAX     = (1 << 31) - 1           # 2147483647  (max positive Q16.16)
Q_MIN     = -(1 << 31)              # -2147483648 (most negative Q16.16)



def to_q(x):
    """Convert float scalar or numpy array to Q16.16 integer representation."""
    return np.clip(np.round(np.array(x, dtype=np.float64) * SCALE),
                   Q_MIN, Q_MAX).astype(np.int32)

def from_q(x):
    """Convert Q16.16 integer representation back to float for inspection."""
    return np.array(x, dtype=np.int64) / SCALE

def q_mul(a, b):
    """
    Multiply two Q16.16 numbers.
    Intermediate product is Q32.32 (64-bit), right-shift by 16 to return Q16.16.
    This is exactly what the RTL DSP48 + right-shift does.
    """
    return np.clip(
        (np.int64(a) * np.int64(b)) >> FRAC_BITS,
        Q_MIN, Q_MAX
    ).astype(np.int32)

def q_add(a, b):
    """Add two Q16.16 numbers with saturation."""
    return np.clip(np.int64(a) + np.int64(b), Q_MIN, Q_MAX).astype(np.int32)

def q_relu(x):
    """ReLU in Q16.16: max(0, x). Negative Q16.16 integers are just negative ints."""
    return np.maximum(x, np.int32(0))

# ─────────────────────────────────────────────
# LAYER NORMALIZATION (Q16.16)
# ─────────────────────────────────────────────
# Hinton uses layer norm before computing goodness.
# Decouples magnitude from direction so goodness measures
# structure, not just "how loud" the activations are.
# RTL equivalent: compute mean, subtract, compute variance, divide.

def q_layer_norm(x, eps_float=1e-5):
    """Mean-only normalization. Removes layer norm variance scaling
    so goodness retains meaningful magnitude differences."""
    x_f = from_q(x).astype(np.float64)
    mean = np.mean(x_f)
    x_centered = x_f - mean
    return to_q(x_centered)

# ─────────────────────────────────────────────
# GOODNESS COMPUTATION
# ─────────────────────────────────────────────
# goodness = sum of squared activations (after layer norm)
# In RTL: square each activation (q_mul with itself), accumulate.
# Returns a single Q16.16 scalar.

def compute_goodness(activations_q):
    squared = q_mul(activations_q, activations_q)
    #print(f"DEBUG squared: min={from_q(squared).min():.2f} max={from_q(squared).max():.2f} sum_float={from_q(squared).sum():.2f}")
    goodness = np.int64(0)
    for s in squared:
        goodness = np.clip(goodness + np.int64(s), Q_MIN, Q_MAX)
    return np.int32(goodness)

# ─────────────────────────────────────────────
# SIGMOID APPROXIMATION (Q16.16)
# ─────────────────────────────────────────────
# Used in the weight update rule.
# sigmoid(x) = 1 / (1 + exp(-x))
# In RTL we approximate with a piecewise linear function.
# Here we compute in float and convert - RTL section will use LUT-based approx.
# Flagged explicitly so the RTL designer (you) knows to match this.

THETA = to_q(3.0)
LR = to_q(0.005)

def q_sigmoid(x_q):
    """
    Compute sigmoid of a Q16.16 value.
    Returns Q16.16 result in range [0, 1] -> stored as [0, 65536].
    RTL NOTE: implement as LUT or piecewise linear approximation.
    """
    x_f = from_q(x_q)
    sig = 1.0 / (1.0 + np.exp(-np.clip(x_f, -10, 10)))
    return to_q(sig)

# ─────────────────────────────────────────────
# FORWARD PASS THROUGH ONE LAYER
# ─────────────────────────────────────────────

def forward_layer(x_q, weights_q, bias_q, apply_relu=True):
    # Matrix multiply in int64 to prevent overflow, then right-shift back to Q16.16
    acc = (weights_q.astype(np.int64) @ x_q.astype(np.int64)) >> FRAC_BITS
    acc = np.clip(acc + bias_q.astype(np.int64), Q_MIN, Q_MAX).astype(np.int32)
    if apply_relu:
        acc = q_relu(acc)
    return acc

# ─────────────────────────────────────────────
# WEIGHT UPDATE (PLASTICITY ENGINE)
# ─────────────────────────────────────────────
# This is the Forward-Forward update rule.
# Δw = η * (pos_flag - sigmoid(goodness - θ)) * x * y
# where x is input activation, y is output activation
# pos_flag = +1 for positive pass, -1 for negative pass
#
# RTL EQUIVALENT: plasticity_engine.sv
# Runs on port B of weight BRAM while inference runs on port A.

LR = to_q(0.01)     # learning rate η in Q16.16

def update_weights(weights_q, bias_q, x_q, y_q, goodness_q, is_positive):
    goodness_minus_theta = np.clip(
        np.int64(goodness_q) - np.int64(THETA), Q_MIN, Q_MAX).astype(np.int32)
    sig = q_sigmoid(goodness_minus_theta)
    pos_flag_q = to_q(1.0) if is_positive else to_q(-1.0)
    delta_factor_q = np.clip(
        np.int64(pos_flag_q) - np.int64(sig), Q_MIN, Q_MAX).astype(np.int32)
    scaled_delta_q = q_mul(LR, delta_factor_q)

    # Outer product: (NUM_NEURONS,) x (INPUT_SIZE,) -> (NUM_NEURONS, INPUT_SIZE)
    y_factor = (np.int64(scaled_delta_q) * y_q.astype(np.int64)) >> FRAC_BITS
    dw = (y_factor[:, None].astype(np.int64) * x_q.astype(np.int64)[None, :]) >> FRAC_BITS
    new_weights = np.clip(
        weights_q.astype(np.int64) + dw, Q_MIN, Q_MAX).astype(np.int32)

    db = np.clip(
        np.int64(scaled_delta_q) * y_q.astype(np.int64) >> FRAC_BITS,
        Q_MIN, Q_MAX).astype(np.int32)
    new_bias = np.clip(
        bias_q.astype(np.int64) + db, Q_MIN, Q_MAX).astype(np.int32)

    return new_weights, new_bias

# ─────────────────────────────────────────────
# LABEL INJECTION
# ─────────────────────────────────────────────
# Overwrites first 10 pixels of the 784-pixel image with one-hot label encoding.
# Positive pass: correct label injected.
# Negative pass: wrong label injected (correct + 1) mod 10.
#
# RTL EQUIVALENT: label_injector.sv - small mux before layer 1 input buffer.

LABEL_MAGNITUDE = to_q(1.0)   # value written into the hot pixel

def inject_label(image_q, label, is_positive):
    """
    image_q: Q16.16 array of shape (784,)
    label:   integer 0-9, the correct label
    is_positive: True injects correct label, False injects wrong label
    
    Returns modified Q16.16 image array.
    First 10 pixels become a one-hot encoding of the injected label.
    """
    img = image_q.copy()
    img[0:10] = np.int32(0)   # clear label pixels
    
    if is_positive:
        injected_label = label
    else:
        injected_label = (label + 1) % 10   # simplest wrong label
    
    img[injected_label] = LABEL_MAGNITUDE
    return img

# ─────────────────────────────────────────────
# MNIST LOADER
# ─────────────────────────────────────────────
def load_mnist():
    """Load MNIST via keras. Returns float arrays normalized to [0,1]."""
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    test_images  = test_images.reshape(-1, 784).astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.int32)
    test_labels  = test_labels.astype(np.int32)
    return train_images, train_labels, test_images, test_labels
# ─────────────────────────────────────────────
# NETWORK INITIALIZATION
# ─────────────────────────────────────────────

def init_network(layer_sizes, seed=42):
    """
    Initialize weights and biases as Q16.16.
    Uses He initialization scaled for ReLU networks.
    layer_sizes: list e.g. [784, 256, 256, 10]
    Returns: list of (weights_q, bias_q) tuples per layer.
    """
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        # He init: std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / fan_in)
        w_f = rng.normal(0, std, (fan_out, fan_in)).astype(np.float64)
        b_f = np.zeros(fan_out, dtype=np.float64)
        layers.append((to_q(w_f), to_q(b_f)))
    return layers

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
# At inference: try all 10 labels, pick highest goodness at final layer.

def infer(image_f, layers):
    """
    Classify one image by running 10 forward passes with each label injected.
    image_f: float array shape (784,), pixel values in [0,1]
    layers:  list of (weights_q, bias_q) from current network state
    Returns: predicted label (int)
    """
    image_q = to_q(image_f)
    best_label    = -1
    best_goodness = np.int32(Q_MIN)
    
    for candidate_label in range(10):
        img_injected = inject_label(image_q, candidate_label, is_positive=True)
        
        x = img_injected
        for idx, (w_q, b_q) in enumerate(layers):
            apply_relu = (idx < len(layers) - 1)
            x = forward_layer(x, w_q, b_q, apply_relu=apply_relu)
            if idx < len(layers) - 1:
                x = q_layer_norm(x)
        
        # Goodness from last hidden layer (layer index 1, the second 256-layer)
        # We evaluate goodness of the penultimate layer per Hinton's paper
        x_hidden = img_injected
        hidden_acts = []
        for idx, (w_q, b_q) in enumerate(layers[:-1]):
            x_hidden = forward_layer(x_hidden, w_q, b_q, apply_relu=True)
            x_hidden = q_layer_norm(x_hidden)
            hidden_acts.append(x_hidden)
        
        goodness = compute_goodness(hidden_acts[-1])
        
        if goodness > best_goodness:
            best_goodness = goodness
            best_label    = candidate_label
    
    return best_label

def evaluate_accuracy(test_images, test_labels, layers, num_samples=1000):
    """Run inference on num_samples test images, return accuracy float."""
    correct = 0
    for i in range(num_samples):
        pred = infer(test_images[i], layers)
        if pred == test_labels[i]:
            correct += 1
    return correct / num_samples

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(train_images, train_labels, test_images, test_labels,
          layer_sizes=(784, 256, 256, 10),
          num_epochs=1,
          eval_every=1000,
          eval_samples=500):
    """
    Full Forward-Forward training loop.
    
    This matches the RTL training_controller.sv state machine exactly:
    INJECT_LABEL -> POS_FORWARD -> POS_UPDATE ->
    INJECT_NEG_LABEL -> NEG_FORWARD -> NEG_UPDATE -> NEXT_SAMPLE
    
    Logs accuracy at checkpoints for the learning curve plot.
    Also saves weight snapshots that can be loaded into RTL testbench.
    """
    print(f"Initializing network: {layer_sizes}")
    layers = init_network(list(layer_sizes))
    
    accuracy_log   = []   # (sample_index, accuracy)
    goodness_log   = []   # (sample_index, pos_goodness, neg_goodness) per layer
    
    num_train = len(train_images)
    total_samples = num_epochs * num_train
    sample_count  = 0
    
    print(f"Training for {num_epochs} epoch(s), {num_train} samples each.")
    print(f"Evaluating every {eval_every} samples on {eval_samples} test images.")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Shuffle training data each epoch
        perm = np.random.permutation(num_train)
        
        for idx in perm:
            image_f = train_images[idx]
            label   = int(train_labels[idx])
            image_q = to_q(image_f)
            
            # ── POSITIVE PASS ──────────────────────────────────────────
            # Inject correct label into first 10 pixels
            pos_input = inject_label(image_q, label, is_positive=True)
            
            pos_activations = [pos_input]
            x = pos_input
            for layer_idx, (w_q, b_q) in enumerate(layers[:-1]):
                x = forward_layer(x, w_q, b_q, apply_relu=True)
                x = q_layer_norm(x)
                pos_activations.append(x)
            
            if sample_count == 0:
                for li, acts in enumerate(pos_activations[1:]):
                    acts_f = from_q(acts)
                    print(f"Layer {li+1} activations: "
                        f"min={acts_f.min():.4f} "
                        f"max={acts_f.max():.4f} "
                        f"mean={acts_f.mean():.4f} "
                        f"nonzero={np.count_nonzero(acts)}/{len(acts)}")

            # Compute goodness per hidden layer
            pos_goodness = [compute_goodness(pos_activations[i+1])
                           for i in range(len(layers)-1)]
            
            # ── NEGATIVE PASS ──────────────────────────────────────────
            # Inject wrong label
            neg_input = inject_label(image_q, label, is_positive=False)
            
            neg_activations = [neg_input]
            x = neg_input
            for layer_idx, (w_q, b_q) in enumerate(layers[:-1]):
                x = forward_layer(x, w_q, b_q, apply_relu=True)
                x = q_layer_norm(x)
                neg_activations.append(x)
            
            neg_goodness = [compute_goodness(neg_activations[i+1])
                           for i in range(len(layers)-1)]
            
            # ── WEIGHT UPDATES ─────────────────────────────────────────
            # Update each hidden layer independently (local learning rule)
            # RTL: plasticity_engine runs on port B during inference idle window
            for layer_idx in range(len(layers) - 1):
                w_q, b_q = layers[layer_idx]
                
                # Positive update
                w_q, b_q = update_weights(
                    w_q, b_q,
                    x_q        = pos_activations[layer_idx],
                    y_q        = pos_activations[layer_idx + 1],
                    goodness_q = pos_goodness[layer_idx],
                    is_positive= True
                )
                
                # Negative update
                w_q, b_q = update_weights(
                    w_q, b_q,
                    x_q        = neg_activations[layer_idx],
                    y_q        = neg_activations[layer_idx + 1],
                    goodness_q = neg_goodness[layer_idx],
                    is_positive= False
                )
                
                layers[layer_idx] = (w_q, b_q)
            
            sample_count += 1
            
            # ── LOGGING ────────────────────────────────────────────────
            if sample_count % eval_every == 0:
                acc = evaluate_accuracy(test_images, test_labels,
                                        layers, num_samples=eval_samples)
                accuracy_log.append((sample_count, acc))
                avg_pos_g = np.mean([from_q(g) for g in pos_goodness])
                avg_neg_g = np.mean([from_q(g) for g in neg_goodness])
                goodness_log.append((sample_count, avg_pos_g, avg_neg_g))
                print(f"  Sample {sample_count:6d} | "
                      f"Acc: {acc*100:5.1f}% | "
                      f"Pos goodness: {avg_pos_g:7.2f} | "
                      f"Neg goodness: {avg_neg_g:7.2f}")
    
    return layers, accuracy_log, goodness_log

# ─────────────────────────────────────────────
# WEIGHT EXPORT FOR RTL TESTBENCH
# ─────────────────────────────────────────────
# Exports trained or initial weights as .mem files
# in the same hex format the Vivado testbench expects.
# Load these into your RTL testbench with $readmemh.

def export_weights_to_mem(layers, prefix="weights"):
    """
    Export all layer weights to .mem files for RTL testbench use.
    Format: one 32-bit hex value per line (8 hex digits), no prefix.
    """
    os.makedirs("mem_files", exist_ok=True)
    for layer_idx, (w_q, b_q) in enumerate(layers):
        # Weights: one file per layer, flattened row-major
        # RTL reads this as: address = neuron_idx * INPUT_SIZE + weight_idx
        fname = f"mem_files/{prefix}_layer{layer_idx+1}_w.mem"
        with open(fname, 'w') as f:
            for val in w_q.flatten():
                # Write as 8-digit hex, two's complement for negatives
                f.write(f"{np.uint32(val):08x}\n")
        
        # Biases: one file per layer
        fname = f"mem_files/{prefix}_layer{layer_idx+1}_b.mem"
        with open(fname, 'w') as f:
            for val in b_q:
                f.write(f"{np.uint32(val):08x}\n")
        
        print(f"  Exported layer {layer_idx+1}: "
              f"{w_q.shape} weights -> mem_files/{prefix}_layer{layer_idx+1}_w.mem")

# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_results(accuracy_log, goodness_log):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    samples = [x[0] for x in accuracy_log]
    accs    = [x[1] * 100 for x in accuracy_log]
    ax1.plot(samples, accs, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel("Training Samples Seen")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Learning Curve — Forward-Forward Q16.16")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    g_samples = [x[0] for x in goodness_log]
    pos_g = [x[1] for x in goodness_log]
    neg_g = [x[2] for x in goodness_log]
    ax2.plot(g_samples, pos_g, 'g-o', label='Positive pass', linewidth=2, markersize=4)
    ax2.plot(g_samples, neg_g, 'r-o', label='Negative pass', linewidth=2, markersize=4)
    ax2.axhline(y=from_q(THETA), color='k', linestyle='--', label=f'θ = {from_q(THETA):.1f}')
    ax2.set_xlabel("Training Samples Seen")
    ax2.set_ylabel("Average Goodness")
    ax2.set_title("Goodness Separation — Positive vs Negative")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ff_training_results.png", dpi=150)
    print("\nPlot saved to ff_training_results.png")
    plt.show()

def export_samples_for_rtl(train_images, train_labels, 
                            num_samples=100, prefix="mem_files"):
    """
    Export flat sample memory and label memory for RTL testbench.
    samples_flat.mem: all pixels for all samples, row-major Q16.16
    labels.mem: one 4-bit label per sample as single hex digit
    """
    os.makedirs(prefix, exist_ok=True)
    
    # Samples flat
    fname = f"{prefix}/samples_flat.mem"
    with open(fname, 'w') as f:
        for i in range(num_samples):
            pixels_q = to_q(train_images[i])
            for px in pixels_q:
                f.write(f"{np.uint32(px):08x}\n")
    print(f"Exported {num_samples} samples -> {fname}")
    
    # Labels
    fname = f"{prefix}/labels.mem"
    with open(fname, 'w') as f:
        for i in range(num_samples):
            f.write(f"{int(train_labels[i]):01x}\n")
    print(f"Exported {num_samples} labels -> {fname}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Forward-Forward Golden Reference — Q16.16 Fixed Point")
    print("Architecture: 784 -> 256 -> 256 -> 10")
    print("=" * 60)
    
    # Load MNIST
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(f"  Train: {len(train_images)} images")
    print(f"  Test:  {len(test_images)} images")
    
    # Export initial random weights for RTL baseline verification
    print("\nExporting initial weights for RTL testbench...")
    init_layers = init_network([784, 256, 256, 10])
    export_weights_to_mem(init_layers, prefix="initial")
    export_samples_for_rtl(train_images, train_labels, num_samples=100)
    
    # Train
    print("\nStarting training...")
    trained_layers, acc_log, g_log = train(
    train_images[:7000], train_labels[:7000],
    test_images,  test_labels,
    layer_sizes  = (784, 256, 256, 10),
    num_epochs   = 1,
    eval_every   = 100,
    eval_samples = 100
)
    
    # Export trained weights
    print("\nExporting trained weights...")
    export_weights_to_mem(trained_layers, prefix="trained")
    
    # Plot
    if acc_log:
        plot_results(acc_log, g_log)
    
    # Final accuracy on full test set
    print("\nRunning final evaluation on 2000 test samples...")
    final_acc = evaluate_accuracy(test_images, test_labels,
                                   trained_layers, num_samples=1000)
    print(f"\nFinal Test Accuracy: {final_acc*100:.1f}%")
    print("\nDone. Use mem_files/ outputs to initialize RTL testbench.")
