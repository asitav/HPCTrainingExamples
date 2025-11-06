# TinyOpenFold V1: PyTorch Baseline

Educational implementation of AlphaFold 2's Evoformer architecture with comprehensive profiling integration.

## Overview

This version provides a clean, well-documented baseline implementation of the core AlphaFold 2 architecture, focusing on the **Evoformer** blocks that process MSA (Multiple Sequence Alignment) and pair representations.

## Quick Start

### Basic Training Run

```bash
# Default configuration: 64 residues, 16 MSA sequences, 4 Evoformer blocks
python tiny_openfold_v1.py --batch-size 4 --num-steps 30

# Expected output:
# Model Configuration:
#    MSA dimension: 64
#    Pair dimension: 128
#    Evoformer blocks: 4
#    Total parameters: 2,641,728
#    Model size: 10.6 MB (FP32)
#
# Training steps complete with loss decreasing
```

### Validation Check

```bash
# Verify your environment is set up correctly
python tiny_openfold_v1.py --validate-setup

# Should output:
# Validation successful! Environment ready.
```

## Architecture Components

### 1. MSA Representation Processing

**MSA Row-wise Attention with Pair Bias**
- Attends across residues within each MSA sequence
- Biased by the pair representation (key innovation!)
- Shape: `(batch, n_seqs, seq_len, msa_dim)`

**MSA Column-wise Attention**
- Attends across different sequences for each position
- Enables communication between sequences in the MSA
- Shape: `(batch, n_seqs, seq_len, msa_dim)`

**MSA Transition**
- Point-wise feed-forward network
- Applied to each MSA element independently

### 2. Pair Representation Processing

**Outer Product Mean**
- Projects MSA patterns onto pairwise space
- Computes mean outer product across MSA sequences
- Updates pair representation with sequence information

**Triangle Multiplicative Updates**
- Geometric reasoning: if i-j and j-k are close, i-k should be considered
- Two versions: outgoing and incoming edges
- Most computationally expensive operation (O(N³))

**Triangle Self-Attention**
- Attention over edges in the residue graph
- Two versions: starting and ending nodes
- Enables long-range communication

**Pair Transition**
- Point-wise feed-forward network for pair representation

### 3. Structure Module

**Simplified Distance Prediction**
- Predicts pairwise distances from pair representation
- In full AlphaFold 2, this is the Invariant Point Attention (IPA) module
- Output: `(batch, seq_len, seq_len, 1)` - distance matrix

## Model Configuration

### Default Configuration

```python
TinyOpenFoldConfig(
    vocab_size=21,              # 20 amino acids + unknown
    msa_dim=64,                 # MSA feature dimension
    pair_dim=128,               # Pair feature dimension
    n_evoformer_blocks=4,       # Number of Evoformer blocks
    n_heads_msa=4,              # MSA attention heads
    n_heads_pair=4,             # Pair attention heads
    msa_intermediate_dim=256,   # MSA FFN dimension (4x msa_dim)
    pair_intermediate_dim=512,  # Pair FFN dimension (4x pair_dim)
    outer_product_dim=32,       # Outer product projection dim
    max_seq_len=64,             # Maximum sequence length
    n_seqs=16,                  # Number of MSA sequences
)
```

### Scaling Configurations

#### Tiny (for testing)
```bash
python tiny_openfold_v1.py \
    --msa-dim 32 \
    --pair-dim 64 \
    --num-blocks 2 \
    --seq-len 32 \
    --num-seqs 8 \
    --batch-size 8

# Parameters: ~660K
# Memory: ~40 MB
# Speed: ~15-20 samples/sec
```

#### Small (default)
```bash
python tiny_openfold_v1.py \
    --msa-dim 64 \
    --pair-dim 128 \
    --num-blocks 4 \
    --seq-len 64 \
    --num-seqs 16 \
    --batch-size 4

# Parameters: ~2.6M
# Memory: ~100 MB
# Speed: ~8-10 samples/sec
```

#### Medium
```bash
python tiny_openfold_v1.py \
    --msa-dim 128 \
    --pair-dim 256 \
    --num-blocks 8 \
    --seq-len 128 \
    --num-seqs 32 \
    --batch-size 2

# Parameters: ~42M
# Memory: ~800 MB
# Speed: ~1-2 samples/sec
```

## Profiling Guide

### PyTorch Profiler

Enable comprehensive profiling with PyTorch's built-in profiler:

```bash
# Basic profiling
python tiny_openfold_v1.py \
    --enable-pytorch-profiler \
    --profile-dir ./profiles \
    --batch-size 4 \
    --num-steps 30

# View in TensorBoard
tensorboard --logdir ./profiles
```

**What to Look For in TensorBoard:**
- **Kernel View**: Which operations take the most time?
- **Memory View**: Where are memory allocations happening?
- **Timeline**: Are there idle periods or synchronization issues?

### Memory Profiling

Track memory usage throughout training:

```bash
python tiny_openfold_v1.py \
    --enable-memory-profiling \
    --profile-dir ./memory_analysis \
    --batch-size 4

# Check performance_summary.json for memory statistics
cat ./memory_analysis/performance_summary.json
```

### Complete Profiling Suite

Enable all profiling features:

```bash
python tiny_openfold_v1.py \
    --enable-all-profiling \
    --profile-dir ./complete_analysis \
    --batch-size 4 \
    --num-steps 50
```

## Performance Analysis

### Expected Bottlenecks

Based on the architecture, expect these components to dominate compute time:

1. **Triangle Operations** (40-50% of time)
   - O(N³) complexity makes these expensive
   - Both multiplicative updates and attention
   - Most sensitive to sequence length

2. **MSA Attention** (25-35% of time)
   - Row-wise attention: O(N_seqs × N_res²)
   - Column-wise attention: O(N_res × N_seqs²)
   - Depends on both MSA depth and sequence length

3. **Outer Product Mean** (10-15% of time)
   - Computing outer products across MSA
   - Memory-bound operation

4. **Transitions** (5-10% of time)
   - Feed-forward networks
   - Usually well-optimized by PyTorch

### Memory Consumption

Memory usage breakdown (approximate):

```
Total GPU Memory = Model Parameters + Activations + Gradients + Optimizer States

For batch=4, seq_len=64, n_seqs=16:
- Model: ~11 MB (FP32)
- MSA activations: ~4 MB
- Pair activations: ~32 MB
- Attention scores: ~8 MB
- Gradients: ~11 MB
- Optimizer (Adam): ~22 MB
- Total: ~90-100 MB
```

**Key Insight**: Pair representation dominates memory (seq_len²)

### Optimization Opportunities

From the baseline implementation, potential optimizations include:

1. **Flash Attention** for MSA attention operations
2. **Kernel Fusion** for triangle operations
3. **Mixed Precision (FP16)** to reduce memory and improve throughput
4. **Gradient Checkpointing** for larger models
5. **Custom CUDA/Triton Kernels** for triangle updates

## Training Output Explanation

### During Training

```
Step   0/50 | Loss: 45.2341 | Speed:   8.5 samples/sec | Memory:  102.3 MB | Time:  470.2ms
```

- **Loss**: MSE on predicted distances (should decrease)
- **Speed**: Throughput in samples/second
- **Memory**: Current GPU memory allocation
- **Time**: Milliseconds per training iteration

### Final Summary

```
Performance Summary:
   Total samples processed: 200
   Average training speed: 8.7 samples/sec
   Average batch time: 459.3 ms
   Average forward time: 285.1 ms
   Average backward time: 165.7 ms
   Average optimizer time: 8.5 ms
   Final loss: 38.4512
   Peak memory usage: 102.3 MB
```

**What to Analyze:**
- Forward/backward time ratio (typically 1.5-2.0x)
- Memory growth over time
- Loss convergence behavior

## Command Reference

### Model Configuration
```bash
--msa-dim 64              # MSA representation dimension
--pair-dim 128            # Pair representation dimension  
--num-blocks 4            # Number of Evoformer blocks
--num-seqs 16             # Number of MSA sequences
--seq-len 64              # Sequence length (residues)
```

### Training Parameters
```bash
--num-steps 50            # Training iterations
--batch-size 4            # Batch size
--learning-rate 3e-4      # Learning rate
--use-amp                 # Enable mixed precision (FP16)
```

### Profiling Options
```bash
--enable-pytorch-profiler # Enable PyTorch profiler
--enable-memory-profiling # Track memory usage
--enable-all-profiling    # Enable all profiling
--profile-dir PATH        # Output directory
--warmup-steps 3          # Profiler warmup iterations
--profile-steps 5         # Iterations to profile
```

## Code Structure

### Main Classes

**`TinyOpenFoldConfig`**: Model configuration dataclass

**`MSARowAttentionWithPairBias`**: MSA row attention + pair bias
- Projects MSA to Q, K, V
- Adds pair representation as attention bias
- Core innovation of AlphaFold 2

**`MSAColumnAttention`**: MSA column attention
- Transposes to attend across sequences
- Independent of pair representation

**`TriangleMultiplication`**: Triangle multiplicative update
- Gated projections for left and right edges
- Einstein summation for triangle computation
- Separate classes for outgoing/incoming

**`TriangleAttention`**: Triangle self-attention
- Standard multi-head attention over edges
- Two variants: starting and ending nodes

**`OuterProductMean`**: Outer product mean computation
- Projects MSA to lower dimension
- Computes outer product between positions
- Averages across MSA depth

**`EvoformerBlock`**: Complete Evoformer block
- Orchestrates all MSA and pair operations
- Includes layer norms and residual connections

**`TinyOpenFold`**: Main model class
- Input embeddings
- Stack of Evoformer blocks
- Structure module for predictions

### Data Flow

```
Input:
  ├─ MSA tokens (batch, n_seqs, seq_len)
  └─ Pair features (batch, seq_len, seq_len, pair_input_dim)

Embeddings:
  ├─ MSA: (batch, n_seqs, seq_len, msa_dim)
  └─ Pair: (batch, seq_len, seq_len, pair_dim)

Evoformer Blocks (repeated N times):
  ├─ MSA updates:
  │   ├─ Row attention (with pair bias)
  │   ├─ Column attention
  │   └─ Transition
  └─ Pair updates:
      ├─ Outer product mean
      ├─ Triangle multiplication (out/in)
      ├─ Triangle attention (start/end)
      └─ Transition

Structure Module:
  └─ Pair → Distances: (batch, seq_len, seq_len, 1)

Output:
  └─ Predicted distance matrix
```

## Debugging Tips

### Model Not Training (Loss Not Decreasing)

```bash
# Check with smaller problem
python tiny_openfold_v1.py \
    --seq-len 16 \
    --num-seqs 4 \
    --batch-size 2 \
    --num-steps 100

# Increase learning rate
python tiny_openfold_v1.py --learning-rate 1e-3
```

### Numerical Instabilities

```bash
# Use mixed precision for better numerical stability
python tiny_openfold_v1.py --use-amp
```

### Slow Performance

```bash
# Profile to find bottlenecks
python tiny_openfold_v1.py \
    --enable-pytorch-profiler \
    --profile-dir ./debug_profile \
    --num-steps 20

# Reduce problem size
python tiny_openfold_v1.py --seq-len 32 --num-seqs 8
```

## Understanding the Code

### Key Code Sections to Study

1. **MSA Row Attention** (lines ~250-310)
   - See how pair bias is added to attention scores
   - Note the broadcasting across MSA sequences

2. **Triangle Multiplication** (lines ~480-530)
   - Examine the Einstein summation for triangle updates
   - Understand gating mechanism

3. **Evoformer Block** (lines ~620-680)
   - See how MSA and pair updates are orchestrated
   - Note the residual connections

4. **Training Loop** (lines ~900-1050)
   - Profiling integration points
   - Timing and metrics collection

### Profiler Integration Points

The code includes `record_function()` calls for profiling:

```python
with record_function("evoformer_block"):
    with record_function("msa_row_attention"):
        # ... attention code
```

These show up in PyTorch Profiler and help identify bottlenecks.

## Comparison with TinyLLaMA

Similar structure to TinyLLaMA but with protein-specific components:

| Aspect | TinyLLaMA | TinyOpenFold |
|--------|-----------|--------------|
| Core Operation | Causal self-attention | Evoformer (MSA + Pair) |
| Input | Token sequence | MSA + pair features |
| Attention Types | 1 (causal) | 5 (row, column, 2×triangle, pair) |
| Complexity | O(N²) | O(N³) triangle updates |
| Key Innovation | RoPE, GQA | Triangle updates, pair bias |
| Output | Next token | 3D structure (distances) |

## Next Steps

After running the baseline:

1. **Analyze Profiling Results**
   - Open TensorBoard to view timeline
   - Identify hotspot operations
   - Check memory usage patterns

2. **Experiment with Configurations**
   - Try different sequence lengths
   - Vary MSA depth
   - Test different numbers of blocks

3. **Consider Optimizations**
   - Implement flash attention for MSA operations
   - Fuse triangle operations
   - Try mixed precision training

## Resources

### AlphaFold 2 Paper
- Main: https://www.nature.com/articles/s41586-021-03819-2
- Supplement: Detailed architecture (Section 1.6 for Evoformer)

### OpenFold (Production Implementation)
- GitHub: https://github.com/aqlaboratory/openfold
- Documentation: https://openfold.readthedocs.io/

### Parent Directory
- See `../ARCHITECTURE.md` for detailed parameter calculations
- See `../README.md` for project overview

---

**Questions or Issues?**

Check the parent README or examine the code comments for detailed explanations of each component.

