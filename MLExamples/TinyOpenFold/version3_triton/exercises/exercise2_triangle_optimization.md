# Exercise 2: Triangle Multiplicative Updates

**Duration**: 60 minutes  
**Difficulty**: Intermediate to Advanced

## Learning Objectives

By the end of this exercise, you will:
1. Understand triangle multiplicative updates in protein structure prediction
2. Analyze the computational complexity of triangle operations
3. Identify optimization opportunities through kernel fusion
4. Implement performance profiling for complex operations

## Background

Triangle multiplicative updates are a key innovation in AlphaFold 2's Evoformer architecture. They implement geometric reasoning: if residues i-j and j-k are spatially close, then i-k should also be considered close.

### Mathematical Formulation

**Outgoing Triangle Update:**
$$z_{ij} = \sum_k \text{gate}(p_{ik}) \odot W_{\text{left}} p_{ik} \times \text{gate}(p_{jk}) \odot W_{\text{right}} p_{jk}$$

**Incoming Triangle Update:**
$$z_{ij} = \sum_k \text{gate}(p_{ki}) \odot W_{\text{left}} p_{ki} \times \text{gate}(p_{kj}) \odot W_{\text{right}} p_{kj}$$

Where:
- $p_{ij}$ = pair representation between residues i and j
- $\text{gate}(x) = \sigma(W_g x)$ = sigmoid gating function
- $\odot$ = element-wise multiplication

### Computational Complexity

- **Time Complexity**: $O(N^3 \cdot D)$ where N = sequence length, D = pair dimension
- **Space Complexity**: $O(N^2 \cdot D)$
- **Operations**: 4 Linear projections + gating + einsum

This is one of the most expensive operations in Evoformer!

## Part 1: Understanding Triangle Updates (20 minutes)

### Task 1.1: Analyze the Implementation

Open `../tiny_openfold_v3.py` and locate the `TritonTriangleMultiplication` class (around line 487).

**Questions:**

1. How many Linear layers are used in triangle multiplication?
   - Count: _____________
   - Purpose of each: _____________

2. What is the difference between "outgoing" and "incoming" updates?
   ```
   Outgoing einsum: 'bikc,bjkc->bijc'
   Incoming einsum: 'bkic,bkjc->bijc'
   ```
   - Your explanation: _____________

3. Why does the implementation use PyTorch Linear layers instead of Triton kernels?
   - Your answer: _____________

### Task 1.2: Complexity Analysis

For a sequence length N=64 and pair_dim D=128:

```python
def analyze_complexity():
    """Calculate the computational cost of triangle updates."""
    
    N = 64  # sequence length
    D = 128  # pair dimension
    batch = 4
    
    # Input: pair representation
    pair_elements = batch * N * N * D
    
    # Linear projections (4 of them: left_proj, right_proj, left_gate, right_gate)
    # Each: (N*N*D) @ (D*D) matrix multiplication
    proj_flops = 4 * (batch * N * N * D * D)
    
    # Gating (sigmoid + element-wise multiply, 2 pairs)
    gate_flops = 2 * (batch * N * N * D * 5)  # approx 5 ops per sigmoid
    
    # Einsum: 'bikc,bjkc->bijc'
    # For each (i,j) pair, sum over k: O(N)
    # Total: N * N pairs, each needing N * D multiplications
    einsum_flops = batch * N * N * N * D
    
    # Output projection and gate
    output_flops = 2 * (batch * N * N * D * D)
    
    total_flops = proj_flops + gate_flops + einsum_flops + output_flops
    
    print(f"Triangle Update Complexity Analysis:")
    print(f"  Sequence length: {N}")
    print(f"  Pair dimension: {D}")
    print(f"  Input size: {pair_elements:,} elements")
    print(f"")
    print(f"  Linear projections: {proj_flops:,} FLOPs ({proj_flops/total_flops*100:.1f}%)")
    print(f"  Gating operations: {gate_flops:,} FLOPs ({gate_flops/total_flops*100:.1f}%)")
    print(f"  Einsum computation: {einsum_flops:,} FLOPs ({einsum_flops/total_flops*100:.1f}%)")
    print(f"  Output operations: {output_flops:,} FLOPs ({output_flops/total_flops*100:.1f}%)")
    print(f"")
    print(f"  Total: {total_flops:,} FLOPs")
    print(f"  Total: {total_flops/1e9:.3f} GFLOPs")
```

**Questions:**

1. Which operation dominates the computation?
   - Your answer: _____________

2. How does the cost scale with sequence length?
   - Linear projections: O(___)
   - Einsum: O(___)
   - Overall: O(___)

3. What happens if we double the sequence length (64 → 128)?
   - FLOPs increase by: _____________

## Part 2: Profiling Triangle Operations (20 minutes)

### Task 2.1: Create Profiling Script

Create `exercise2_profile.py`:

```python
import torch
import time
import sys
sys.path.append('..')
from tiny_openfold_v3 import TritonTriangleMultiplication, TinyOpenFoldConfig

def profile_triangle_operation(seq_len=64, batch_size=4, num_runs=50):
    """Profile triangle multiplicative update."""
    
    config = TinyOpenFoldConfig(
        pair_dim=128,
        max_seq_len=seq_len
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    triangle_mult = TritonTriangleMultiplication(config, outgoing=True).to(device)
    
    # Create input
    pair = torch.randn(batch_size, seq_len, seq_len, config.pair_dim, device=device)
    
    # Warmup
    for _ in range(10):
        _ = triangle_mult(pair)
    
    # Profile forward pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_runs):
        output = triangle_mult(pair)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs
    
    # Calculate throughput
    # FLOPs calculation (approximate)
    linear_flops = 6 * batch_size * seq_len * seq_len * config.pair_dim * config.pair_dim
    einsum_flops = batch_size * seq_len * seq_len * seq_len * config.pair_dim
    total_flops = linear_flops + einsum_flops
    
    flops_per_sec = total_flops / avg_time
    
    print(f"Triangle Update Profile (seq_len={seq_len}, batch={batch_size}):")
    print(f"  Average time: {avg_time*1000:.3f} ms")
    print(f"  Throughput: {flops_per_sec/1e9:.2f} GFLOPS")
    
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  GPU memory: {memory_mb:.1f} MB")
    
    return avg_time, flops_per_sec

def profile_scaling(batch_size=4):
    """Profile how performance scales with sequence length."""
    
    seq_lengths = [16, 32, 64, 128]
    
    print("Scaling Analysis:")
    print("=" * 70)
    
    results = []
    for seq_len in seq_lengths:
        try:
            avg_time, flops = profile_triangle_operation(seq_len, batch_size, num_runs=20)
            results.append((seq_len, avg_time, flops))
            print()
        except RuntimeError as e:
            print(f"  Skipped seq_len={seq_len}: {e}")
            print()
    
    # Analyze scaling
    print("Scaling Summary:")
    print(f"{'Seq Len':>10} {'Time (ms)':>12} {'GFLOPS':>12} {'Time Ratio':>12}")
    print("-" * 50)
    
    baseline_time = results[0][1] if results else 0
    for seq_len, avg_time, flops in results:
        ratio = avg_time / baseline_time if baseline_time > 0 else 0
        print(f"{seq_len:>10} {avg_time*1000:>12.3f} {flops/1e9:>12.2f} {ratio:>12.2f}x")

if __name__ == "__main__":
    profile_scaling()
```

Run the profiling script and answer:

**Questions:**

1. How does time scale with sequence length?
   - Your observations: _____________

2. What is the achieved GFLOPS? How does it compare to peak GPU FLOPS?
   - Your answer: _____________

3. Is the operation compute-bound or memory-bound?
   - Your reasoning: _____________

## Part 3: Optimization Analysis (20 minutes)

### Task 3.1: Identify Bottlenecks

Current implementation has several operations:
1. LayerNorm (Triton kernel)
2. 4 Linear projections (PyTorch/rocBLAS)
3. 2 Sigmoid activations (PyTorch)
4. 2 Element-wise multiplications (PyTorch)
5. Einsum (PyTorch)

**Questions:**

1. Which operations could benefit from fusion?
   - Your ideas: _____________

2. What would be the memory bandwidth savings from fusing operations?
   - Your calculation: _____________

3. Why keep Linear projections in PyTorch instead of Triton?
   - Your understanding: _____________

### Task 3.2: Roofline Analysis

Calculate the arithmetic intensity:

```python
def roofline_analysis():
    """Perform roofline analysis for triangle update."""
    
    N = 64
    D = 128
    batch = 4
    
    # FLOPs
    total_flops = 6 * batch * N * N * D * D + batch * N * N * N * D
    
    # Memory transfers (bytes)
    # Input: pair (N*N*D*4 bytes)
    # Weights: 6 weight matrices (D*D*4 bytes each)
    # Intermediate: gates and projections (2*N*N*D*4 bytes)
    # Output: (N*N*D*4 bytes)
    input_mem = batch * N * N * D * 4
    weights_mem = 6 * D * D * 4
    intermediate_mem = 2 * batch * N * N * D * 4
    output_mem = batch * N * N * D * 4
    
    total_mem = input_mem + weights_mem + intermediate_mem + output_mem
    
    # Arithmetic intensity (FLOPs per byte)
    arithmetic_intensity = total_flops / total_mem
    
    print(f"Roofline Analysis:")
    print(f"  Total FLOPs: {total_flops/1e9:.3f} GFLOPS")
    print(f"  Total Memory: {total_mem/1e6:.2f} MB")
    print(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
    print(f"")
    
    # Compare with hardware specs (MI300X)
    peak_flops = 163e12  # 163 TFLOPS FP32
    memory_bandwidth = 5.3e12  # 5.3 TB/s
    
    # Compute bound if: time_compute > time_memory
    # time_compute = FLOPs / peak_flops
    # time_memory = bytes / bandwidth
    
    time_compute = total_flops / peak_flops
    time_memory = total_mem / memory_bandwidth
    
    print(f"  Time (compute-bound): {time_compute*1000:.3f} ms")
    print(f"  Time (memory-bound): {time_memory*1000:.3f} ms")
    print(f"")
    
    if time_compute > time_memory:
        print(f"  → Compute-bound operation")
        print(f"  → Linear projections dominate (rocBLAS optimal)")
    else:
        print(f"  → Memory-bound operation")
        print(f"  → Kernel fusion would help")

if __name__ == "__main__":
    roofline_analysis()
```

## Part 4: Alternative Implementation (Optional Challenge)

### Task 4.1: Implement Gated Linear Unit Fusion

Try implementing a fused gated projection:

```python
@triton.jit
def fused_gated_projection_kernel(
    input_ptr, weight_proj_ptr, weight_gate_ptr, output_ptr,
    batch_size, seq_len, in_dim, out_dim,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Fuse: output = proj(input) * sigmoid(gate(input))
    
    This would replace:
        left = left_proj(x) * sigmoid(left_gate(x))
    
    With a single fused kernel.
    """
    # TODO: Implement
    # Hints:
    # 1. Tile the matrix multiplication
    # 2. Apply sigmoid to gate values
    # 3. Element-wise multiply proj and gate results
    # 4. All in one kernel!
    pass
```

**Question**: How much speedup would this fusion provide?

## Key Takeaways

1. **Complexity**: Triangle updates are O(N³·D) - one of the most expensive Evoformer operations
2. **Bottlenecks**: Linear projections dominate computation (good for rocBLAS)
3. **Hybrid Optimization**: Use Triton for memory-bound ops, PyTorch for compute-bound ops
4. **Roofline Model**: Helps identify if operation is compute or memory bound
5. **Scaling**: Understanding how operations scale with input size is crucial

## Discussion Questions

1. Why are triangle updates necessary for protein structure prediction?
2. How would you optimize triangle updates for very long sequences (N > 256)?
3. What trade-offs exist between computation and accuracy in triangle updates?

## Solutions

Solutions are provided in `solutions/exercise2_solution.py`.

## Next Steps

Proceed to Exercise 3 to learn about Flash Attention for MSA operations!

