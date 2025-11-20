# Exercise 1: Triton Basics - LayerNorm Kernel

**Duration**: 45 minutes  
**Difficulty**: Beginner to Intermediate

## Learning Objectives

By the end of this exercise, you will:
1. Understand Triton kernel structure and syntax
2. Analyze memory access patterns in GPU kernels
3. Optimize LayerNorm for different input sizes
4. Compare Triton vs PyTorch performance

## Background

LayerNorm is a fundamental operation in transformer architectures including Evoformer. It normalizes activations across features:

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma$$

Where:
- $\mu$ = mean across features
- $\sigma^2$ = variance across features
- $\gamma$ = learned scale parameter
- $\epsilon$ = small constant for numerical stability

## Part 1: Understanding the Triton LayerNorm Kernel (15 minutes)

### Task 1.1: Analyze the Kernel Structure

Open `../tiny_openfold_v3.py` and locate the `layernorm_kernel` function (around line 44).

**Questions:**

1. How many passes does the kernel make through the input data?
   - Hint: Count the `for` loops

2. What data is computed and stored in registers vs global memory?
   - Registers: _____________
   - Global memory: _____________

3. Why is the block size (BLOCK_SIZE) a `tl.constexpr`?
   - Your answer: _____________

### Task 1.2: Memory Access Pattern Analysis

```python
@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements, eps, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # Pass 1: Compute mean
    mean = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x_vals = tl.load(x_ptr + row_idx * n_elements + offsets, mask=mask, other=0.0)
        mean += tl.sum(x_vals, axis=0)
```

**Questions:**

1. Is this memory access pattern coalesced? Why or why not?
   - Your answer: _____________

2. How many times is each element of `x` loaded from memory?
   - Your answer: _____________

3. What is the purpose of the `mask` parameter in `tl.load`?
   - Your answer: _____________

## Part 2: Implementing a Simpler RMS Norm (15 minutes)

RMS (Root Mean Square) Normalization is a simplified version of LayerNorm that doesn't subtract the mean:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

### Task 2.1: Implement RMS Norm Kernel

Create a new file `exercise1_rmsnorm.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    TODO: Implement RMS normalization kernel.
    
    Hints:
    1. You only need one pass for variance (no mean computation)
    2. Compute: variance = mean(x^2)
    3. Apply: output = x / sqrt(variance + eps) * weight
    """
    row_idx = tl.program_id(0)
    
    # TODO: Compute variance (mean of squares)
    variance = 0.0
    # Your code here...
    
    # TODO: Compute inverse std
    # Your code here...
    
    # TODO: Normalize and scale
    # Your code here...

# Wrapper class
class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test your implementation
def test_rmsnorm():
    dim = 128
    batch = 1024
    
    x = torch.randn(batch, dim, device='cuda')
    
    # Your Triton implementation
    triton_norm = TritonRMSNorm(dim).cuda()
    triton_output = triton_norm(x)
    
    # Reference implementation
    def reference_rmsnorm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight
    
    ref_output = reference_rmsnorm(x, triton_norm.weight, triton_norm.eps)
    
    # Check correctness
    max_diff = (triton_output - ref_output).abs().max()
    print(f"Max difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Too large difference: {max_diff}"
    print("✓ Correctness test passed!")

if __name__ == "__main__":
    test_rmsnorm()
```

**Checkpoint**: Run your implementation and verify correctness.

## Part 3: Performance Optimization (15 minutes)

### Task 3.1: Block Size Tuning

Experiment with different block sizes to find the optimal configuration.

Create `exercise1_benchmark.py`:

```python
import torch
import time
from exercise1_rmsnorm import TritonRMSNorm

def benchmark_block_size(dim=128, batch=4096, block_sizes=[64, 128, 256, 512, 1024]):
    """Benchmark different block sizes."""
    x = torch.randn(batch, dim, device='cuda')
    
    results = {}
    
    for block_size in block_sizes:
        # Modify your RMSNorm to accept block_size parameter
        # Then benchmark here
        
        # Warmup
        model = TritonRMSNorm(dim).cuda()
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        results[block_size] = elapsed / 100 * 1000  # ms
        print(f"Block size {block_size:4d}: {results[block_size]:.3f} ms")
    
    # Find best
    best_block = min(results, key=results.get)
    print(f"\nBest block size: {best_block} ({results[best_block]:.3f} ms)")
    
    return results

if __name__ == "__main__":
    benchmark_block_size()
```

**Questions:**

1. Which block size is fastest? Why?
   - Your answer: _____________

2. How does performance change with block size?
   - Your observations: _____________

3. What hardware constraints affect the optimal block size?
   - Your answer: _____________

### Task 3.2: Compare with PyTorch

Add comparison code:

```python
def compare_with_pytorch(dim=128, batch=4096):
    """Compare Triton vs PyTorch LayerNorm."""
    x = torch.randn(batch, dim, device='cuda')
    
    # Triton RMSNorm
    triton_norm = TritonRMSNorm(dim).cuda()
    
    # Warmup
    for _ in range(10):
        _ = triton_norm(x)
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = triton_norm(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 100
    
    # PyTorch LayerNorm
    pytorch_norm = torch.nn.LayerNorm(dim).cuda()
    
    # Warmup
    for _ in range(10):
        _ = pytorch_norm(x)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = pytorch_norm(x)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 100
    
    print(f"\nPerformance Comparison (dim={dim}, batch={batch}):")
    print(f"  Triton RMSNorm: {triton_time*1000:.3f} ms")
    print(f"  PyTorch LayerNorm: {pytorch_time*1000:.3f} ms")
    print(f"  Speedup: {pytorch_time/triton_time:.2f}x")

if __name__ == "__main__":
    compare_with_pytorch()
```

**Expected Results:**
- Triton RMSNorm should be 1.5-2.5x faster than PyTorch LayerNorm
- RMSNorm is simpler (no mean computation) so faster than LayerNorm

## Part 4: Advanced Challenge (Optional)

### Task 4.1: Fused RMSNorm + Activation

Implement a fused kernel that combines RMSNorm with ReLU activation:

```python
@triton.jit
def fused_rmsnorm_relu_kernel(
    x_ptr, weight_ptr, output_ptr,
    n_elements, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """
    Fuse RMSNorm with ReLU activation.
    output = ReLU(RMSNorm(x))
    """
    # TODO: Implement
    pass
```

**Question**: How much speedup do you get from fusion compared to separate operations?

## Key Takeaways

1. **Triton Syntax**: Triton uses Python-like syntax but compiles to efficient GPU code
2. **Memory Patterns**: Coalesced memory access is crucial for performance
3. **Block Sizes**: Optimal block size depends on problem size and hardware
4. **Fusion**: Combining operations reduces memory bandwidth requirements
5. **Trade-offs**: Simpler operations (RMSNorm vs LayerNorm) can be faster

## Solutions

Solutions are provided in `solutions/exercise1_solution.py`.

Compare your implementation with the solution after attempting the exercise.

## Next Steps

Proceed to Exercise 2 to learn about optimizing triangle multiplicative updates!

