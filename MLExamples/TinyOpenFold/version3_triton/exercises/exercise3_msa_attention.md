# Exercise 3: Flash Attention for MSA Operations

**Duration**: 75 minutes  
**Difficulty**: Advanced

## Learning Objectives

By the end of this exercise, you will:
1. Understand Flash Attention algorithm and implementation
2. Apply Flash Attention to MSA row and column attention
3. Analyze memory efficiency improvements
4. Handle pair bias in attention mechanisms
5. Optimize for different sequence lengths and MSA depths

## Background

Multiple Sequence Alignment (MSA) attention is central to AlphaFold's ability to leverage evolutionary information for structure prediction. However, standard attention has O(N²) memory complexity, which becomes prohibitive for long sequences or deep MSAs.

### Standard Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Memory Complexity**: O(batch × heads × seq_len²)  
**Problem**: Materializes the full attention matrix

### Flash Attention

Flash Attention uses tiling and online softmax to reduce memory to O(N):

**Key Innovation**: Never materialize the full attention matrix!

**Algorithm**:
1. Tile Q into blocks that fit in SRAM
2. For each Q block:
   - Stream K, V blocks from HBM
   - Compute attention incrementally
   - Use online softmax for numerical stability
3. Result: O(N) memory, same O(N²) compute

## Part 1: Understanding Flash Attention (20 minutes)

### Task 1.1: Analyze the Kernel

Open `../tiny_openfold_v3.py` and locate the `flash_attention_kernel` (around line 118).

**Questions:**

1. What is stored in the accumulators?
   ```python
   output_acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
   max_scores = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)
   sum_exp = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
   ```
   - `output_acc`: _____________
   - `max_scores`: _____________
   - `sum_exp`: _____________

2. Why do we track `max_scores` and update it incrementally?
   - Your answer: _____________

3. What is the "online softmax" algorithm doing?
   ```python
   new_max = tl.maximum(max_scores, block_max)
   decay = tl.exp(max_scores - new_max)
   output_acc = output_acc * decay[:, None]
   ```
   - Your explanation: _____________

### Task 1.2: Memory Analysis

For standard attention vs Flash Attention:

```python
def memory_comparison():
    """Compare memory usage of standard vs flash attention."""
    
    batch = 4
    n_heads = 4
    seq_len = 64
    head_dim = 16
    
    # Standard attention
    # Stores: Q, K, V, attention_matrix, output
    qkv_memory = 3 * batch * n_heads * seq_len * head_dim * 4  # bytes (FP32)
    attention_matrix = batch * n_heads * seq_len * seq_len * 4
    output_memory = batch * n_heads * seq_len * head_dim * 4
    standard_total = qkv_memory + attention_matrix + output_memory
    
    # Flash attention
    # Stores: Q, K, V (blocks), accumulators, output
    # No full attention matrix!
    flash_qkv = qkv_memory  # Same input memory
    block_size = 64
    accumulator_memory = batch * n_heads * block_size * head_dim * 4
    flash_total = flash_qkv + accumulator_memory + output_memory
    
    print(f"Memory Comparison (batch={batch}, seq_len={seq_len}):")
    print(f"")
    print(f"Standard Attention:")
    print(f"  Q, K, V: {qkv_memory/1e6:.2f} MB")
    print(f"  Attention matrix: {attention_matrix/1e6:.2f} MB")
    print(f"  Output: {output_memory/1e6:.2f} MB")
    print(f"  Total: {standard_total/1e6:.2f} MB")
    print(f"")
    print(f"Flash Attention:")
    print(f"  Q, K, V: {flash_qkv/1e6:.2f} MB")
    print(f"  Accumulators: {accumulator_memory/1e6:.2f} MB")
    print(f"  Output: {output_memory/1e6:.2f} MB")
    print(f"  Total: {flash_total/1e6:.2f} MB")
    print(f"")
    print(f"Memory Reduction: {(1 - flash_total/standard_total)*100:.1f}%")
    
    # Show scaling
    print(f"\nScaling with sequence length:")
    print(f"{'Seq Len':>10} {'Standard (MB)':>15} {'Flash (MB)':>15} {'Reduction':>12}")
    print("-" * 55)
    
    for seq_len in [16, 32, 64, 128, 256, 512]:
        std_mem = (qkv_memory/1e6) + (batch * n_heads * seq_len * seq_len * 4 / 1e6)
        flash_mem = (qkv_memory/1e6) + (accumulator_memory/1e6)
        reduction = (1 - flash_mem/std_mem)*100
        print(f"{seq_len:>10} {std_mem:>15.2f} {flash_mem:>15.2f} {reduction:>11.1f}%")

if __name__ == "__main__":
    memory_comparison()
```

**Questions:**

1. How does memory scale with sequence length for each approach?
   - Standard: O(___)
   - Flash: O(___)

2. At what sequence length does the memory reduction become significant?
   - Your answer: _____________

3. What is the theoretical maximum memory reduction?
   - Your calculation: _____________

## Part 2: MSA Row Attention with Pair Bias (25 minutes)

MSA row attention is unique because it includes a **pair bias** term:

$$\text{Attention}(Q, K, V, b) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + b\right) V$$

Where $b$ is derived from the pair representation.

### Task 2.1: Understand Pair Bias Integration

**Current Implementation**: Applies pair bias after Flash Attention  
**Optimal Implementation**: Integrate pair bias into the Flash Attention kernel

Create `exercise3_pair_bias.py`:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_with_bias_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim, scale,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """
    Flash Attention with integrated pair bias.
    
    TODO: Modify the Flash Attention kernel to incorporate pair bias
    during the attention score computation.
    
    Hints:
    1. Load bias block corresponding to current Q and K blocks
    2. Add bias to scores before softmax
    3. Continue with standard Flash Attention algorithm
    """
    
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    
    # Calculate offsets
    head_offset = batch_idx * num_heads * seq_len * HEAD_DIM + head_idx * seq_len * HEAD_DIM
    
    # Load Q block
    q_start = q_block_idx * BLOCK_SIZE_Q
    q_range = tl.arange(0, BLOCK_SIZE_Q)
    d_range = tl.arange(0, HEAD_DIM)
    
    q_offsets = head_offset + (q_start + q_range[:, None]) * HEAD_DIM + d_range[None, :]
    q_mask = (q_start + q_range[:, None]) < seq_len
    q_block = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    output_acc = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)
    max_scores = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    
    # Process K, V blocks
    num_k_blocks = tl.cdiv(seq_len, BLOCK_SIZE_K)
    for k_block_idx in range(num_k_blocks):
        k_start = k_block_idx * BLOCK_SIZE_K
        k_range = tl.arange(0, BLOCK_SIZE_K)
        
        # Load K, V blocks (same as before)
        k_offsets = head_offset + (k_start + k_range[:, None]) * HEAD_DIM + d_range[None, :]
        k_mask = (k_start + k_range[:, None]) < seq_len
        k_block = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        
        v_offsets = head_offset + (k_start + k_range[:, None]) * HEAD_DIM + d_range[None, :]
        v_mask = (k_start + k_range[:, None]) < seq_len
        v_block = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)
        
        # Compute attention scores
        scores = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # TODO: Load and add pair bias
        # bias shape: [batch, num_heads, seq_len, seq_len]
        # Need to load the [q_start:q_start+BLOCK_SIZE_Q, k_start:k_start+BLOCK_SIZE_K] block
        
        # Your code here to load and add bias
        # bias_offsets = ...
        # bias_block = tl.load(bias_ptr + bias_offsets, ...)
        # scores = scores + bias_block
        
        # Continue with online softmax (same as before)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_scores, block_max)
        
        decay = tl.exp(max_scores - new_max)
        output_acc = output_acc * decay[:, None]
        
        exp_scores = tl.exp(scores - new_max[:, None])
        sum_exp = sum_exp * decay + tl.sum(exp_scores, axis=1)
        max_scores = new_max
        
        output_acc += tl.dot(exp_scores, v_block)
    
    # Final normalization
    output = output_acc / sum_exp[:, None]
    
    # Store output
    out_offsets = head_offset + (q_start + q_range[:, None]) * HEAD_DIM + d_range[None, :]
    out_mask = (q_start + q_range[:, None]) < seq_len
    tl.store(output_ptr + out_offsets, output, mask=out_mask)
```

### Task 2.2: Test Your Implementation

```python
def test_flash_attention_with_bias():
    """Test Flash Attention with pair bias."""
    
    batch = 2
    n_heads = 4
    seq_len = 64
    head_dim = 16
    
    device = 'cuda'
    
    # Create test data
    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    bias = torch.randn(batch, n_heads, seq_len, seq_len, device=device)
    
    # Reference implementation
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + bias
    attn_weights = torch.softmax(scores, dim=-1)
    ref_output = torch.matmul(attn_weights, v)
    
    # Your Triton implementation
    # triton_output = ... (call your kernel)
    
    # Compare
    # max_diff = (triton_output - ref_output).abs().max()
    # print(f"Max difference: {max_diff:.2e}")
    # assert max_diff < 1e-3, f"Too large difference: {max_diff}"
    
    print("Test your implementation here!")

if __name__ == "__main__":
    test_flash_attention_with_bias()
```

## Part 3: MSA Column Attention (15 minutes)

MSA column attention attends across sequences for each residue position.

**Difference from Row Attention:**
- Row: Attention over residues (seq_len dimension)
- Column: Attention over sequences (n_seqs dimension)

### Task 3.1: Analyze Column Attention

Look at `TritonMSAColumnAttention` in `tiny_openfold_v3.py`.

**Questions:**

1. How is the MSA tensor reshaped for column attention?
   - Original: (batch, n_seqs, seq_len, msa_dim)
   - Reshaped: _____________

2. Why is column attention typically faster than row attention?
   - Your answer: _____________

3. What is the memory complexity for column attention with Flash Attention?
   - Answer: O(___)

### Task 3.2: Performance Comparison

Create `exercise3_compare.py`:

```python
import torch
import time
import sys
sys.path.append('..')
from tiny_openfold_v3 import (
    TritonMSARowAttention,
    TritonMSAColumnAttention,
    TinyOpenFoldConfig
)

def compare_msa_attention(seq_len=64, n_seqs=16, batch=4):
    """Compare row vs column attention performance."""
    
    config = TinyOpenFoldConfig(
        msa_dim=64,
        pair_dim=128,
        n_seqs=n_seqs,
        max_seq_len=seq_len
    )
    
    device = 'cuda'
    
    # Create test data
    msa = torch.randn(batch, n_seqs, seq_len, config.msa_dim, device=device)
    pair = torch.randn(batch, seq_len, seq_len, config.pair_dim, device=device)
    
    # Row attention
    row_attn = TritonMSARowAttention(config).to(device)
    
    # Warmup
    for _ in range(10):
        _ = row_attn(msa, pair)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = row_attn(msa, pair)
    torch.cuda.synchronize()
    row_time = (time.time() - start) / 50
    
    # Column attention
    col_attn = TritonMSAColumnAttention(config).to(device)
    
    # Warmup
    for _ in range(10):
        _ = col_attn(msa)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        _ = col_attn(msa)
    torch.cuda.synchronize()
    col_time = (time.time() - start) / 50
    
    print(f"MSA Attention Comparison (seq_len={seq_len}, n_seqs={n_seqs}):")
    print(f"  Row attention: {row_time*1000:.3f} ms")
    print(f"  Column attention: {col_time*1000:.3f} ms")
    print(f"  Speedup (row/col): {row_time/col_time:.2f}x")
    
    # Memory
    row_mem = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    _ = col_attn(msa)
    col_mem = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"  Row memory: {row_mem:.1f} MB")
    print(f"  Column memory: {col_mem:.1f} MB")

if __name__ == "__main__":
    compare_msa_attention()
```

## Part 4: Optimization Challenge (15 minutes)

### Task 4.1: Block Size Tuning

Flash Attention performance depends heavily on block sizes.

**Trade-offs:**
- Larger blocks: More data reuse, but higher SRAM usage
- Smaller blocks: Less SRAM usage, but more kernel launches

Create `exercise3_tune.py`:

```python
def tune_block_sizes(seq_len=64):
    """Find optimal block sizes for Flash Attention."""
    
    block_sizes = [16, 32, 64, 128]
    
    results = {}
    
    for block_q in block_sizes:
        for block_k in block_sizes:
            # Modify Flash Attention to accept block sizes as parameters
            # Then benchmark here
            
            # Check if configuration fits in SRAM
            # MI300X: ~164KB shared memory per CU
            sram_usage = block_q * block_k * 4  # bytes (FP32)
            
            if sram_usage > 160000:  # 160KB limit
                print(f"Block ({block_q}, {block_k}): Too large for SRAM")
                continue
            
            # Benchmark this configuration
            # ...
            
            results[(block_q, block_k)] = {
                'time': 0.0,  # Your measurement
                'sram_usage': sram_usage
            }
    
    # Find best configuration
    # ...
```

**Questions:**

1. What is the optimal block size for seq_len=64?
   - Your answer: _____________

2. How does optimal block size change with sequence length?
   - Your observations: _____________

3. What hardware constraints limit block size?
   - SRAM size: _____________
   - Register file: _____________
   - Occupancy: _____________

## Key Takeaways

1. **Flash Attention**: Reduces memory from O(N²) to O(N) while maintaining O(N²) compute
2. **Online Softmax**: Enables incremental computation without materializing full attention matrix
3. **Pair Bias**: Can be integrated into Flash Attention for efficiency
4. **Row vs Column**: Different attention patterns have different performance characteristics
5. **Block Size Tuning**: Critical for optimal performance, depends on hardware and problem size

## Discussion Questions

1. Why is attention a bottleneck in protein structure prediction?
2. How would Flash Attention help with very deep MSAs (thousands of sequences)?
3. What other attention variants could benefit from Flash Attention?
4. How does Flash Attention compare to other efficient attention methods (e.g., sparse attention)?

## Solutions

Solutions are provided in `solutions/exercise3_solution.py`.

## Congratulations!

You've completed all three exercises on Triton kernel optimization for TinyOpenFold. You now understand:

- Basic Triton kernel programming (LayerNorm)
- Complex operations and hybrid optimization (Triangle updates)
- Advanced memory-efficient algorithms (Flash Attention)

Continue experimenting with different configurations and optimizations!

