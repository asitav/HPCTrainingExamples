# TinyOpenFold V2 Workshop Exercises

Hands-on exercises for understanding kernel fusion optimizations and ROCm profiling tools.

## Prerequisites

- Completed TinyOpenFold V1 baseline exercises
- Familiarity with PyTorch attention mechanisms
- Basic understanding of GPU memory hierarchy
- ROCm environment (for ROCm-specific exercises)

## Exercise Structure

Each exercise includes:
- **Learning Objectives**: What you'll learn
- **Estimated Time**: Expected completion time
- **Setup**: Required preparation
- **Tasks**: Step-by-step instructions
- **Verification**: How to check your results
- **Questions**: Deeper understanding prompts

## Exercise 1: Kernel Fusion Impact Analysis

**Duration**: 30 minutes  
**Objectives**: Quantify the performance impact of kernel fusion optimizations

### Setup

```bash
cd /path/to/TinyOpenFold/version2_pytorch_fused
```

### Part A: Baseline vs Fused Comparison

**Task 1: Run Baseline (All Fusions Disabled)**

```bash
python tiny_openfold_v2.py \
    --disable-all-fusion \
    --batch-size 4 \
    --seq-len 64 \
    --num-steps 50 \
    --profile-dir ./exercises/results/baseline
```

Record the following metrics from the output:
- Average training speed (samples/sec): _______
- Average batch time (ms): _______
- Peak memory usage (MB): _______
- Final loss: _______

**Task 2: Run Fully Fused Version**

```bash
python tiny_openfold_v2.py \
    --enable-all-fusion \
    --batch-size 4 \
    --seq-len 64 \
    --num-steps 50 \
    --profile-dir ./exercises/results/fused
```

Record the following metrics:
- Average training speed (samples/sec): _______
- Average batch time (ms): _______
- Peak memory usage (MB): _______
- Kernel reduction (%): _______

**Task 3: Calculate Improvements**

```bash
# Speedup = Fused Speed / Baseline Speed
echo "Speedup: $(python -c 'print(FUSED_SPEED / BASELINE_SPEED)')"

# Memory Reduction = (1 - Fused Memory / Baseline Memory) * 100
echo "Memory Reduction: $(python -c 'print((1 - FUSED_MEM / BASELINE_MEM) * 100)')%"
```

Fill in:
- Speedup: _______ x
- Memory reduction: _______ %
- Expected speedup range: 1.5-2.2x
- Expected memory reduction: 15-30% (50-80% with larger sequences)

### Verification

Your results should show:
- ✓ Speedup between 1.5-2.2x
- ✓ Memory reduction of 15-30%
- ✓ Kernel reduction of 60-80%
- ✓ Similar final loss values (within 5%)

### Questions

1. **Why is the memory reduction smaller than expected at seq_len=64?**
   - Hint: Flash Attention benefits scale with sequence length

2. **Which phase (forward, backward, optimizer) benefited most from fusion?**
   - Check the timing breakdown in the output

3. **How does the speedup change with batch size?**
   - Try batch sizes 2, 4, 8 and observe the trend

## Exercise 2: Ablation Study - Individual Fusion Contributions

**Duration**: 45 minutes  
**Objectives**: Understand the contribution of each fusion technique

### Setup

Create a results table:

```bash
mkdir -p exercises/results/ablation
cd exercises/results/ablation
echo "Configuration,Speed(s/s),Memory(MB),Time(ms)" > ablation_results.csv
```

### Part A: Test Each Fusion Independently

**Task 1: Only MSA QKV Fusion**

```bash
python ../../../tiny_openfold_v2.py \
    --disable-qkv-fusion-triangle \
    --disable-flash-attention \
    --disable-triangle-fusion \
    --batch-size 4 \
    --num-steps 50 \
    > msa_qkv_only.log 2>&1

# Extract metrics
grep "Average training speed" msa_qkv_only.log
```

**Task 2: Only Triangle QKV Fusion**

```bash
python ../../../tiny_openfold_v2.py \
    --disable-qkv-fusion-msa \
    --disable-flash-attention \
    --disable-triangle-fusion \
    --batch-size 4 \
    --num-steps 50 \
    > triangle_qkv_only.log 2>&1
```

**Task 3: Only Flash Attention**

```bash
python ../../../tiny_openfold_v2.py \
    --disable-qkv-fusion-msa \
    --disable-qkv-fusion-triangle \
    --disable-triangle-fusion \
    --batch-size 4 \
    --num-steps 50 \
    > flash_attn_only.log 2>&1
```

**Task 4: Only Triangle Gate/Proj Fusion**

```bash
python ../../../tiny_openfold_v2.py \
    --disable-qkv-fusion-msa \
    --disable-qkv-fusion-triangle \
    --disable-flash-attention \
    --batch-size 4 \
    --num-steps 50 \
    > triangle_fusion_only.log 2>&1
```

### Part B: Combine Top Performers

**Task 5: Best Two Fusions**

Based on your results from Part A, enable the two most impactful fusions and measure performance.

```bash
python ../../../tiny_openfold_v2.py \
    --enable-<FUSION_1> \
    --enable-<FUSION_2> \
    --batch-size 4 \
    --num-steps 50 \
    > top_two_fusions.log 2>&1
```

### Analysis Template

| Configuration | Speed (s/s) | Speedup vs Baseline | Memory (MB) | Time (ms) |
|--------------|-------------|---------------------|-------------|-----------|
| Baseline     | _____       | 1.00x               | _____       | _____     |
| MSA QKV      | _____       | _____               | _____       | _____     |
| Triangle QKV | _____       | _____               | _____       | _____     |
| Flash Attn   | _____       | _____               | _____       | _____     |
| Triangle F   | _____       | _____               | _____       | _____     |
| All Fusions  | _____       | _____               | _____       | _____     |

### Questions

1. **Which fusion provided the largest speedup?**
   - Expected: Flash Attention for longer sequences, QKV fusion for shorter sequences

2. **Are the fusion benefits additive?**
   - Compare: (MSA QKV speedup + Triangle QKV speedup) vs (MSA+Triangle QKV speedup)

3. **Why does Flash Attention have minimal impact at seq_len=64?**
   - Hint: Flash Attention benefits scale quadratically with sequence length

4. **Extra Credit**: Plot speedup vs sequence length (32, 64, 128, 256)
   - Which fusion benefits most from longer sequences?

## Exercise 3: Memory Scaling with Flash Attention

**Duration**: 30 minutes  
**Objectives**: Understand Flash Attention's memory efficiency improvements

### Setup

```bash
mkdir -p exercises/results/memory_scaling
```

### Part A: Memory Scaling Without Flash Attention

**Task 1: Test Multiple Sequence Lengths (No Flash Attention)**

```bash
for seq_len in 32 64 128 256; do
    python tiny_openfold_v2.py \
        --disable-flash-attention \
        --seq-len $seq_len \
        --batch-size 4 \
        --num-steps 20 \
        --enable-memory-profiling \
        > exercises/results/memory_scaling/no_flash_${seq_len}.log 2>&1
    
    # Extract peak memory
    grep "Peak memory" exercises/results/memory_scaling/no_flash_${seq_len}.log
done
```

Record results:
- seq_len=32: _______ MB
- seq_len=64: _______ MB
- seq_len=128: _______ MB
- seq_len=256: _______ MB (may OOM!)

### Part B: Memory Scaling With Flash Attention

**Task 2: Repeat with Flash Attention Enabled**

```bash
for seq_len in 32 64 128 256; do
    python tiny_openfold_v2.py \
        --enable-flash-attention \
        --seq-len $seq_len \
        --batch-size 4 \
        --num-steps 20 \
        --enable-memory-profiling \
        > exercises/results/memory_scaling/flash_${seq_len}.log 2>&1
    
    grep "Peak memory" exercises/results/memory_scaling/flash_${seq_len}.log
done
```

Record results:
- seq_len=32: _______ MB
- seq_len=64: _______ MB
- seq_len=128: _______ MB
- seq_len=256: _______ MB

### Part C: Analysis

**Task 3: Plot Memory Growth**

```python
# Create a simple plot
import matplotlib.pyplot as plt

seq_lens = [32, 64, 128, 256]
no_flash_mem = [_____, _____, _____, _____]  # Fill from Part A
flash_mem = [_____, _____, _____, _____]      # Fill from Part B

plt.figure(figsize=(10, 6))
plt.plot(seq_lens, no_flash_mem, 'o-', label='Standard Attention', linewidth=2)
plt.plot(seq_lens, flash_mem, 's-', label='Flash Attention', linewidth=2)
plt.xlabel('Sequence Length')
plt.ylabel('Peak Memory (MB)')
plt.title('Memory Scaling: Standard vs Flash Attention')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exercises/results/memory_scaling/memory_comparison.png', dpi=150)
```

### Questions

1. **What is the memory growth rate for standard attention?**
   - Hint: Memory ∝ O(S²) for attention matrices

2. **What is the memory growth rate for Flash Attention?**
   - Expected: Memory ∝ O(S) (linear growth)

3. **At what sequence length does Flash Attention become critical?**
   - When does standard attention OOM?

4. **Calculate memory reduction at seq_len=256:**
   - Reduction = (1 - Flash Memory / Standard Memory) × 100%

## Exercise 4: ROCm Profiling Deep Dive

**Duration**: 60 minutes (ROCm required)  
**Objectives**: Master ROCm profiling tools for hardware-level analysis

### Setup

Verify ROCm environment:
```bash
rocm-smi
rocminfo | grep "Name"
which rocprofv3
```

### Part A: rocprofv3 - Kernel Statistics

**Task 1: Profile Baseline**

```bash
./run_rocprofv3.sh \
    --batch-size 4 \
    --seq-len 64 \
    --disable-all-fusion
```

**Task 2: Profile Fused Version**

```bash
./run_rocprofv3.sh \
    --batch-size 4 \
    --seq-len 64 \
    --enable-all-fusion
```

**Task 3: Compare Kernel Counts**

```bash
# Count unique kernels in baseline
grep "Kernel Name" rocprofv3_results_baseline/stats.csv | wc -l

# Count unique kernels in fused version
grep "Kernel Name" rocprofv3_results_fused/stats.csv | wc -l
```

Record:
- Baseline kernel count: _______
- Fused kernel count: _______
- Reduction: _______ %

### Part B: rocprof-sys - Timeline Analysis

**Task 1: Generate Timeline Trace**

```bash
./run_rocprof_sys.sh \
    --batch-size 4 \
    --seq-len 64 \
    --enable-all-fusion
```

**Task 2: Visualize with Perfetto**

1. Copy the `.proto` file to your local machine
2. Open https://ui.perfetto.dev
3. Load the trace file
4. Zoom into a single training step

**Task 3: Identify Patterns**

In the Perfetto UI, look for:
- Kernel launch patterns
- CPU-GPU synchronization gaps
- Memory transfer operations
- HIP API calls

### Part C: rocprof-compute - Roofline Analysis

**Task 1: Generate Roofline Plot**

```bash
./run_rocprof_compute.sh \
    --roof-only \
    --batch-size 4 \
    --seq-len 64
```

**Task 2: Analyze Key Kernels**

```bash
# Find the dispatch ID of a key kernel (e.g., GEMM)
grep "gemm" rocprof_compute_results/roofline_analysis.csv

# Detailed analysis of specific dispatch
./run_rocprof_compute.sh --mode analyze --dispatch <DISPATCH_ID>
```

**Task 3: Classify Operations**

For the top 5 kernels by time, determine if they are:
- Compute-bound (above roofline)
- Memory-bound (below roofline)
- Well-optimized (near roofline)

### Verification

You should observe:
- ✓ 60-80% reduction in kernel count with fusions
- ✓ Improved kernel utilization in timeline
- ✓ Most GEMM operations are compute-bound
- ✓ Attention operations benefit from Flash Attention

### Questions

1. **What percentage of time is spent in GEMM kernels?**
   - Use rocprofv3 statistics to calculate

2. **Where are the CPU-GPU synchronization points?**
   - Check rocprof-sys timeline

3. **Are MSA attention operations memory-bound or compute-bound?**
   - Use roofline analysis from rocprof-compute

4. **How does kernel occupancy change with fusion?**
   - Look for occupancy metrics in rocprofv3 output

## Exercise 5: torch.compile Optimization

**Duration**: 30 minutes  
**Objectives**: Understand torch.compile's automatic fusion capabilities

### Setup

```bash
mkdir -p exercises/results/torch_compile
```

### Part A: Baseline vs torch.compile

**Task 1: Run Without torch.compile**

```bash
python tiny_openfold_v2.py \
    --enable-all-fusion \
    --batch-size 4 \
    --num-steps 50 \
    > exercises/results/torch_compile/no_compile.log 2>&1
```

**Task 2: Run With torch.compile (default mode)**

```bash
python tiny_openfold_v2.py \
    --enable-all-fusion \
    --enable-torch-compile \
    --torch-compile-mode default \
    --batch-size 4 \
    --num-steps 50 \
    > exercises/results/torch_compile/compile_default.log 2>&1
```

**Task 3: Run With torch.compile (max-autotune)**

```bash
python tiny_openfold_v2.py \
    --enable-all-fusion \
    --enable-torch-compile \
    --torch-compile-mode max-autotune \
    --batch-size 4 \
    --num-steps 50 \
    > exercises/results/torch_compile/compile_max.log 2>&1
```

### Part B: Compare Compilation Overhead

Note the timing:
- Warmup step time: _______ ms (includes compilation)
- Regular step time: _______ ms (after compilation)
- Compilation overhead: _______ ms

### Questions

1. **What is the torch.compile compilation time?**
   - First step vs subsequent steps

2. **What additional speedup does torch.compile provide?**
   - Beyond manual fusion optimizations

3. **When is torch.compile worth the compilation overhead?**
   - Consider training steps vs inference

4. **Which operations benefit most from torch.compile?**
   - Check profiling output

## Exercise 6: Production Optimization Strategy

**Duration**: 45 minutes  
**Objectives**: Develop optimization strategy for production workloads

### Scenario

You need to optimize AlphaFold 2 inference for production:
- Sequence lengths: 128-512 residues
- MSA depth: 64-256 sequences
- Target: <1 second per prediction
- Hardware: AMD MI250X (8 GCDs)

### Part A: Optimization Priority Matrix

Based on your exercises, rank optimizations by impact:

| Optimization | Speedup | Memory Reduction | Implementation Effort | Priority |
|--------------|---------|------------------|-----------------------|----------|
| MSA QKV Fusion | _____ | _____ | Low | _____ |
| Triangle QKV Fusion | _____ | _____ | Low | _____ |
| Flash Attention | _____ | _____ | Medium | _____ |
| Triangle Fusion | _____ | _____ | Low | _____ |
| torch.compile | _____ | _____ | Very Low | _____ |

Priority: 1 (highest) to 5 (lowest)

### Part B: Bottleneck Analysis

**Task 1: Profile Large Sequence**

```bash
python tiny_openfold_v2.py \
    --seq-len 256 \
    --num-seqs 32 \
    --batch-size 2 \
    --enable-all-fusion \
    --enable-pytorch-profiler \
    --num-steps 20
```

**Task 2: Identify Top 3 Bottlenecks**

```bash
# Analyze profiling data
grep "Self CPU total" pytorch_profiles_v2/*.txt | head -20
```

List the top 3 operations by time:
1. _________________________
2. _________________________
3. _________________________

### Part C: Optimization Roadmap

Create a 3-phase optimization plan:

**Phase 1: Quick Wins (Week 1)**
- Optimizations: _________________________
- Expected speedup: _______x
- Risk: Low/Medium/High

**Phase 2: Medium Effort (Week 2-3)**
- Optimizations: _________________________
- Expected speedup: _______x
- Risk: Low/Medium/High

**Phase 3: Advanced (Week 4+)**
- Optimizations: _________________________
- Expected speedup: _______x
- Risk: Low/Medium/High

### Verification

Your roadmap should:
- ✓ Target 2-3x total speedup
- ✓ Start with low-risk optimizations
- ✓ Consider memory constraints
- ✓ Include profiling checkpoints

## Bonus Exercise: Multi-GPU Scaling

**Duration**: 60 minutes  
**Objectives**: Analyze fusion impact on multi-GPU scaling

### Setup

```bash
# Ensure you have access to multiple GPUs
rocm-smi  # Should show 2+ GPUs
```

### Part A: Single vs Multi-GPU

**Task 1: Baseline Single GPU**

```bash
ROCR_VISIBLE_DEVICES=0 python tiny_openfold_v2.py \
    --batch-size 8 \
    --num-steps 50 \
    --disable-all-fusion
```

**Task 2: Baseline Multi-GPU (2 GPUs)**

```bash
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v2.py \
    --batch-size 16 \
    --num-steps 50 \
    --disable-all-fusion
```

**Task 3: Fused Multi-GPU**

```bash
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v2.py \
    --batch-size 16 \
    --num-steps 50 \
    --enable-all-fusion
```

### Part B: Calculate Scaling Efficiency

```
Scaling Efficiency = (Multi-GPU Speedup) / (Number of GPUs) × 100%
```

Fill in:
- Single GPU baseline speed: _______ samples/sec
- Single GPU fused speed: _______ samples/sec
- 2-GPU baseline speed: _______ samples/sec
- 2-GPU fused speed: _______ samples/sec
- Baseline scaling efficiency: _______ %
- Fused scaling efficiency: _______ %

### Questions

1. **Does fusion improve multi-GPU scaling efficiency?**
   - Why or why not?

2. **What are the scaling bottlenecks?**
   - Communication overhead, load imbalance, synchronization?

3. **At what point does adding more GPUs not help?**
   - Test with 4, 8 GPUs if available

## Workshop Completion Checklist

After completing all exercises, you should be able to:

- [ ] Quantify fusion performance improvements
- [ ] Conduct ablation studies independently
- [ ] Analyze memory scaling with Flash Attention
- [ ] Use rocprofv3 for kernel statistics
- [ ] Interpret rocprof-sys timeline traces
- [ ] Perform roofline analysis with rocprof-compute
- [ ] Apply torch.compile effectively
- [ ] Develop optimization roadmaps
- [ ] Understand multi-GPU scaling trade-offs

## Additional Resources

- TinyOpenFold V2 README: `../README.md`
- ROCm Profiling Guide: See parent directory
- Flash Attention Paper: https://arxiv.org/abs/2205.14135
- PyTorch Profiler Documentation: https://pytorch.org/docs/stable/profiler.html

## Getting Help

If you're stuck:
1. Check the comprehensive profiling reports in your results directories
2. Review the fusion statistics in model output
3. Compare with baseline (V1) results
4. Consult the detailed README documentation

---

**Happy Learning!** 🚀

