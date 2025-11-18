# DeepSpeed FLOPS Analysis for TinyOpenFold

This directory includes DeepSpeed FLOPS profiling tools for comprehensive computational efficiency analysis of the Evoformer architecture.

## Overview

The FLOPS profiler helps you understand:
- **Total FLOPS** required per training step
- **FLOPS breakdown** by Evoformer component (MSA attention, triangle multiplication, etc.)
- **Model FLOPS Utilization (MFU)** - how efficiently you're using the GPU
- **Computational intensity** - memory vs compute bound analysis
- **Roofline model data** - for identifying optimization opportunities

## Quick Start

### Basic Usage

```bash
# Run FLOPS profiling with default settings
./run_deepspeed_flops.sh

# Comprehensive analysis with all features
./run_deepspeed_flops.sh --all

# Custom configuration
./run_deepspeed_flops.sh --batch-size 8 --seq-len 128 --num-blocks 8
```

### Installation Requirements

```bash
# Install DeepSpeed (if not already installed)
pip install deepspeed

# Or install from requirements
pip install -r ../requirements.txt
```

## Features

### 1. FLOPS Profiling

Measures the total floating-point operations required for:
- MSA Row/Column Attention
- Triangle Multiplication (Outgoing/Incoming)
- Triangle Attention
- Outer Product Mean
- MSA and Pair Transitions
- Embeddings and Output Head

**Example output:**
```
FLOPS Analysis Summary:
   Total FLOPS per step: 2.45e+11
   FLOPS per parameter: 92.34
   Throughput: 155.9 samples/sec
   Model FLOPS Utilization: 15.3%

Evoformer FLOPS Breakdown:
   msa_attention: 8.32e+10 (34.0%)
   triangle_multiplication: 6.21e+10 (25.4%)
   pair_transition: 4.15e+10 (17.0%)
   ...
```

### 2. Model FLOPS Utilization (MFU)

MFU measures how efficiently your model uses the theoretical peak FLOPS of your GPU:

```
MFU = (Achieved FLOPS) / (Peak GPU FLOPS) × 100%
```

**Interpretation:**
- **< 20% MFU**: Heavy kernel launch overhead, poor kernel fusion
- **20-40% MFU**: Typical for unoptimized baseline models
- **40-60% MFU**: Good optimization with kernel fusion
- **60-80% MFU**: Excellent efficiency (state-of-the-art implementations)
- **> 80% MFU**: Near theoretical maximum (very rare)

### 3. Computational Intensity Analysis

Analyzes the arithmetic intensity (FLOPS per byte of memory transferred):

```bash
./run_deepspeed_flops.sh --intensity
```

**Output:**
```
Computational Intensity Analysis:
   Arithmetic Intensity: 15.2 FLOPS/byte
   Memory Bandwidth Used: 1250 GB/s
   Memory Bandwidth Utilization: 24.0%
   Classification: compute_bound
```

**Interpretation:**
- **< 10 FLOPS/byte**: Memory-bound (limited by memory bandwidth)
- **10-50 FLOPS/byte**: Balanced (both memory and compute matter)
- **> 50 FLOPS/byte**: Compute-bound (limited by FLOPS capacity)

### 4. Roofline Model Data

Generates data for roofline model visualization:

```bash
./run_deepspeed_flops.sh --roofline
```

Creates `roofline_data.json` with:
- Device peak FLOPS and memory bandwidth
- Achieved performance point
- Optimization recommendations

### 5. Evoformer-Specific Analysis

The profiler provides detailed breakdown of Evoformer operations:

```json
{
  "evoformer_breakdown": {
    "msa_attention": 8.32e+10,
    "msa_transition": 3.21e+10,
    "outer_product_mean": 2.15e+10,
    "triangle_multiplication": 6.21e+10,
    "triangle_attention": 4.82e+10,
    "pair_transition": 4.15e+10,
    "embeddings": 1.23e+10,
    "output_head": 0.95e+10
  }
}
```

## Command-Line Options

```bash
./run_deepspeed_flops.sh --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size <n>` | Batch size for profiling | 4 |
| `--seq-len <n>` | Sequence length | 64 |
| `--num-seqs <n>` | Number of MSA sequences | 16 |
| `--msa-dim <n>` | MSA dimension | 64 |
| `--pair-dim <n>` | Pair dimension | 128 |
| `--num-blocks <n>` | Number of Evoformer blocks | 4 |
| `--num-steps <n>` | Number of profiling steps | 10 |
| `--output-dir <path>` | Output directory | `./flops_analysis` |
| `--device <n>` | Specific GPU device ID to use (e.g., 0, 1, 2) | default device |
| `--multi-gpu` | Profile across all available GPUs | false |
| `--devices <ids>` | Comma-separated GPU IDs (e.g., "0,1,2") | none |
| `--detailed` | Enable detailed FLOPS breakdown | false |
| `--roofline` | Generate roofline analysis data | false |
| `--intensity` | Analyze computational intensity | false |
| `--all` | Run all analysis types | false |

## Output Files

The profiler generates several JSON files in the output directory:

### 1. `flops_profile.json`
Complete FLOPS analysis including:
- Model configuration
- Total FLOPS per operation
- Step-by-step results
- Efficiency metrics

### 2. `computational_intensity.json`
Memory bandwidth analysis:
- Arithmetic intensity
- Memory bandwidth utilization
- Memory breakdown by component

### 3. `roofline_data.json`
Roofline model data:
- Device specifications
- Performance point
- Optimization targets

## Example Workflows

### Workflow 1: Basic Performance Baseline

```bash
# Profile your baseline configuration
./run_deepspeed_flops.sh --batch-size 4 --seq-len 64

# Check the MFU
cat flops_analysis/flops_profile.json | jq '.efficiency_metrics.mfu_percent'
```

### Workflow 2: Identify Bottlenecks

```bash
# Run comprehensive analysis
./run_deepspeed_flops.sh --all --output-dir analysis_baseline

# Find the most expensive operations
cat analysis_baseline/flops_profile.json | \
  jq '.flops_analysis.evoformer_breakdown | to_entries | sort_by(.value) | reverse | .[0:5]'

# Check optimization recommendations
cat analysis_baseline/roofline_data.json | jq '.optimization_targets'
```

### Workflow 3: Scaling Study

```bash
# Profile different model sizes
for blocks in 4 8 16; do
  ./run_deepspeed_flops.sh \
    --num-blocks $blocks \
    --output-dir analysis_blocks_$blocks \
    --all
done

# Compare MFU across configurations
for dir in analysis_blocks_*; do
  echo "$dir: $(cat $dir/flops_profile.json | jq -r '.efficiency_metrics.mfu_percent')%"
done
```

### Workflow 4: Memory vs Compute Analysis

```bash
# Analyze computational intensity
./run_deepspeed_flops.sh --intensity --output-dir intensity_analysis

# Check if memory-bound or compute-bound
cat intensity_analysis/computational_intensity.json | \
  jq '.memory_bound_vs_compute_bound'

# View memory breakdown
cat intensity_analysis/computational_intensity.json | \
  jq '.memory_breakdown'
```

### Workflow 5: Multi-GPU Profiling

```bash
# Profile single GPU for baseline
./run_deepspeed_flops.sh --device 0 --output-dir gpu0_results

# Profile all available GPUs (8 on MI250X/MI300X nodes)
./run_deepspeed_flops.sh --multi-gpu --output-dir multi_gpu_results

# Profile specific GPUs
./run_deepspeed_flops.sh --devices "0,1,2,3" --output-dir quad_gpu_results

# Compare single vs multi-GPU efficiency
echo "Single GPU MFU:"
cat gpu0_results/flops_profile.json | jq '.efficiency_metrics.mfu_percent'

echo "Multi-GPU Average MFU:"
cat multi_gpu_results/flops_profile_multi_gpu.json | jq '.aggregate_metrics.avg_mfu_percent'

echo "Multi-GPU Efficiency:"
cat multi_gpu_results/flops_profile_multi_gpu.json | jq '.aggregate_metrics.multi_gpu_efficiency_percent'

echo "Speedup:"
cat multi_gpu_results/flops_profile_multi_gpu.json | jq '.comparison.speedup'
```

**Multi-GPU Output:**
```json
{
  "aggregate_metrics": {
    "num_gpus": 8,
    "avg_mfu_percent": 15.8,
    "total_system_tflops": 196.8,
    "total_throughput": 84.6,
    "multi_gpu_efficiency_percent": 95.2
  },
  "comparison": {
    "single_gpu_throughput": 10.5,
    "multi_gpu_throughput": 84.6,
    "speedup": 7.62
  }
}
```

**Key Multi-GPU Metrics:**
- **Multi-GPU Efficiency**: Actual speedup / Ideal speedup (target: >90%)
- **Total System TFLOPS**: Sum of achieved TFLOPS across all GPUs
- **MFU Std Dev**: Performance variance across GPUs (lower is better)
- **Speedup**: Multi-GPU throughput / Single GPU throughput

## Interpreting Results

### Understanding MFU

Compare your MFU with these benchmarks:

| Model Type | Typical MFU | Notes |
|------------|-------------|-------|
| Baseline (unoptimized) | 10-25% | Heavy Python overhead, no kernel fusion |
| Fused Kernels | 30-45% | QKV fusion, attention optimization |
| Flash Attention | 45-65% | Memory-efficient attention |
| State-of-the-art | 60-80% | Triton kernels, custom CUDA |

### Optimization Priority

Based on FLOPS breakdown, prioritize optimizations:

1. **If Triangle Multiplication > 25%**:
   - Implement fused triangle multiplication kernels
   - Use memory-efficient implementations
   - Expected improvement: 30-40%

2. **If MSA Attention > 30%**:
   - Adapt Flash Attention for MSA
   - Fuse attention operations
   - Expected improvement: 2-3x speedup

3. **If Low MFU (< 20%)**:
   - Focus on kernel fusion
   - Reduce Python overhead
   - Use torch.compile() or custom kernels

4. **If Memory-bound (AI < 10)**:
   - Use mixed precision (FP16/BF16)
   - Enable gradient checkpointing
   - Optimize memory access patterns

## Integration with PyTorch Profiler

Compare FLOPS analysis with PyTorch profiler results:

```bash
# Run both profilers
./run_deepspeed_flops.sh --all --output-dir flops_results
python tiny_openfold_v1.py --enable-pytorch-profiler --profile-dir pytorch_results

# Compare results
echo "DeepSpeed MFU:"
cat flops_results/flops_profile.json | jq '.efficiency_metrics.mfu_percent'

echo "PyTorch Throughput:"
cat pytorch_results/performance_summary.json | jq '.performance_summary.avg_training_speed'
```

## Troubleshooting

### DeepSpeed Not Available

If DeepSpeed is not installed:
```bash
pip install deepspeed
```

The script provides FLOPS estimates without DeepSpeed, but detailed profiling requires it.

### GPU Not Detected

The profiler will automatically detect:
- AMD GPUs (via ROCm)
- NVIDIA GPUs (via CUDA)

Peak FLOPS values are based on known GPU specifications. If your GPU is not recognized, it will use conservative defaults.

### Memory Errors

If you encounter OOM errors:
```bash
# Reduce batch size
./run_deepspeed_flops.sh --batch-size 2

# Or reduce sequence length
./run_deepspeed_flops.sh --seq-len 32
```

## GPU-Specific Notes

### AMD MI300X
- Peak FP32: 163.4 TFLOPS (matrix operations)
- Peak Memory Bandwidth: 5300 GB/s
- Target MFU: 40-60% for baseline models

### NVIDIA H100
- Peak FP32: 67 TFLOPS
- Peak Memory Bandwidth: 3350 GB/s
- Target MFU: 45-65% for baseline models

### NVIDIA A100
- Peak FP32: 19.5 TFLOPS
- Peak Memory Bandwidth: 2039 GB/s
- Target MFU: 35-55% for baseline models

## References

- [DeepSpeed FLOPS Profiler Documentation](https://www.deepspeed.ai/tutorials/flops-profiler/)
- [AlphaFold 2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- [Model FLOPS Utilization](https://arxiv.org/abs/2204.02311)

## See Also

- `README.md` - Main documentation for TinyOpenFold V1
- `OPTIMIZATION_NOTES.md` - Detailed optimization strategies
- `SCALING_QUICKSTART.md` - Multi-GPU scaling guide
- `run.sh` - Multi-GPU scaling study script

