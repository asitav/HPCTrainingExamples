# TinyOpenFold Multi-GPU Scaling Quick Start

## Prerequisites

```bash
cd /path/to/HPCTrainingExamples/MLExamples/TinyOpenFold/version1_pytorch_baseline
# Activate your Python environment with PyTorch installed
```

## Quick Commands

### Single Run Examples

```bash
# 1 GPU (4 samples)
ROCR_VISIBLE_DEVICES=0 python tiny_openfold_v1.py --batch-size 4

# 2 GPUs (8 samples = 4 per GPU)
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v1.py --batch-size 8

# 4 GPUs (16 samples = 4 per GPU)
ROCR_VISIBLE_DEVICES=0,1,2,3 python tiny_openfold_v1.py --batch-size 16

# 8 GPUs (32 samples = 4 per GPU)
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tiny_openfold_v1.py --batch-size 32
```

**For NVIDIA GPUs:** Replace `ROCR_VISIBLE_DEVICES` with `CUDA_VISIBLE_DEVICES`

### Automated Scaling Studies

#### Option 1: Quick Test (Simple)

```bash
chmod +x quick_scaling_test.sh
./quick_scaling_test.sh
```

**What it does:**
- Tests 1, 2, 4, and 8 GPUs
- Uses 4 samples per GPU (batch sizes: 4, 8, 16, 32)
- Runs 50 training steps per configuration
- Saves logs to timestamped directory
- Displays summary with speedup and efficiency

**Output Example:**
```
GPUs     Throughput (s/s)     Speedup      Efficiency
----     -------------------  ---------    ----------
1        166.9                1.00x        100.0%
2        202.7                1.21x        60.5%
4        245.3                1.47x        36.8%
8        249.1                1.49x        18.6%
```

#### Option 2: Comprehensive Study (Advanced)

```bash
chmod +x run.sh

# Basic usage
./run.sh

# Custom configurations
./run.sh --gpus "1 2 4 8" --batch-per-gpu 8 --steps 100
./run.sh --amp --profile
./run.sh --runs 3 --output-dir my_study
./run.sh --help  # Show all options
```

**What it does:**
- Flexible GPU configuration
- Multiple runs for statistical analysis
- Optional mixed precision and profiling
- Generates CSV and text summaries
- Detailed per-run logs

**Output Files:**
```
scaling_study_TIMESTAMP/
├── config.txt                # Configuration used
├── summary.txt               # Results with statistics
├── summary.csv               # Machine-readable data
└── gpu*_batch*_run*.log      # Individual run logs
```

## Understanding Results

### Key Metrics

- **Throughput**: Samples processed per second
- **Speedup**: Performance gain relative to 1 GPU
  - Formula: `Throughput(N GPUs) / Throughput(1 GPU)`
- **Efficiency**: How well GPUs are utilized
  - Formula: `(Speedup / N GPUs) × 100%`
  - 100% = perfect linear scaling
  - 70-90% = good scaling
  - <50% = significant overhead

### Expected Behavior for TinyOpenFold

| GPUs | Expected Speedup | Expected Efficiency | Notes |
|------|-----------------|---------------------|-------|
| 1    | 1.00x          | 100%                | Baseline |
| 2    | 1.15-1.30x     | 58-65%              | Good for small model |
| 4    | 1.40-1.60x     | 35-40%              | Diminishing returns |
| 8    | 1.45-1.70x     | 18-21%              | High overhead |

**Why Sub-linear Scaling?**
- Small model size (2.6M parameters)
- DataParallel communication overhead
- GPU synchronization costs
- Memory bandwidth limitations

### Optimization Tips

1. **Batch Size**: Use 4-8 samples per GPU for best efficiency
2. **Mixed Precision**: Add `--use-amp` to increase throughput (may be slower on MI300X)
3. **Model Size**: Larger models (more blocks, bigger dimensions) scale better
4. **Hardware**: Faster interconnects (NVLink, Infinity Fabric) improve scaling

## Troubleshooting

### Issue: "grad can be implicitly created only for scalar outputs"

**Solution**: Already fixed in the code. If you see this, update to the latest version.

### Issue: Not using all GPUs

```bash
# Check visible GPUs
echo $ROCR_VISIBLE_DEVICES  # or $CUDA_VISIBLE_DEVICES

# Force single GPU mode
python tiny_openfold_v1.py --no-data-parallel --device 0
```

### Issue: Out of memory

```bash
# Reduce batch size per GPU
./run.sh --batch-per-gpu 4  # Instead of 8

# Enable mixed precision
./run.sh --amp
```

### Issue: Warning about "gather along dimension 0"

This is a benign PyTorch warning related to DataParallel gathering scalar losses. It's expected and doesn't affect training.

## Quick Comparison Commands

After running experiments, compare results:

```bash
# From individual runs
grep 'Average training speed:' out_gpus*.log

# From scaling study output
cat scaling_study_*/summary.txt

# Extract specific values
grep 'Average training speed:' scaling_study_*/gpu*.log | \
  awk '{print $1, $4}' | sort
```

## Example Workflow

```bash
# 1. Quick validation run
python tiny_openfold_v1.py --batch-size 4 --num-steps 10 --validate-setup

# 2. Single GPU baseline
ROCR_VISIBLE_DEVICES=0 python tiny_openfold_v1.py --batch-size 4 --num-steps 50

# 3. Test multi-GPU
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v1.py --batch-size 8 --num-steps 50

# 4. Run full scaling study
./quick_scaling_test.sh

# 5. Analyze results
cat scaling_study_*/summary.txt
```

## Advanced: Custom Scaling Study

For publication-quality data:

```bash
# Multiple runs with mixed precision
./run.sh \
  --gpus "1 2 4 8" \
  --batch-per-gpu 4 \
  --steps 200 \
  --runs 5 \
  --amp \
  --output-dir scaling_study_final

# Analyze with standard deviation
python -c "
import pandas as pd
df = pd.read_csv('scaling_study_final/summary.csv')
print(df.groupby('num_gpus')['throughput_samples_per_sec'].agg(['mean', 'std']))
"
```

## Performance Baselines (MI300X)

Reference performance on AMD Instinct MI300X (with 8 samples per GPU):

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| 1 GPU, batch 8  | ~167 s/s   | Baseline (8/GPU) |
| 2 GPUs, batch 16 | ~203 s/s   | 1.21x speedup (8/GPU) |
| 4 GPUs, batch 32 | ~245 s/s   | 1.47x speedup (8/GPU) |
| 8 GPUs, batch 64 | ~249 s/s   | 1.49x speedup (8/GPU) |

*Your results may vary based on hardware, drivers, and system load.*  
*Note: The quick_scaling_test.sh script now defaults to 4 samples per GPU for faster iteration.*

## Additional Resources

- Full documentation: [`README.md`](README.md)
- Architecture details: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- OpenFold training analysis: [`../../../../llm_notes/tiny_openfold_throughput.md`](../../../../llm_notes/tiny_openfold_throughput.md)

