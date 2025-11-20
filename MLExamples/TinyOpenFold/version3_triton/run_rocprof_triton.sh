#!/bin/bash
#
# ROCProfiler Integration for TinyOpenFold V3 Triton Kernels
#
# This script uses ROCProfiler to collect hardware-level metrics
# for Triton kernels running on AMD GPUs.
#
# Usage:
#   chmod +x run_rocprof_triton.sh
#   ./run_rocprof_triton.sh

echo "========================================="
echo "ROCProfiler for TinyOpenFold V3"
echo "Triton Kernel Hardware Profiling"
echo "========================================="
echo ""

# Configuration
OUTPUT_DIR="rocprof_results_v3"
PYTHON_SCRIPT="tiny_openfold_v3.py"
BATCH_SIZE=4
NUM_STEPS=20

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Configuration:"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Python script: ${PYTHON_SCRIPT}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Training steps: ${NUM_STEPS}"
echo ""

# Check if rocprof is available
if ! command -v rocprof &> /dev/null; then
    echo "ERROR: rocprof not found in PATH"
    echo "Please ensure ROCm is properly installed and configured"
    exit 1
fi

echo "ROCm version:"
rocminfo | grep "Name:" | head -n 1
echo ""

# =========================================================================
# 1. Basic Kernel Timing
# =========================================================================
echo "========================================="
echo "1. Basic Kernel Timing"
echo "========================================="

rocprof \
    --stats \
    --timestamp on \
    --output-file ${OUTPUT_DIR}/kernel_stats.csv \
    python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/kernel_timing.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Kernel timing complete"
    echo "  Results: ${OUTPUT_DIR}/kernel_stats.csv"
else
    echo "✗ Kernel timing failed"
fi
echo ""

# =========================================================================
# 2. HIP API Trace
# =========================================================================
echo "========================================="
echo "2. HIP API Trace"
echo "========================================="

rocprof \
    --hip-trace \
    --output-file ${OUTPUT_DIR}/hip_trace.csv \
    python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/hip_trace.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ HIP trace complete"
    echo "  Results: ${OUTPUT_DIR}/hip_trace.csv"
else
    echo "✗ HIP trace failed"
fi
echo ""

# =========================================================================
# 3. Memory Copy Analysis
# =========================================================================
echo "========================================="
echo "3. Memory Copy Analysis"
echo "========================================="

rocprof \
    --hsa-trace \
    --output-file ${OUTPUT_DIR}/memory_trace.csv \
    python3 ${PYTHON_SCRIPT} \
        --batch-size ${BATCH_SIZE} \
        --num-steps ${NUM_STEPS} \
    > ${OUTPUT_DIR}/memory_trace.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Memory trace complete"
    echo "  Results: ${OUTPUT_DIR}/memory_trace.csv"
else
    echo "✗ Memory trace failed"
fi
echo ""

# =========================================================================
# 4. Generate Summary Report
# =========================================================================
echo "========================================="
echo "4. Generating Summary Report"
echo "========================================="

cat > ${OUTPUT_DIR}/triton_analysis_summary.md << 'EOF'
# TinyOpenFold V3 Triton Kernel Profiling Summary

## Profiling Session

**Date**: $(date)
**Model Version**: V3 (Triton Custom Kernels)
**Hardware**: AMD MI300X

## Files Generated

1. `kernel_stats.csv` - Kernel execution statistics
2. `hip_trace.csv` - HIP API trace
3. `memory_trace.csv` - Memory transfer trace
4. `*.log` - Execution logs

## Analysis Steps

### 1. Kernel Statistics Analysis

```bash
# View top kernels by execution time
cat kernel_stats.csv | sort -t',' -k2 -nr | head -20
```

### 2. HIP API Overhead

```bash
# Analyze HIP API calls
grep -i "hipMalloc\|hipMemcpy\|hipLaunchKernel" hip_trace.csv
```

### 3. Memory Bandwidth Utilization

Look for:
- Memory copy patterns
- Kernel memory access patterns
- Cache utilization

### 4. Triton Kernel Identification

Triton kernels will appear with names containing:
- `layernorm_kernel`
- `flash_attention_kernel`
- `triton_` prefix

## Key Metrics to Review

1. **Kernel Execution Time**: Total time spent in each kernel
2. **Launch Overhead**: Time between kernel launches
3. **Memory Bandwidth**: Achieved vs theoretical bandwidth
4. **Occupancy**: SM utilization percentage

## Comparison with Baseline

Compare these metrics with Version 1 and Version 2 results to validate
the performance improvements from Triton kernel optimizations.

EOF

echo "✓ Summary report generated"
echo "  Report: ${OUTPUT_DIR}/triton_analysis_summary.md"
echo ""

# =========================================================================
# 5. Display Summary
# =========================================================================
echo "========================================="
echo "Profiling Complete!"
echo "========================================="
echo ""
echo "Results saved in: ${OUTPUT_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Review ${OUTPUT_DIR}/triton_analysis_summary.md"
echo "  2. Analyze kernel statistics in ${OUTPUT_DIR}/kernel_stats.csv"
echo "  3. Compare with V1/V2 baseline results"
echo ""
echo "To view kernel statistics:"
echo "  cat ${OUTPUT_DIR}/kernel_stats.csv | column -t -s, | less -S"
echo ""

