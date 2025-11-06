#!/bin/bash
################################################################################
# Quick TinyOpenFold Scaling Test
# 
# Simplified script for quick scaling tests with 1, 2, 4, and 8 GPUs
# Uses 4 samples per GPU and 50 training steps
################################################################################

set -e

# Configuration
BATCH_PER_GPU=4
STEPS=50
OUTPUT_DIR="scaling_study_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "Quick TinyOpenFold Scaling Test"
echo "================================================================================"
echo "Batch size per GPU: $BATCH_PER_GPU"
echo "Training steps: $STEPS"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Test 1 GPU
echo "Testing 1 GPU (batch size 4)..."
ROCR_VISIBLE_DEVICES=0 python tiny_openfold_v1.py --batch-size 4 --num-steps $STEPS 2>&1 | tee "$OUTPUT_DIR/gpu1_batch4.log"
echo ""

# Test 2 GPUs
echo "Testing 2 GPUs (batch size 8)..."
ROCR_VISIBLE_DEVICES=0,1 python tiny_openfold_v1.py --batch-size 8 --num-steps $STEPS 2>&1 | tee "$OUTPUT_DIR/gpu2_batch8.log"
echo ""

# Test 4 GPUs
echo "Testing 4 GPUs (batch size 16)..."
ROCR_VISIBLE_DEVICES=0,1,2,3 python tiny_openfold_v1.py --batch-size 16 --num-steps $STEPS 2>&1 | tee "$OUTPUT_DIR/gpu4_batch16.log"
echo ""

# Test 8 GPUs
echo "Testing 8 GPUs (batch size 32)..."
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tiny_openfold_v1.py --batch-size 32 --num-steps $STEPS 2>&1 | tee "$OUTPUT_DIR/gpu8_batch32.log"
echo ""

# Generate summary
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo ""
grep 'Average training speed:' "$OUTPUT_DIR"/*.log

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
echo ""
echo "Speedup calculation:"
echo "-------------------"

# Extract throughputs and calculate speedup
throughput_1gpu=$(grep 'Average training speed:' "$OUTPUT_DIR/gpu1_batch4.log" | awk '{print $4}')
throughput_2gpu=$(grep 'Average training speed:' "$OUTPUT_DIR/gpu2_batch8.log" | awk '{print $4}')
throughput_4gpu=$(grep 'Average training speed:' "$OUTPUT_DIR/gpu4_batch16.log" | awk '{print $4}')
throughput_8gpu=$(grep 'Average training speed:' "$OUTPUT_DIR/gpu8_batch32.log" | awk '{print $4}')

if command -v bc &> /dev/null; then
    speedup_2=$(echo "scale=2; $throughput_2gpu / $throughput_1gpu" | bc)
    speedup_4=$(echo "scale=2; $throughput_4gpu / $throughput_1gpu" | bc)
    speedup_8=$(echo "scale=2; $throughput_8gpu / $throughput_1gpu" | bc)
    
    efficiency_2=$(echo "scale=1; 100 * $speedup_2 / 2" | bc)
    efficiency_4=$(echo "scale=1; 100 * $speedup_4 / 4" | bc)
    efficiency_8=$(echo "scale=1; 100 * $speedup_8 / 8" | bc)
    
    printf "%-8s %-20s %-12s %-12s\n" "GPUs" "Throughput (s/s)" "Speedup" "Efficiency"
    printf "%-8s %-20s %-12s %-12s\n" "----" "-------------------" "---------" "----------"
    printf "%-8s %-20s %-12s %-12s\n" "1" "$throughput_1gpu" "1.00x" "100.0%"
    printf "%-8s %-20s %-12s %-12s\n" "2" "$throughput_2gpu" "${speedup_2}x" "${efficiency_2}%"
    printf "%-8s %-20s %-12s %-12s\n" "4" "$throughput_4gpu" "${speedup_4}x" "${efficiency_4}%"
    printf "%-8s %-20s %-12s %-12s\n" "8" "$throughput_8gpu" "${speedup_8}x" "${efficiency_8}%"
else
    echo "Install 'bc' for speedup calculations"
    echo "1 GPU: $throughput_1gpu samples/sec"
    echo "2 GPUs: $throughput_2gpu samples/sec"
    echo "4 GPUs: $throughput_4gpu samples/sec"
    echo "8 GPUs: $throughput_8gpu samples/sec"
fi

echo ""
echo "Done!"

