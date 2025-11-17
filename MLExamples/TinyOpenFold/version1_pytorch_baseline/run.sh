#!/bin/bash
################################################################################
# TinyOpenFold Scaling Study Script
# 
# This script runs TinyOpenFold training with different GPU counts to measure
# scaling efficiency and throughput.
#
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   --gpus <list>        GPU counts to test (default: "1 2 4 8")
#   --batch-per-gpu <n>  Batch size per GPU (default: 8)
#   --steps <n>          Training steps (default: 50)
#   --runs <n>           Number of runs per configuration (default: 1)
#   --amp                Enable mixed precision training
#   --profile            Enable PyTorch profiler
#   --output-dir <dir>   Output directory for logs (default: scaling_study_TIMESTAMP)
#   --help               Show this help message
#
# Example:
#   ./run.sh --gpus "1 2 4" --batch-per-gpu 8 --steps 100
#   ./run.sh --amp --profile --output-dir my_scaling_study
#
################################################################################

set -e  # Exit on error

# Default configuration
GPU_COUNTS="1 2 4 8"
BATCH_PER_GPU=8
STEPS=50
RUNS=1
USE_AMP=false
USE_PROFILE=false
OUTPUT_DIR=""
PYTHON_SCRIPT="tiny_openfold_v1.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPU_COUNTS="$2"
            shift 2
            ;;
        --batch-per-gpu)
            BATCH_PER_GPU="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --amp)
            USE_AMP=true
            shift
            ;;
        --profile)
            USE_PROFILE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory with timestamp if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="scaling_study_${TIMESTAMP}"
fi

mkdir -p "$OUTPUT_DIR"

# Detect GPU environment (ROCm vs CUDA)
if command -v rocm-smi &> /dev/null; then
    GPU_ENV="ROCM"
    GPU_VAR="ROCR_VISIBLE_DEVICES"
    echo -e "${CYAN}Detected ROCm environment${NC}"
elif command -v nvidia-smi &> /dev/null; then
    GPU_ENV="CUDA"
    GPU_VAR="CUDA_VISIBLE_DEVICES"
    echo -e "${CYAN}Detected CUDA environment${NC}"
else
    echo -e "${YELLOW}Warning: Could not detect GPU environment, assuming CUDA${NC}"
    GPU_ENV="CUDA"
    GPU_VAR="CUDA_VISIBLE_DEVICES"
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: $PYTHON_SCRIPT not found${NC}"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo -e "${BLUE}TinyOpenFold Scaling Study${NC}"
echo "================================================================================"
echo "Configuration:"
echo "  GPU counts to test: $GPU_COUNTS"
echo "  Batch size per GPU: $BATCH_PER_GPU"
echo "  Training steps: $STEPS"
echo "  Runs per config: $RUNS"
echo "  Mixed precision: $USE_AMP"
echo "  Profiling: $USE_PROFILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPU environment: $GPU_ENV ($GPU_VAR)"
echo "================================================================================"
echo ""

# Save configuration to file
CONFIG_FILE="$OUTPUT_DIR/config.txt"
cat > "$CONFIG_FILE" << EOF
TinyOpenFold Scaling Study Configuration
=========================================
Date: $(date)
Host: $(hostname)
GPU Environment: $GPU_ENV

Test Configuration:
- GPU counts: $GPU_COUNTS
- Batch size per GPU: $BATCH_PER_GPU
- Training steps: $STEPS
- Runs per configuration: $RUNS
- Mixed precision: $USE_AMP
- Profiling: $USE_PROFILE

Python Script: $PYTHON_SCRIPT
Output Directory: $OUTPUT_DIR
EOF

# Array to store results
declare -a RESULTS

# Function to parse throughput from log
parse_throughput() {
    local log_file=$1
    grep "Average training speed:" "$log_file" | awk '{print $4}'
}

# Function to run experiment
run_experiment() {
    local num_gpus=$1
    local run_num=$2
    local batch_size=$((BATCH_PER_GPU * num_gpus))
    
    # Generate GPU device list (0,1,2,...)
    local gpu_list=$(seq -s',' 0 $((num_gpus - 1)))
    
    # Create log filename
    local log_file="$OUTPUT_DIR/gpu${num_gpus}_batch${batch_size}_run${run_num}.log"
    
    # Build command
    local cmd="python $PYTHON_SCRIPT --batch-size $batch_size --num-steps $STEPS"
    
    if [ "$USE_AMP" = true ]; then
        cmd="$cmd --use-amp"
    fi
    
    if [ "$USE_PROFILE" = true ]; then
        cmd="$cmd --enable-pytorch-profiler"
    fi
    
    # Set environment and run
    echo -e "${GREEN}Running: $num_gpus GPU(s), batch size $batch_size, run $run_num/$RUNS${NC}"
    echo "  Command: $GPU_VAR=$gpu_list $cmd"
    echo "  Log file: $log_file"
    
    # Run the experiment
    export $GPU_VAR=$gpu_list
    $cmd 2>&1 | tee "$log_file"
    
    # Parse throughput
    local throughput=$(parse_throughput "$log_file")
    
    if [ -n "$throughput" ]; then
        echo -e "${CYAN}  Result: $throughput samples/sec${NC}"
        RESULTS+=("$num_gpus,$batch_size,$run_num,$throughput")
    else
        echo -e "${RED}  Warning: Could not parse throughput from log${NC}"
        RESULTS+=("$num_gpus,$batch_size,$run_num,ERROR")
    fi
    
    echo ""
}

# Main experiment loop
echo "Starting experiments..."
echo ""

for num_gpus in $GPU_COUNTS; do
    echo "================================================================================"
    echo -e "${BLUE}Testing with $num_gpus GPU(s)${NC}"
    echo "================================================================================"
    
    for ((run=1; run<=RUNS; run++)); do
        run_experiment $num_gpus $run
        
        # Brief pause between runs
        if [ $run -lt $RUNS ]; then
            sleep 2
        fi
    done
    
    echo ""
done

# Generate summary
echo "================================================================================"
echo -e "${BLUE}Generating Summary${NC}"
echo "================================================================================"

SUMMARY_FILE="$OUTPUT_DIR/summary.csv"
SUMMARY_TXT="$OUTPUT_DIR/summary.txt"

# Create CSV header
echo "num_gpus,batch_size,run,throughput_samples_per_sec" > "$SUMMARY_FILE"

# Write results to CSV
for result in "${RESULTS[@]}"; do
    echo "$result" >> "$SUMMARY_FILE"
done

# Create text summary with statistics
{
    echo "TinyOpenFold Scaling Study Summary"
    echo "=================================="
    echo ""
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo ""
    echo "Configuration:"
    echo "  Batch size per GPU: $BATCH_PER_GPU"
    echo "  Training steps: $STEPS"
    echo "  Runs per config: $RUNS"
    echo "  Mixed precision: $USE_AMP"
    echo ""
    echo "Results:"
    echo "--------"
    printf "%-8s %-12s %-15s %-15s %-15s\n" "GPUs" "Batch Size" "Avg Throughput" "Speedup" "Efficiency"
    printf "%-8s %-12s %-15s %-15s %-15s\n" "----" "----------" "--------------" "-------" "----------"
    
    # Calculate averages and speedup
    baseline_throughput=""
    
    for num_gpus in $GPU_COUNTS; do
        batch_size=$((BATCH_PER_GPU * num_gpus))
        
        # Calculate average throughput for this GPU count
        total=0
        count=0
        for result in "${RESULTS[@]}"; do
            IFS=',' read -r gpus bs run throughput <<< "$result"
            if [ "$gpus" = "$num_gpus" ] && [ "$throughput" != "ERROR" ]; then
                total=$(echo "$total + $throughput" | bc -l)
                count=$((count + 1))
            fi
        done
        
        if [ $count -gt 0 ]; then
            avg_throughput=$(echo "scale=1; $total / $count" | bc -l)
            
            # Calculate speedup and efficiency
            if [ -z "$baseline_throughput" ]; then
                baseline_throughput=$avg_throughput
                speedup="1.0x"
                efficiency="100.0%"
            else
                speedup=$(echo "scale=2; $avg_throughput / $baseline_throughput" | bc -l)
                efficiency=$(echo "scale=1; 100 * $speedup / $num_gpus" | bc -l)
                speedup="${speedup}x"
                efficiency="${efficiency}%"
            fi
            
            printf "%-8s %-12s %-15s %-15s %-15s\n" \
                "$num_gpus" "$batch_size" "$avg_throughput" "$speedup" "$efficiency"
        else
            printf "%-8s %-12s %-15s %-15s %-15s\n" \
                "$num_gpus" "$batch_size" "ERROR" "N/A" "N/A"
        fi
    done
    
    echo ""
    echo "Notes:"
    echo "  - Throughput is in samples/second"
    echo "  - Speedup is relative to single GPU baseline"
    echo "  - Efficiency = (Speedup / Number of GPUs) * 100%"
    echo "  - Ideal linear scaling would show 100% efficiency"
    echo ""
    echo "Output files:"
    echo "  - Detailed logs: $OUTPUT_DIR/gpu*_batch*_run*.log"
    echo "  - CSV data: $SUMMARY_FILE"
    echo "  - This summary: $SUMMARY_TXT"
    
} | tee "$SUMMARY_TXT"

echo ""
echo "================================================================================"
echo -e "${GREEN}Scaling study complete!${NC}"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Display summary file location
echo -e "${CYAN}Quick summary:${NC}"
cat "$SUMMARY_TXT" | grep -A 20 "Results:"

echo ""
echo -e "${YELLOW}To analyze results:${NC}"
echo "  cat $SUMMARY_TXT"
echo "  cat $SUMMARY_FILE"
echo "  grep 'Average training speed:' $OUTPUT_DIR/*.log"

