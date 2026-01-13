#!/bin/bash

# rocprof-sys-python Profiling Integration for Tiny OpenFold V2
# This script provides Python call stack profiling with source-level instrumentation
# Based on: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_rocprof() { echo -e "${PURPLE}[ROCPROF-SYS]${NC} $1"; }

# Default configuration (smaller defaults for profiling to reduce output size)
BATCH_SIZE=2
SEQ_LEN=16
NUM_BLOCKS=4
NUM_SEQS=16
NUM_STEPS=30
OUTPUT_DIR="./rocprof_sys_results_$(date +%Y%m%d_%H%M%S)"
ENABLE_ALL_FUSION=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
        --num-seqs) NUM_SEQS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --disable-all-fusion) ENABLE_ALL_FUSION=false; shift ;;
        --help|-h)
            echo "rocprof-sys-python Profiling for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script uses rocprof-sys-python for Python call stack profiling"
            echo "with source-level instrumentation. See:"
            echo "https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 2, smaller for profiling)"
            echo "  --seq-len N             Sequence length (default: 16, smaller for profiling)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Training steps (default: 30)"
            echo "  --output-dir DIR        Output directory"
            echo "  --disable-all-fusion    Disable all fusions"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Profile with defaults (batch=2, seq=16)"
            echo "  $0 --batch-size 4 --seq-len 64        # Larger workload"
            echo "  $0 --disable-all-fusion              # Baseline comparison"
            echo ""
            echo "Output:"
            echo "  - Python call stack profiling with function call counts"
            echo "  - ROCPD trace files (.rocpd or .rocpd.json) for AI/ML workloads"
            echo "  - Detailed profiling log in rocprof_sys.log"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check for rocprof-sys-python or python3 -m rocprofsys
ROCPROF_SYS_PYTHON_CMD=""
if command -v rocprof-sys-python &> /dev/null; then
    ROCPROF_SYS_PYTHON_CMD="rocprof-sys-python"
elif python3 -m rocprofsys --help &> /dev/null; then
    ROCPROF_SYS_PYTHON_CMD="python3 -m rocprofsys"
else
    log_info "rocprof-sys-python not found. Please ensure ROCm Systems Profiler Python bindings are installed."
    log_info "The Python package should be in: /opt/rocprofiler-systems/lib/python*/site-packages/rocprofsys"
    log_info "Or ensure PYTHONPATH includes the rocprofsys package location."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

log_info "======================================================================"
log_info "Tiny OpenFold V2 - rocprof-sys-python Call Stack Profiling"
log_info "======================================================================"
echo ""
log_info "Configuration:"
log_info "  Batch size: $BATCH_SIZE"
log_info "  Sequence length: $SEQ_LEN"
log_info "  Evoformer blocks: $NUM_BLOCKS"
log_info "  MSA sequences: $NUM_SEQS"
log_info "  Training steps: $NUM_STEPS"
log_info "  All fusions: $ENABLE_ALL_FUSION"
log_info "  Output directory: $OUTPUT_DIR"
echo ""

# Build Python command
PYTHON_ARGS="--batch-size $BATCH_SIZE --seq-len $SEQ_LEN --num-blocks $NUM_BLOCKS --num-seqs $NUM_SEQS --num-steps $NUM_STEPS"
[ "$ENABLE_ALL_FUSION" = false ] && PYTHON_ARGS="$PYTHON_ARGS --disable-all-fusion"

# Run profiling with Python call stack support
log_step "Starting rocprof-sys-python profiling..."
log_rocprof "This will generate Python call stack profiling output"
log_rocprof "Using command: $ROCPROF_SYS_PYTHON_CMD"
echo ""

# Set environment variables for profiling
# ROCPD output is recommended for AI/ML workloads (better child thread support)
export ROCPROFSYS_USE_ROCPD=ON
export ROCPROFSYS_PROFILE=ON

# Optional: Configure profiling components (e.g., trip_count, wall_clock, etc.)
# export ROCPROFSYS_TIMEMORY_COMPONENTS="trip_count,wall_clock"

cd "$OUTPUT_DIR"
# rocprof-sys-python syntax: rocprof-sys-python <ARGS> -- <SCRIPT> <SCRIPT_ARGS>
# Profiling is controlled via ROCPROFSYS_PROFILE=ON environment variable
$ROCPROF_SYS_PYTHON_CMD -- ../tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_sys.log
cd - > /dev/null

log_step "Profiling complete!"

# Find generated files
PROTO_FILE=$(find "$OUTPUT_DIR" -name "*.proto" | head -n 1)
ROCPD_FILE=$(find "$OUTPUT_DIR" -name "*.rocpd" | head -n 1)
ROCPD_JSON_FILE=$(find "$OUTPUT_DIR" -name "*.rocpd.json" | head -n 1)

echo ""
log_info "======================================================================"
log_info "rocprof-sys-python Profiling Complete!"
log_info "======================================================================"
echo ""
log_info "Results directory: $OUTPUT_DIR"
echo ""

if [ -f "$ROCPD_FILE" ] || [ -f "$ROCPD_JSON_FILE" ]; then
    if [ -f "$ROCPD_FILE" ]; then
        log_info "ROCPD trace file: $ROCPD_FILE"
        log_info "File size: $(ls -lh "$ROCPD_FILE" | awk '{print $5}')"
    fi
    if [ -f "$ROCPD_JSON_FILE" ]; then
        log_info "ROCPD JSON file: $ROCPD_JSON_FILE"
        log_info "File size: $(ls -lh "$ROCPD_JSON_FILE" | awk '{print $5}')"
    fi
    echo ""
    log_info "ROCPD format is recommended for AI/ML workloads with better thread support."
elif [ -f "$PROTO_FILE" ]; then
    log_info "Perfetto trace file: $PROTO_FILE"
    echo ""
    log_info "To visualize the trace:"
    log_info "  1. Copy .proto file to your local machine"
    log_info "  2. Open https://ui.perfetto.dev in your browser"
    log_info "  3. Click 'Open trace file' and select the .proto file"
    echo ""
    log_info "File size: $(ls -lh "$PROTO_FILE" | awk '{print $5}')"
    log_info "Note: For AI/ML workloads, ROCPD output is recommended over Perfetto."
else
    log_info "No trace file found. Check rocprof_sys.log for profiling output."
    log_info "Python call stack profiling results may be in the log file."
fi

echo ""
log_info "Log file: $OUTPUT_DIR/rocprof_sys.log"
log_info "Check the log for Python call stack profiling output with function call counts and timing."
echo ""


