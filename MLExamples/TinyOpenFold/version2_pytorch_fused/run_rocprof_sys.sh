#!/bin/bash

# rocprof-sys (System) Profiling Integration for Tiny OpenFold V2
# This script provides comprehensive system-level profiling with timeline tracing

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

# Default configuration
BATCH_SIZE=4
SEQ_LEN=64
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
            echo "rocprof-sys Profiling for Tiny OpenFold V2"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N          Batch size (default: 4)"
            echo "  --seq-len N             Sequence length (default: 64)"
            echo "  --num-blocks N          Number of Evoformer blocks (default: 4)"
            echo "  --num-seqs N            Number of MSA sequences (default: 16)"
            echo "  --num-steps N           Training steps (default: 30)"
            echo "  --output-dir DIR        Output directory"
            echo "  --disable-all-fusion    Disable all fusions"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Profile with all fusions"
            echo "  $0 --batch-size 8 --seq-len 128      # Larger workload"
            echo "  $0 --disable-all-fusion              # Baseline comparison"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check for rocprof-sys
if ! command -v rocprof-sys &> /dev/null && ! command -v rocprof-sys-run &> /dev/null; then
    log_info "rocprof-sys not found. Please ensure ROCm tools are installed."
    exit 1
fi

# Detect correct command
ROCPROF_SYS_CMD="rocprof-sys-run"
if ! command -v $ROCPROF_SYS_CMD &> /dev/null; then
    ROCPROF_SYS_CMD="rocprof-sys"
fi

mkdir -p "$OUTPUT_DIR"

log_info "======================================================================"
log_info "Tiny OpenFold V2 - rocprof-sys Profiling"
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

# Run profiling
log_step "Starting rocprof-sys profiling..."
log_rocprof "This will generate a timeline trace (.proto file)"
echo ""

cd "$OUTPUT_DIR"
$ROCPROF_SYS_CMD --profile --trace -- python ../tiny_openfold_v2.py $PYTHON_ARGS 2>&1 | tee rocprof_sys.log
cd - > /dev/null

log_step "Profiling complete!"

# Find generated files
PROTO_FILE=$(find "$OUTPUT_DIR" -name "*.proto" | head -n 1)

echo ""
log_info "======================================================================"
log_info "rocprof-sys Profiling Complete!"
log_info "======================================================================"
echo ""
log_info "Results directory: $OUTPUT_DIR"
echo ""

if [ -f "$PROTO_FILE" ]; then
    log_info "Timeline trace: $PROTO_FILE"
    echo ""
    log_info "To visualize the trace:"
    log_info "  1. Copy .proto file to your local machine"
    log_info "  2. Open https://ui.perfetto.dev in your browser"
    log_info "  3. Click 'Open trace file' and select the .proto file"
    echo ""
    log_info "File size: $(ls -lh "$PROTO_FILE" | awk '{print $5}')"
else
    log_info "No .proto file found. Check rocprof_sys.log for errors."
fi

echo ""
log_info "Log file: $OUTPUT_DIR/rocprof_sys.log"
echo ""


