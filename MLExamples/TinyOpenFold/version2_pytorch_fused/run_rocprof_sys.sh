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
            echo ""
            echo "Configuration:"
            echo "  The script sets up environment variables for rocprof-sys-python:"
            echo "  - Sources setup-env.sh: Automatically sets PYTHONPATH, PATH, LD_LIBRARY_PATH"
            echo "  - PYTHONPATH: Includes rocprofsys package location (if not set by setup-env.sh)"
            echo "  - ROCPROFSYS_PROFILE=ON: Enables profiling"
            echo "  - ROCPROFSYS_USE_ROCPD=ON: Enables ROCPD format (recommended for AI/ML workloads)"
            echo "  - ROCPROFSYS_USE_TRACE=OFF: Disables Perfetto trace (using ROCPD instead)"
            echo "  - PATH: Includes /opt/rocm-7.1.1/share/rocprofiler-systems for schema discovery"
            echo "  - LD_LIBRARY_PATH: Includes PyTorch lib and ROCm lib directories"
            echo ""
            echo "Note: ROCPD format is recommended for AI/ML workloads (better child thread support)"
            echo "      See: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html"
            echo ""
            echo "Note: Default rocprof-sys configuration is used (no config file needed)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Check Python version matches compiled bindings
# The Python interpreter major.minor version must match the version used to compile the bindings
# See: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")
if [ -n "$PYTHON_VERSION" ]; then
    # Check if matching library exists (e.g., libpyrocprofsys.cpython-312-x86_64-linux-gnu.so for Python 3.12)
    PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | tr '.' '_')
    if [ ! -f "/opt/rocm-7.1.1/lib/python${PYTHON_VERSION}/site-packages/rocprofsys/libpyrocprofsys.cpython-${PYTHON_MAJOR_MINOR}-x86_64-linux-gnu.so" ]; then
        log_info "Warning: Python ${PYTHON_VERSION} bindings may not be available."
        log_info "Available bindings: $(find /opt/rocm-7.1.1/lib/python*/site-packages/rocprofsys -name 'libpyrocprofsys*.so' 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo 'Not found')"
        log_info "The Python version must match the version used to compile the bindings."
    fi
fi

# Check for rocprof-sys-python or python3 -m rocprofsys
ROCPROF_SYS_PYTHON_CMD=""
if command -v rocprof-sys-python &> /dev/null; then
    ROCPROF_SYS_PYTHON_CMD="rocprof-sys-python"
    log_rocprof "Using rocprof-sys-python helper script"
elif python3 -m rocprofsys --help &> /dev/null; then
    ROCPROF_SYS_PYTHON_CMD="python3 -m rocprofsys"
    log_rocprof "Using python3 -m rocprofsys"
else
    log_info "rocprof-sys-python not found. Please ensure ROCm Systems Profiler Python bindings are installed."
    log_info "The Python package should be in: /opt/rocm-7.1.1/lib/python*/site-packages/rocprofsys"
    log_info "Or ensure PYTHONPATH includes the rocprofsys package location."
    log_info "You may need to source: /opt/rocm-7.1.1/share/rocprofiler-systems/setup-env.sh"
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

# Source setup-env.sh if available (sets PYTHONPATH, PATH, LD_LIBRARY_PATH automatically)
# See: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html
if [ -f "/opt/rocm-7.1.1/share/rocprofiler-systems/setup-env.sh" ]; then
    source /opt/rocm-7.1.1/share/rocprofiler-systems/setup-env.sh
    log_rocprof "Sourced setup-env.sh for environment configuration"
fi

# Ensure LD_LIBRARY_PATH includes PyTorch lib directory and ROCm lib directory
# This is critical for PyTorch to detect ROCm GPUs and load required libraries
# See: TinyOpenFold/README.md for details
if command -v python3 &> /dev/null; then
    # Add PyTorch lib directory (contains libcaffe2_nvrtc.so and other ROCm libraries)
    PYTORCH_LIB_DIR=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || echo "")
    if [ -n "$PYTORCH_LIB_DIR" ] && [ -d "$PYTORCH_LIB_DIR" ]; then
        export LD_LIBRARY_PATH="${PYTORCH_LIB_DIR}:${LD_LIBRARY_PATH}"
        log_rocprof "Added PyTorch lib directory to LD_LIBRARY_PATH: $PYTORCH_LIB_DIR"
    fi
    
    # Add ROCm lib directory (if not already in LD_LIBRARY_PATH)
    if [[ "$LD_LIBRARY_PATH" != *"/opt/rocm-7.1.1/lib"* ]]; then
        export LD_LIBRARY_PATH="/opt/rocm-7.1.1/lib:${LD_LIBRARY_PATH}"
        log_rocprof "Added ROCm lib directory to LD_LIBRARY_PATH: /opt/rocm-7.1.1/lib"
    fi
    
    # Add system library paths (for libdrm.so.2, libatomic.so.1, etc.)
    if [[ "$LD_LIBRARY_PATH" != *"/usr/lib64"* ]]; then
        export LD_LIBRARY_PATH="/usr/lib64:/lib64:${LD_LIBRARY_PATH}"
        log_rocprof "Added system library paths to LD_LIBRARY_PATH"
    fi
fi

# Ensure PYTHONPATH includes rocprofsys package (if setup-env.sh didn't set it)
# The Python package is installed in lib/pythonX.Y/site-packages/rocprofsys
# See: https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/profiling-python-scripts.html
if [ -z "$PYTHONPATH" ] || [[ "$PYTHONPATH" != *"rocprofsys"* ]]; then
    # Try to find Python version and add appropriate path
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12")
    ROCPROFSYS_PYTHON_PATH="/opt/rocm-7.1.1/lib/python${PYTHON_VERSION}/site-packages"
    if [ -d "$ROCPROFSYS_PYTHON_PATH" ]; then
        export PYTHONPATH="${ROCPROFSYS_PYTHON_PATH}:${PYTHONPATH}"
        log_rocprof "Added $ROCPROFSYS_PYTHON_PATH to PYTHONPATH"
    fi
fi

# Basic system setup for rocprof-sys configuration
# Use default configuration (no config file needed)
unset ROCPROFSYS_CONFIG_FILE

# Enable profiling
export ROCPROFSYS_PROFILE=ON

# ROCPD output configuration
# ROCPD is enabled for AI/ML workloads (better child thread support)
# Note: rocprof-sys looks for schema at relative path "source/lib/core/rocpd/data_storage/schema/rocpd_tables.sql"
# This is hardcoded in the compiled library and may need workaround
export ROCPROFSYS_USE_ROCPD=ON

# Try setting schema path (may not be respected if hardcoded)
export ROCPROFSYS_ROCPD_SCHEMA_PATH=/opt/rocm-7.1.1/share/rocprofiler-systems/rocpd_tables.sql

# Alternative: Create symlink from expected location if environment variable doesn't work
# The error comes from libpyrocprofsys.so looking for: source/lib/core/rocpd/data_storage/schema/rocpd_tables.sql
# This appears to be a build-time path hardcoded in the compiled library

# Trace output configuration (Perfetto format)
# Disabled - using ROCPD format instead
export ROCPROFSYS_USE_TRACE=OFF

# Optional: Enable ROCProfiler integration
# export ROCPROFSYS_USE_ROCPROFILER=ON

# Optional: Configure profiling components (e.g., trip_count, wall_clock, etc.)
# export ROCPROFSYS_TIMEMORY_COMPONENTS="trip_count,wall_clock"

# Verify GPU/ROCm availability before running
log_step "Verifying GPU/ROCm availability..."
if command -v rocm-smi &> /dev/null; then
    log_info "ROCm detected - checking GPU availability..."
    if rocm-smi &> /dev/null; then
        GPU_COUNT=$(rocm-smi --showproductname 2>/dev/null | grep -c "Card series" || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            log_info "Found $GPU_COUNT GPU(s) via rocm-smi"
            rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1 || true
        else
            log_info "rocm-smi available but no GPUs detected"
        fi
    fi
else
    log_info "rocm-smi not found - GPU detection may be limited"
fi

# Verify PyTorch can see ROCm devices
log_step "Verifying PyTorch ROCm support..."
PYTORCH_GPU_CHECK=$(python3 -c "
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f'PyTorch GPU: Available ({torch.cuda.device_count()} device(s))')
        for i in range(torch.cuda.device_count()):
            print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        sys.exit(0)
    else:
        print('PyTorch GPU: Not available')
        print('  torch.cuda.is_available() = False')
        sys.exit(1)
except Exception as e:
    print(f'PyTorch GPU check failed: {e}')
    sys.exit(1)
" 2>&1)

if echo "$PYTORCH_GPU_CHECK" | grep -q "Not available\|failed"; then
    log_info "Warning: PyTorch cannot detect GPU devices"
    log_info "This may cause DeepSpeed to fall back to CPU mode"
    log_info ""
    log_info "Common causes:"
    log_info "  1. Missing ROCm libraries in LD_LIBRARY_PATH"
    log_info "  2. PyTorch not built with ROCm support"
    log_info "  3. HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES incorrectly set"
    log_info ""
    log_info "Current LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
    log_info "Current HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
    log_info "Current ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "$PYTORCH_GPU_CHECK"
fi
echo ""

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


