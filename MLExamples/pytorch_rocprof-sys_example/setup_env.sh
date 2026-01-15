#!/bin/bash
# Setup script for GPU tracing example on compute node
# Usage: ./setup_env.sh [node_name]

set -e

NODE=${1:-TheraC18}
WORK_DIR="/mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/pytorch_rocprof-sys_example"

echo "Setting up environment on $NODE..."
ssh $NODE "cd $WORK_DIR && \
    module load rocm/7.1.1 python/3.12.10 && \
    python3 -m venv venv && \
    echo 'Setup complete!' && \
    echo 'Modules loaded:' && \
    module list && \
    echo '' && \
    echo 'Python version:' && \
    python3 --version"

echo ""
echo "To activate the environment on $NODE:"
echo "  ssh $NODE"
echo "  cd $WORK_DIR"
echo "  module load rocm/7.1.1 python/3.12.10"
echo "  source venv/bin/activate"
