#!/bin/bash
# Activate environment script - run this on the compute node
# Usage: source activate_env.sh

cd /mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/pytorch_rocprof-sys_example
module load rocm/7.1.1 python/3.12.10 libffi/3.3
source venv/bin/activate

# Set up LD_LIBRARY_PATH for PyTorch libraries (if PyTorch is installed)
if python3 -c "import torch" 2>/dev/null; then
    export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):/opt/rocm-7.1.1/lib:$LD_LIBRARY_PATH
fi

echo "Environment activated!"
echo "Python: $(python --version)"
echo "Device: $(python -c 'import torch; print("CUDA available:", torch.cuda.is_available())' 2>/dev/null || echo "PyTorch not installed")"
