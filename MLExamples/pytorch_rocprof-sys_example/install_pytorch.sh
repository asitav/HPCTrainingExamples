#!/bin/bash
# Install PyTorch libraries following TinyOpenFold README.md instructions
# For ROCm 7.1.1 and Python 3.12
# Usage: ./install_pytorch.sh

set -e

cd /mnt/thera/data/incoming/asimishr/aiml_prof/HPCTrainingExamples/MLExamples/pytorch_rocprof-sys_example

# Load modules
module load rocm/7.1.1 python/3.12.10 libffi/3.3

# Activate virtual environment
source venv/bin/activate

echo "Upgrading pip and build tools..."
pip3 install --upgrade pip setuptools wheel

echo "Uninstalling existing PyTorch packages..."
pip3 uninstall -y torch torchvision triton torchaudio 2>/dev/null || true

echo "Installing PyTorch with ROCm 7.1 support..."
# Install PyTorch from PyTorch nightly repository for ROCm 7.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1

echo "Setting up LD_LIBRARY_PATH..."
# Add PyTorch lib directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):/opt/rocm-7.1.1/lib:$LD_LIBRARY_PATH

echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Installation complete!"
echo ""
echo "To use PyTorch, activate the environment:"
echo "  source activate_env.sh"
echo "  export LD_LIBRARY_PATH=\$(python3 -c \"import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\"):/opt/rocm-7.1.1/lib:\$LD_LIBRARY_PATH"
