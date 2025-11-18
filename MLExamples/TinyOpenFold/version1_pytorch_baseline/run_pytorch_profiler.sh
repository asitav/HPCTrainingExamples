#!/bin/bash
# Run TinyOpenFold V1 with PyTorch Profiler

set -e

echo "========================================================================"
echo "Running TinyOpenFold V1 - PyTorch Profiler"
echo "========================================================================"

# Create profile directory
mkdir -p pytorch_profiles

python tiny_openfold_v1.py \
    --enable-pytorch-profiler \
    --device 0 \
    --batch-size 4 \
    --num-steps 50 \
    --seq-len 64 \
    --num-seqs 16 \
    --profile-dir pytorch_profiles

echo ""
echo "PyTorch profiler run completed!"
echo "Profile data saved to: pytorch_profiles/"
echo "Launch TensorBoard: tensorboard --logdir pytorch_profiles"

