#!/usr/bin/env python3
"""
Simple PyTorch GPU example for tracing/profiling.
This script demonstrates a basic neural network forward pass on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model and move to GPU
model = SimpleNet().to(device)
model.eval()  # Set to evaluation mode

# Create dummy input data
batch_size = 32
input_data = torch.randn(batch_size, 3, 32, 32).to(device)

print(f"Input shape: {input_data.shape}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Warmup run (optional, helps with GPU initialization)
with torch.no_grad():
    _ = model(input_data)

# Synchronize to ensure warmup is complete
if device.type == 'cuda':
    torch.cuda.synchronize()

print("Starting forward pass...")

# Forward pass - this is what you'll trace/profile
with torch.no_grad():
    output = model(input_data)

# Synchronize to ensure computation is complete
if device.type == 'cuda':
    torch.cuda.synchronize()

print(f"Output shape: {output.shape}")
print("Forward pass completed!")
