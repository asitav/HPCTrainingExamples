# MLCommons TinyOpenFold: Educational AlphaFold 2 Benchmark

**Educational implementation of AlphaFold 2's Evoformer mimicking the MLCommons HPC OpenFold benchmark structure**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## Overview

This is an **educational** implementation of the Evoformer architecture from AlphaFold 2, designed to reproduce the key aspects of the [MLCommons HPC OpenFold benchmark](https://github.com/mlcommons/hpc/tree/main/openfold/) at a smaller scale for learning and experimentation.

### Key Features

- **Core Evoformer Architecture**: Complete implementation of MSA and pair representations with attention mechanisms
- **MLCommons-Style Benchmarking**: Aligned metrics and reporting similar to official MLCommons HPC benchmarks
- **Multi-GPU Support**: DataParallel support for scaling studies
- **Performance Profiling**: Integrated PyTorch profiler and DeepSpeed FLOPS profiler
- **Deterministic Execution**: Reproducible training runs with fixed seeds
- **Workshop Structure**: Progressive exercises from baseline to optimization

### What's Included

```
MLCommons_TinyOpenFold/
├── README.md                      # This file
├── ARCHITECTURE.md                # Detailed architecture documentation
├── TECHNICAL_APPENDICES.md        # Performance analysis and profiling
├── MLCOMMONS_COMPARISON.md        # Comparison with full MLCommons benchmark
├── requirements.txt               # Python dependencies
├── setup/                         # Environment setup scripts
├── version1_pytorch_baseline/     # V1: PyTorch baseline implementation
│   ├── tiny_openfold_mlc.py      # Main implementation
│   ├── run_baseline.sh           # Basic training script
│   ├── run_scaling_study.sh      # Multi-GPU scaling study
│   ├── run_pytorch_profiler.py   # PyTorch profiler
│   ├── run_deepspeed_flops.py    # DeepSpeed FLOPS profiler
│   ├── run_all_profilers.sh      # Run all profilers
│   ├── exercises/                # Workshop exercises
│   └── README.md                 # Version-specific documentation
└── benchmark/                     # Benchmarking framework
    ├── benchmark_runner.py       # Automated benchmark orchestration
    ├── metrics_collector.py      # Metrics collection
    ├── scaling_analyzer.py       # Scaling efficiency analysis
    ├── report_generator.py       # Report generation
    └── configs/                  # Benchmark configurations
```

## Quick Start

### Installation

```bash
# Clone repository (if not already)
cd HPCTrainingExamples/MLExamples/MLCommons_TinyOpenFold

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Basic Training

```bash
cd version1_pytorch_baseline

# Quick test (32x32 sequence, 4 blocks, 50 steps)
python tiny_openfold_mlc.py --n-seq 32 --seq-len 32 --num-steps 50

# Standard benchmark (32x64 sequence, 4 blocks, 100 steps)
python tiny_openfold_mlc.py --batch-size 4 --seq-len 64 --num-steps 100

# Multi-GPU training
python tiny_openfold_mlc.py --batch-size 8 --use-multi-gpu --num-steps 100
```

### Run Profiling

```bash
# PyTorch profiler
./run_pytorch_profiler.py

# DeepSpeed FLOPS profiler
./run_deepspeed_flops.py

# All profilers
./run_all_profilers.sh
```

### Run Scaling Study

```bash
# Multi-GPU scaling study
./run_scaling_study.sh
```

## Workshop Structure

This repository follows a progressive workshop structure:

### Version 1: PyTorch Baseline (`version1_pytorch_baseline/`)

**Focus**: Understanding the baseline implementation and identifying bottlenecks

- Complete Evoformer architecture with all attention mechanisms
- MLCommons-style benchmarking and metrics
- PyTorch profiler for performance analysis
- DeepSpeed FLOPS profiler for computational analysis
- Multi-GPU scaling experiments

**Exercises**:
1. Baseline performance analysis
2. Memory profiling and optimization
3. Multi-GPU scaling study

### Future Versions (Planned)

- **Version 2**: Fused operations and kernel optimizations
- **Version 3**: Triton kernels for custom operations
- **Version 4**: Advanced optimizations (FlashAttention, mixed precision)

## Architecture

The implementation includes all key components of the Evoformer:

### MSA (Multiple Sequence Alignment) Stack
- **MSA Row Attention with Pair Bias**: Attention over sequences at each residue position
- **MSA Column Attention**: Global attention over residue positions
- **MSA Transition**: Feed-forward network for MSA representation
- **Outer Product Mean**: Communication from MSA to pair representation

### Pair Stack
- **Triangle Multiplicative Update** (Outgoing/Incoming): Geometric reasoning
- **Triangle Attention** (Starting/Ending): Attention along triangle edges
- **Pair Transition**: Feed-forward network for pair representation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## MLCommons Benchmark Alignment

This implementation aligns with the MLCommons HPC OpenFold benchmark in several ways:

### Similarities
- Core Evoformer architecture and attention mechanisms
- Training loop structure and optimization approach
- Benchmarking metrics (time-to-train, throughput, scalability)
- Multi-GPU training support

### Differences
- **Scale**: Smaller model for educational purposes (256/128 channels vs 256/128+ in production)
- **Dataset**: Synthetic random data instead of real protein databases
- **Complexity**: Simplified structure prediction head vs. full structure module
- **Convergence**: Fixed steps vs. convergence-based stopping

See [MLCOMMONS_COMPARISON.md](MLCOMMONS_COMPARISON.md) for detailed comparison.

## Performance Characteristics

### Model Size (Default Config)
- **Parameters**: ~15M trainable parameters
- **MSA channels**: 256
- **Pair channels**: 128
- **Evoformer blocks**: 4
- **Attention heads**: 8

### Throughput (Example: MI250X single GCD)
- **Batch size 4, seq_len 64**: ~2-3 samples/sec
- **Memory**: ~8-12 GB
- **FLOPs**: ~500 GFLOPs per forward pass

### Scaling
- Near-linear scaling up to 4 GPUs
- Efficiency ~85-90% at 2 GPUs
- Communication overhead becomes significant beyond 4 GPUs

See [TECHNICAL_APPENDICES.md](TECHNICAL_APPENDICES.md) for detailed performance analysis.

## Command-Line Arguments

### Model Configuration
- `--n-seq N`: Number of sequences in MSA (default: 32)
- `--seq-len L`: Sequence length (default: 64)
- `--c-m C`: MSA representation channels (default: 256)
- `--c-z C`: Pair representation channels (default: 128)
- `--n-blocks N`: Number of Evoformer blocks (default: 4)
- `--n-heads H`: Number of attention heads (default: 8)

### Training Configuration
- `--batch-size B`: Training batch size (default: 4)
- `--num-steps S`: Number of training steps (default: 100)
- `--learning-rate LR`: Learning rate (default: 1e-4)
- `--seed S`: Random seed (default: 42)
- `--use-multi-gpu`: Enable multi-GPU training
- `--use-amp`: Enable automatic mixed precision

### Benchmarking Configuration
- `--warmup-steps W`: Number of warmup steps (default: 10)
- `--measured-steps M`: Number of measured steps (default: 50)
- `--results-dir DIR`: Results output directory (default: benchmark_results)

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed Evoformer architecture with parameter counts and complexity analysis
- **[TECHNICAL_APPENDICES.md](TECHNICAL_APPENDICES.md)**: Performance analysis, profiling tools, optimization patterns
- **[MLCOMMONS_COMPARISON.md](MLCOMMONS_COMPARISON.md)**: Comparison with full MLCommons HPC OpenFold benchmark
- **[version1_pytorch_baseline/README.md](version1_pytorch_baseline/README.md)**: Version-specific documentation and exercises

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ or ROCm 5.4+ (for GPU acceleration)
- 16GB+ GPU memory recommended
- See [requirements.txt](requirements.txt) for complete dependencies

## Citation

If you use this educational implementation in your teaching or research, please cite the original AlphaFold 2 paper:

```bibtex
@article{jumper2021highly,
  title={Highly accurate protein structure prediction with AlphaFold},
  author={Jumper, John and Evans, Richard and Pritzel, Alexander and others},
  journal={Nature},
  volume={596},
  number={7873},
  pages={583--589},
  year={2021},
  publisher={Nature Publishing Group}
}
```

And the MLCommons HPC OpenFold benchmark:

```bibtex
@misc{mlcommons2023hpc,
  title={MLCommons HPC Benchmark Suite},
  author={MLCommons},
  year={2023},
  url={https://github.com/mlcommons/hpc}
}
```

## License

This educational implementation is provided under the Apache 2.0 License.

## Acknowledgments

- **DeepMind**: Original AlphaFold 2 architecture and paper
- **OpenFold**: Open-source AlphaFold 2 implementation
- **MLCommons**: HPC benchmark suite and evaluation framework
- **AMD**: ROCm platform and profiling tools

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- See workshop exercises for guided exploration
- Check TECHNICAL_APPENDICES.md for troubleshooting

## Getting Started Next Steps

1. **Run baseline**: `cd version1_pytorch_baseline && python tiny_openfold_mlc.py`
2. **Profile performance**: `./run_pytorch_profiler.py`
3. **Try exercises**: `cd exercises/ && less exercise_1_baseline_analysis.md`
4. **Scale up**: Try `--use-multi-gpu` for multi-GPU training
5. **Explore architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)

Happy learning!
