#!/usr/bin/env python3
"""Analyze performance study results."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results(study_dir):
    """Load all performance results."""
    results = {}
    study_path = Path(study_dir)
    
    for version in ['v1_baseline', 'v2_fused', 'v3_triton']:
        results[version] = []
        
        for run_dir in sorted(study_path.glob(f'{version}_run*')):
            # Try different file names
            for filename in ['performance_summary.json', 'performance_summary_v2.json', 'performance_summary_v3.json']:
                json_file = run_dir / filename
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        results[version].append(data)
                    break
    
    return results

def compute_statistics(results):
    """Compute mean and std for each metric."""
    stats = {}
    
    for version, runs in results.items():
        if not runs:
            continue
        
        stats[version] = {}
        
        # Extract metrics from all runs
        metrics = {}
        for run in runs:
            perf = run.get('performance_summary', {})
            for key, value in perf.items():
                if isinstance(value, (int, float)):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Compute statistics (convert numpy types to Python native types for JSON)
        for metric, values in metrics.items():
            stats[version][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return stats

def create_comparison_plots(stats, output_dir):
    """Create comparison plots."""
    output_path = Path(output_dir)
    
    # Training speed comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    versions = list(stats.keys())
    speeds = [stats[v]['avg_training_speed']['mean'] for v in versions if 'avg_training_speed' in stats[v]]
    errors = [stats[v]['avg_training_speed']['std'] for v in versions if 'avg_training_speed' in stats[v]]
    
    x = np.arange(len(versions))
    bars = ax.bar(x, speeds, yerr=errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Version', fontsize=12)
    ax.set_ylabel('Training Speed (samples/sec)', fontsize=12)
    ax.set_title('TinyOpenFold Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['V1: Baseline', 'V2: Fused', 'V3: Triton'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, speed) in enumerate(zip(bars, speeds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'performance_comparison.png'}")
    plt.close()
    
    # Memory usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    memory = [stats[v]['peak_memory_mb']['mean'] for v in versions if 'peak_memory_mb' in stats[v]]
    memory_errors = [stats[v]['peak_memory_mb']['std'] for v in versions if 'peak_memory_mb' in stats[v]]
    
    bars = ax.bar(x, memory, yerr=memory_errors, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Version', fontsize=12)
    ax.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['V1: Baseline', 'V2: Fused', 'V3: Triton'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mem) in enumerate(zip(bars, memory)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'memory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path / 'memory_comparison.png'}")
    plt.close()

def generate_summary_report(stats, config, output_dir):
    """Generate markdown summary report."""
    output_path = Path(output_dir)
    
    with open(output_path / 'results_summary.md', 'w') as f:
        f.write('# TinyOpenFold Performance Study Results\n\n')
        f.write(f"**Study Date**: {config.get('timestamp', 'N/A')}\n\n")
        f.write(f"**Configuration**:\n")
        f.write(f"- Batch size: {config.get('batch_size', 'N/A')}\n")
        f.write(f"- Sequence length: {config.get('seq_len', 'N/A')}\n")
        f.write(f"- Training steps: {config.get('num_steps', 'N/A')}\n")
        f.write(f"- Runs per version: {config.get('num_runs', 'N/A')}\n\n")
        
        f.write('## Performance Summary\n\n')
        f.write('| Metric | V1 Baseline | V2 Fused | V3 Triton | V3 vs V1 |\n')
        f.write('|--------|-------------|----------|-----------|----------|\n')
        
        # Training speed
        v1_speed = stats.get('v1_baseline', {}).get('avg_training_speed', {}).get('mean', 0)
        v2_speed = stats.get('v2_fused', {}).get('avg_training_speed', {}).get('mean', 0)
        v3_speed = stats.get('v3_triton', {}).get('avg_training_speed', {}).get('mean', 0)
        
        speedup = v3_speed / v1_speed if v1_speed > 0 else 0
        
        f.write(f'| Training Speed (samples/s) | {v1_speed:.1f} | {v2_speed:.1f} | {v3_speed:.1f} | {speedup:.2f}x |\n')
        
        # Memory usage
        v1_mem = stats.get('v1_baseline', {}).get('peak_memory_mb', {}).get('mean', 0)
        v2_mem = stats.get('v2_fused', {}).get('peak_memory_mb', {}).get('mean', 0)
        v3_mem = stats.get('v3_triton', {}).get('peak_memory_mb', {}).get('mean', 0)
        
        mem_reduction = (v1_mem - v3_mem) / v1_mem * 100 if v1_mem > 0 else 0
        
        f.write(f'| Peak Memory (MB) | {v1_mem:.1f} | {v2_mem:.1f} | {v3_mem:.1f} | {mem_reduction:.1f}% reduction |\n')
        
        # Batch time
        v1_batch = stats.get('v1_baseline', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        v2_batch = stats.get('v2_fused', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        v3_batch = stats.get('v3_triton', {}).get('avg_batch_time', {}).get('mean', 0) * 1000
        
        f.write(f'| Batch Time (ms) | {v1_batch:.1f} | {v2_batch:.1f} | {v3_batch:.1f} | {v1_batch/v3_batch:.2f}x faster |\n')
        
        f.write('\n## Detailed Results\n\n')
        
        for version in ['v1_baseline', 'v2_fused', 'v3_triton']:
            if version not in stats:
                continue
            
            f.write(f'### {version.upper()}\n\n')
            f.write('| Metric | Mean | Std Dev | Min | Max |\n')
            f.write('|--------|------|---------|-----|-----|\n')
            
            for metric, values in stats[version].items():
                if metric == 'avg_training_speed':
                    f.write(f"| Training Speed (s/s) | {values['mean']:.2f} | {values['std']:.2f} | {values['min']:.2f} | {values['max']:.2f} |\n")
                elif metric == 'peak_memory_mb':
                    f.write(f"| Peak Memory (MB) | {values['mean']:.1f} | {values['std']:.1f} | {values['min']:.1f} | {values['max']:.1f} |\n")
                elif 'time' in metric.lower():
                    f.write(f"| {metric} (ms) | {values['mean']*1000:.2f} | {values['std']*1000:.2f} | {values['min']*1000:.2f} | {values['max']*1000:.2f} |\n")
            
            f.write('\n')
        
        f.write('## Key Findings\n\n')
        f.write(f'1. **Performance**: Version 3 achieves {speedup:.2f}x speedup over baseline\n')
        f.write(f'2. **Memory**: {mem_reduction:.1f}% reduction in peak memory usage\n')
        f.write(f'3. **Optimizations**: Triton custom kernels provide significant improvements\n')
        f.write('\n')
        f.write('## Plots\n\n')
        f.write('![Performance Comparison](performance_comparison.png)\n\n')
        f.write('![Memory Comparison](memory_comparison.png)\n\n')
    
    print(f"  Saved: {output_path / 'results_summary.md'}")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <study_dir>")
        sys.exit(1)
    
    study_dir = sys.argv[1]
    
    print(f"Analyzing results from: {study_dir}")
    print("")
    
    # Load configuration
    config_file = Path(study_dir) / 'config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load results
    print("Loading results...")
    results = load_results(study_dir)
    
    for version, runs in results.items():
        print(f"  {version}: {len(runs)} runs")
    print("")
    
    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(results)
    
    # Save statistics
    stats_file = Path(study_dir) / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_file}")
    print("")
    
    # Create plots
    print("Creating plots...")
    create_comparison_plots(stats, study_dir)
    print("")
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(stats, config, study_dir)
    print("")
    
    print("Analysis complete!")

if __name__ == '__main__':
    main()
