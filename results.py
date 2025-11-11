import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_process_data(csv_file='benchmark_results.csv'):
    """Load benchmark data and compute statistics"""
    df = pd.read_csv(csv_file)
    
    # Group by method and size, compute statistics
    stats = df.groupby(['method', 'size']).agg({
        'wall_ms': ['mean', 'std', 'min', 'max', 'median'],
        'kernel_ms': ['mean', 'std', 'min', 'max', 'median'],
        'memcpy_htod_ms': ['mean', 'std'],
        'memcpy_dtoh_ms': ['mean', 'std'],
        'gflops': ['mean', 'std', 'min', 'max'],
        'accuracy': 'max'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    return df, stats

def plot_runtime_vs_size(stats, output_dir='plots'):
    """Plot 1: Runtime vs Matrix Size"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1a: Kernel time only (log-log)
    for method in stats['method'].unique():
        data = stats[stats['method'] == method]
        ax1.errorbar(data['size'], data['kernel_ms_mean'], 
                    yerr=data['kernel_ms_std'],
                    marker='o', label=method, linewidth=2, capsize=5)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Kernel Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Kernel Execution Time vs Matrix Size', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 1b: Total wall time (including transfers for CUDA)
    for method in stats['method'].unique():
        data = stats[stats['method'] == method]
        ax2.errorbar(data['size'], data['wall_ms_mean'], 
                    yerr=data['wall_ms_std'],
                    marker='s', label=method, linewidth=2, capsize=5)
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Wall Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Total Time (Kernel + Transfers) vs Matrix Size', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/runtime_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/runtime_vs_size.png")
    plt.close()

def plot_gflops_vs_size(stats, output_dir='plots'):
    """Plot 2: GFLOPS vs Matrix Size"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in stats['method'].unique():
        data = stats[stats['method'] == method]
        ax.errorbar(data['size'], data['gflops_mean'], 
                   yerr=data['gflops_std'],
                   marker='o', label=method, linewidth=2, 
                   markersize=8, capsize=5)
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Performance vs Matrix Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gflops_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/gflops_vs_size.png")
    plt.close()

def plot_speedup(stats, output_dir='plots'):
    """Plot 3: Speedup relative to OpenMP Naive"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    baseline = stats[stats['method'] == 'OpenMP_Naive'][['size', 'kernel_ms_mean']]
    baseline.columns = ['size', 'baseline_time']
    
    for method in stats['method'].unique():
        if method == 'OpenMP_Naive':
            continue
        data = stats[stats['method'] == method].merge(baseline, on='size')
        speedup = data['baseline_time'] / data['kernel_ms_mean']
        ax.plot(data['size'], speedup, marker='o', label=method, 
               linewidth=2, markersize=8)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1x)')
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (relative to OpenMP Naive)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Over OpenMP Naive Implementation', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/speedup_vs_size.png")
    plt.close()

def plot_kernel_vs_transfer(stats, output_dir='plots'):
    """Plot 4: Kernel vs Transfer Time (stacked bar for CUDA)"""
    Path(output_dir).mkdir(exist_ok=True)
    
    cuda_methods = [m for m in stats['method'].unique() if 'CUDA' in m]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(stats[stats['method'] == cuda_methods[0]]['size']))
    width = 0.35
    
    for i, method in enumerate(cuda_methods):
        data = stats[stats['method'] == method]
        
        kernel = data['kernel_ms_mean'].values
        h2d = data['memcpy_htod_ms_mean'].values
        d2h = data['memcpy_dtoh_ms_mean'].values
        
        offset = width * i
        ax.bar(x + offset, kernel, width, label=f'{method} - Kernel')
        ax.bar(x + offset, h2d, width, bottom=kernel, label=f'{method} - H2D')
        ax.bar(x + offset, d2h, width, bottom=kernel+h2d, label=f'{method} - D2H')
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('CUDA: Kernel vs Data Transfer Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(data['size'].values)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_vs_transfer.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/kernel_vs_transfer.png")
    plt.close()

def plot_variance_boxplots(df, output_dir='plots'):
    """Plot 5: Boxplots showing variance across trials"""
    Path(output_dir).mkdir(exist_ok=True)
    
    sizes = sorted(df['size'].unique())
    n_sizes = len(sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 6))
    if n_sizes == 1:
        axes = [axes]
    
    for idx, size in enumerate(sizes):
        data = df[df['size'] == size]
        sns.boxplot(data=data, x='method', y='kernel_ms', ax=axes[idx])
        axes[idx].set_title(f'N = {size}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Method', fontsize=10)
        axes[idx].set_ylabel('Kernel Time (ms)', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/variance_boxplots.png")
    plt.close()

def plot_efficiency_heatmap(stats, theoretical_peak_gflops=None, output_dir='plots'):
    """Plot 6: Efficiency heatmap (if theoretical peak is provided)"""
    if theoretical_peak_gflops is None:
        print("Skipping efficiency heatmap (theoretical_peak_gflops not provided)")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create pivot table
    pivot = stats.pivot(index='method', columns='size', values='gflops_mean')
    efficiency = (pivot / theoretical_peak_gflops) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(efficiency, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Efficiency (%)'}, ax=ax)
    ax.set_title('Compute Efficiency (% of Theoretical Peak)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/efficiency_heatmap.png")
    plt.close()

def plot_comparison_table(stats, output_dir='plots'):
    """Plot 7: Summary comparison table"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get largest size performance
    max_size = stats['size'].max()
    summary = stats[stats['size'] == max_size][['method', 'kernel_ms_mean', 'gflops_mean', 'accuracy']]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row['method'],
            f"{row['kernel_ms_mean']:.2f} ms",
            f"{row['gflops_mean']:.2f}",
            f"{row['accuracy']:.2e}" if row['accuracy'] > 0 else "N/A"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method', 'Time (ms)', 'GFLOPS', 'Max Error'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'Performance Summary (N = {max_size})', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/summary_table.png")
    plt.close()

def generate_report(df, stats, output_file='benchmark_report.txt'):
    """Generate text report with statistics"""
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MATRIX MULTIPLICATION BENCHMARK REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*60 + "\n")
        
        for method in stats['method'].unique():
            f.write(f"\n{method}:\n")
            method_data = stats[stats['method'] == method]
            for _, row in method_data.iterrows():
                f.write(f"  Size {int(row['size'])}:\n")
                f.write(f"    Kernel Time: {row['kernel_ms_mean']:.3f} ± {row['kernel_ms_std']:.3f} ms\n")
                f.write(f"    Performance: {row['gflops_mean']:.2f} ± {row['gflops_std']:.2f} GFLOPS\n")
                if 'CUDA' in method:
                    f.write(f"    H2D Transfer: {row['memcpy_htod_ms_mean']:.3f} ms\n")
                    f.write(f"    D2H Transfer: {row['memcpy_dtoh_ms_mean']:.3f} ms\n")
                if row['accuracy'] > 0:
                    f.write(f"    Max Error: {row['accuracy']:.2e}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SPEEDUP ANALYSIS (relative to OpenMP Naive)\n")
        f.write("="*60 + "\n")
        
        baseline = stats[stats['method'] == 'OpenMP_Naive'][['size', 'kernel_ms_mean']]
        baseline.columns = ['size', 'baseline_time']
        
        for method in stats['method'].unique():
            if method == 'OpenMP_Naive':
                continue
            f.write(f"\n{method}:\n")
            method_data = stats[stats['method'] == method].merge(baseline, on='size')
            for _, row in method_data.iterrows():
                speedup = row['baseline_time'] / row['kernel_ms_mean']
                f.write(f"  Size {int(row['size'])}: {speedup:.2f}x speedup\n")
    
    print(f"\nSaved report: {output_file}")

def main():
    """Main analysis function"""
    print("Loading benchmark data...")
    df, stats = load_and_process_data()
    
    print(f"Data loaded: {len(df)} records")
    print(f"Methods: {df['method'].unique()}")
    print(f"Sizes: {sorted(df['size'].unique())}")
    print(f"Trials per config: {df.groupby(['method', 'size']).size().iloc[0]}")
    
    print("\nGenerating plots...")
    plot_runtime_vs_size(stats)
    plot_gflops_vs_size(stats)
    plot_speedup(stats)
    plot_kernel_vs_transfer(stats)
    plot_variance_boxplots(df)
    plot_comparison_table(stats)
    
    # Optional: if you know your GPU's theoretical peak
    # For example, RTX 3090 ≈ 35.6 TFLOPS FP32 = 35600 GFLOPS
    # Uncomment and adjust:
    # plot_efficiency_heatmap(stats, theoretical_peak_gflops=35600)
    
    print("\nGenerating report...")
    generate_report(df, stats)
    
    print("\n✓ Analysis complete! Check the 'plots' directory for visualizations.")

if __name__ == '__main__':
    main()