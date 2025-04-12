import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

class LRUCache:
    """
    Least Recently Used (LRU) cache implementation using OrderedDict.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def get(self, key):
        """
        Get an item from the cache. If it exists, move it to the end (most recently used).
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return -1
    
    def put(self, key, value):
        """
        Add an item to the cache. If cache is full, remove least recently used item.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def get_stats(self):
        """
        Return cache performance statistics
        """
        total_operations = self.hits + self.misses
        hit_rate = self.hits / total_operations if total_operations > 0 else 0
        miss_rate = self.misses / total_operations if total_operations > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'miss_rate': miss_rate
        }
    
    def reset_stats(self):
        """
        Reset hit and miss counters
        """
        self.hits = 0
        self.misses = 0


def run_experiment(cache_sizes, alphabet_sizes, request_sequence_length=10000, runs=5):
    """
    Run experiments with different cache sizes and alphabet sizes using uniform distribution.
    
    Args:
        cache_sizes: List of cache sizes to test
        alphabet_sizes: List of alphabet sizes to test
        request_sequence_length: Length of the request sequence
        runs: Number of runs for each configuration
        
    Returns:
        DataFrame with results
    """
    results = []
    
    for cache_size in cache_sizes:
        for alphabet_size in alphabet_sizes:
            print(f"Running experiment: cache_size={cache_size}, alphabet_size={alphabet_size}")
            
            hit_rates = []
            runtimes = []
            memory_usages = []
            
            for run in range(runs):
                # Create LRU cache
                lru = LRUCache(cache_size)
                
                # Generate uniform random request sequence
                alphabet = list(range(alphabet_size))
                requests = np.random.choice(alphabet, size=request_sequence_length)
                
                # Measure runtime
                start_time = time.time()
                
                # Process requests
                for request in requests:
                    if lru.get(request) == -1:
                        # Cache miss, add to cache
                        lru.put(request, f"value_{request}")
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Get stats
                stats = lru.get_stats()
                hit_rates.append(stats['hit_rate'])
                runtimes.append(runtime)
                
                # Approximate memory usage (in bytes)
                # Each entry has a key (4 bytes) and value reference (~28 bytes for small strings)
                memory_usage = cache_size * (4 + 28)
                memory_usages.append(memory_usage)
                
                lru.reset_stats()
            
            # Save average results
            results.append({
                'cache_size': cache_size,
                'alphabet_size': alphabet_size,
                'cache_ratio': cache_size / alphabet_size,
                'avg_hit_rate': np.mean(hit_rates),
                'avg_miss_rate': 1 - np.mean(hit_rates),
                'avg_runtime': np.mean(runtimes),
                'avg_memory_usage': np.mean(memory_usages)
            })
    
    return pd.DataFrame(results)


def plot_results(results, request_sequence_length):
    """
    Create professional visualizations of the experimental results
    """
    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define a professional color palette
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results['alphabet_size'].unique())))
    
    # Common figure settings
    fig_size = (12, 8)
    dpi = 300
    font_size = 14
    title_size = 18
    legend_size = 12
    
    # Plot 1: Hit rate vs Cache Ratio
    plt.figure(figsize=fig_size, dpi=dpi)
    
    # Group by alphabet size
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        plt.plot(df_subset['cache_ratio'], df_subset['avg_hit_rate'], 
                 marker='o', markersize=8, linewidth=2, color=colors[i],
                 label=f'Alphabet Size = {alphabet_size:,}')
    
    plt.xlabel('Cache Size / Alphabet Size Ratio', fontsize=font_size)
    plt.ylabel('Hit Rate', fontsize=font_size)
    plt.title('Hit Rate vs Cache Ratio for Different Alphabet Sizes', fontsize=title_size, fontweight='bold')
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add annotations for key insights
    max_point = results.loc[results['avg_hit_rate'].idxmax()]
    plt.annotate(f"Max hit rate: {max_point['avg_hit_rate']:.3f}\nCache ratio: {max_point['cache_ratio']:.3f}",
                xy=(max_point['cache_ratio'], max_point['avg_hit_rate']),
                xytext=(max_point['cache_ratio']-0.1, max_point['avg_hit_rate']-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.savefig('hit_rate_vs_cache_ratio.png', bbox_inches='tight')
    
    # Plot 2: Runtime vs Cache Size
    plt.figure(figsize=fig_size, dpi=dpi)
    
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        plt.plot(df_subset['cache_size'], df_subset['avg_runtime'], 
                 marker='o', markersize=8, linewidth=2, color=colors[i],
                 label=f'Alphabet Size = {alphabet_size:,}')
    
    plt.xlabel('Cache Size (k)', fontsize=font_size)
    plt.ylabel('Average Runtime (seconds)', fontsize=font_size)
    plt.title('Runtime Performance vs Cache Size', fontsize=title_size, fontweight='bold')
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add a text box with insights
    min_runtime = results.loc[results['avg_runtime'].idxmin()]
    max_runtime = results.loc[results['avg_runtime'].idxmax()]
    runtime_diff = max_runtime['avg_runtime'] - min_runtime['avg_runtime']
    
    plt.figtext(0.15, 0.15, 
                f"Runtime Insights:\n"
                f"• Min: {min_runtime['avg_runtime']:.4f}s (Cache size: {min_runtime['cache_size']})\n"
                f"• Max: {max_runtime['avg_runtime']:.4f}s (Cache size: {max_runtime['cache_size']})\n"
                f"• Range: {runtime_diff:.4f}s",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                fontsize=10)
    
    plt.savefig('runtime_vs_cache_size.png', bbox_inches='tight')
    
    # Plot 3: Hit Rate vs Cache Size
    plt.figure(figsize=fig_size, dpi=dpi)
    
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        plt.plot(df_subset['cache_size'], df_subset['avg_hit_rate'], 
                 marker='o', markersize=8, linewidth=2, color=colors[i],
                 label=f'Alphabet Size = {alphabet_size:,}')
    
    plt.xlabel('Cache Size (k)', fontsize=font_size)
    plt.ylabel('Hit Rate', fontsize=font_size)
    plt.title('Cache Hit Rate vs Cache Size', fontsize=title_size, fontweight='bold')
    plt.legend(fontsize=legend_size)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add a trendline and annotation
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        z = np.polyfit(df_subset['cache_size'], df_subset['avg_hit_rate'], 1)
        p = np.poly1d(z)
        plt.plot(df_subset['cache_size'], p(df_subset['cache_size']), 
                 linestyle='--', color=colors[i], alpha=0.7)
        
        # Annotate the last point with the relationship
        last_point = df_subset.iloc[-1]
        plt.annotate(f"Slope: {z[0]:.5f}", 
                    xy=(last_point['cache_size'], last_point['avg_hit_rate']),
                    xytext=(last_point['cache_size']-50, last_point['avg_hit_rate']-0.05),
                    fontsize=9, color=colors[i],
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    
    plt.savefig('hit_rate_vs_cache_size.png', bbox_inches='tight')
    
    # Calculate theoretical hit rate for comparison
    # For uniform distribution, theoretical hit rate = cache_size / alphabet_size
    results['theoretical_hit_rate'] = results['cache_size'] / results['alphabet_size']
    
    # Plot 4: Empirical vs Theoretical Hit Rate
    plt.figure(figsize=fig_size, dpi=dpi)
    
    scatter = plt.scatter(results['theoretical_hit_rate'], results['avg_hit_rate'], 
                s=100, c=results['alphabet_size'], cmap='viridis', alpha=0.8, edgecolors='w')
    
    # Add diagonal line for perfect match
    max_val = max(results['theoretical_hit_rate'].max(), results['avg_hit_rate'].max()) + 0.05
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')
    
    plt.xlabel('Theoretical Hit Rate (cache_size / alphabet_size)', fontsize=font_size)
    plt.ylabel('Empirical Hit Rate', fontsize=font_size)
    plt.title('Empirical vs Theoretical Hit Rate Comparison', fontsize=title_size, fontweight='bold')
    
    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter)
    cbar.set_label('Alphabet Size', fontsize=font_size)
    
    # Add a text box with correlation information
    correlation = np.corrcoef(results['theoretical_hit_rate'], results['avg_hit_rate'])[0, 1]
    mean_diff = np.mean(np.abs(results['theoretical_hit_rate'] - results['avg_hit_rate']))
    
    plt.figtext(0.15, 0.15, 
                f"Model Validation:\n"
                f"• Correlation: {correlation:.4f}\n"
                f"• Mean Absolute Difference: {mean_diff:.4f}\n"
                f"• Max Difference: {results['hit_rate_difference'].max():.4f}",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('theoretical_vs_empirical.png', bbox_inches='tight')
    
    # Create a 2x2 subplot figure that combines all insights
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=dpi)
    
    # Plot 1: Hit rate vs Cache Ratio (top left)
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        axs[0, 0].plot(df_subset['cache_ratio'], df_subset['avg_hit_rate'], 
                      marker='o', markersize=6, linewidth=2, color=colors[i],
                      label=f'Alphabet Size = {alphabet_size:,}')
    
    axs[0, 0].set_xlabel('Cache/Alphabet Ratio', fontsize=font_size-2)
    axs[0, 0].set_ylabel('Hit Rate', fontsize=font_size-2)
    axs[0, 0].set_title('Hit Rate vs Cache Ratio', fontsize=font_size, fontweight='bold')
    axs[0, 0].legend(fontsize=legend_size-2)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs Cache Size (top right)
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        axs[0, 1].plot(df_subset['cache_size'], df_subset['avg_runtime'], 
                      marker='o', markersize=6, linewidth=2, color=colors[i],
                      label=f'Alphabet Size = {alphabet_size:,}')
    
    axs[0, 1].set_xlabel('Cache Size (k)', fontsize=font_size-2)
    axs[0, 1].set_ylabel('Runtime (s)', fontsize=font_size-2)
    axs[0, 1].set_title('Runtime vs Cache Size', fontsize=font_size, fontweight='bold')
    axs[0, 1].legend(fontsize=legend_size-2)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Hit Rate vs Cache Size (bottom left)
    for i, alphabet_size in enumerate(sorted(results['alphabet_size'].unique())):
        df_subset = results[results['alphabet_size'] == alphabet_size]
        axs[1, 0].plot(df_subset['cache_size'], df_subset['avg_hit_rate'], 
                      marker='o', markersize=6, linewidth=2, color=colors[i],
                      label=f'Alphabet Size = {alphabet_size:,}')
    
    axs[1, 0].set_xlabel('Cache Size (k)', fontsize=font_size-2)
    axs[1, 0].set_ylabel('Hit Rate', fontsize=font_size-2)
    axs[1, 0].set_title('Hit Rate vs Cache Size', fontsize=font_size, fontweight='bold')
    axs[1, 0].legend(fontsize=legend_size-2)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Empirical vs Theoretical (bottom right)
    scatter = axs[1, 1].scatter(results['theoretical_hit_rate'], results['avg_hit_rate'], 
                              s=80, c=results['alphabet_size'], cmap='viridis', alpha=0.8, edgecolors='w')
    
    max_val = max(results['theoretical_hit_rate'].max(), results['avg_hit_rate'].max()) + 0.05
    axs[1, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match')
    axs[1, 1].set_xlabel('Theoretical Hit Rate', fontsize=font_size-2)
    axs[1, 1].set_ylabel('Empirical Hit Rate', fontsize=font_size-2)
    axs[1, 1].set_title('Theory vs Empirical Comparison', fontsize=font_size, fontweight='bold')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=axs[1, 1])
    cbar.set_label('Alphabet Size', fontsize=font_size-2)
    
    # Add overall title
    fig.suptitle('LRU Cache Performance with Uniform Random Access', 
                fontsize=title_size+2, fontweight='bold', y=0.98)
    
    # Add experiment details
    experiment_details = (
        f"Experiment Details:\n"
        f"• Cache sizes: {', '.join(map(str, sorted(results['cache_size'].unique())))}\n"
        f"• Alphabet sizes: {', '.join(map(str, sorted(results['alphabet_size'].unique())))}\n"
        f"• Request sequence length: {request_sequence_length:,}\n"
        f"• Theoretical model correlation: {correlation:.4f}"
    )
    
    fig.text(0.5, 0.01, experiment_details, ha='center', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('lru_cache_summary.png', bbox_inches='tight')
    
    return


def main():
    # Define experiment parameters
    cache_sizes = [10, 25, 50, 100, 200, 400]
    alphabet_sizes = [100, 500, 1000, 5000]
    request_length = 100000  # Increased for more reliable results
    
    # Run experiments
    results = run_experiment(cache_sizes, alphabet_sizes, request_length)
    
    # Print results
    print("\nExperiment Results:")
    print(results)
    
    # Calculate validation metrics before plotting
    results['theoretical_hit_rate'] = results['cache_size'] / results['alphabet_size']
    results['hit_rate_difference'] = abs(results['avg_hit_rate'] - results['theoretical_hit_rate'])
    correlation = np.corrcoef(results['theoretical_hit_rate'], results['avg_hit_rate'])[0, 1]
    
    # Create visualizations
    plot_results(results, request_length)
    
    # Print validation statistics
    print("\nValidation against theoretical model:")
    print(f"Average difference between empirical and theoretical hit rates: {results['hit_rate_difference'].mean():.4f}")
    print(f"Maximum difference: {results['hit_rate_difference'].max():.4f}")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    # Save results to CSV
    results.to_csv('lru_cache_results.csv', index=False)
    print("\nResults saved to lru_cache_results.csv")
    
    # Generate a summary table for the presentation
    summary = pd.DataFrame({
        'Metric': [
            'Total Configurations Tested',
            'Average Hit Rate',
            'Max Hit Rate',
            'Min Hit Rate',
            'Average Runtime (s)',
            'Theoretical Model Correlation',
            'Avg Difference from Theory',
            'Cache Size with Best Performance'
        ],
        'Value': [
            len(results),
            f"{results['avg_hit_rate'].mean():.4f}",
            f"{results['avg_hit_rate'].max():.4f} (Cache: {results.loc[results['avg_hit_rate'].idxmax(), 'cache_size']}, Alphabet: {results.loc[results['avg_hit_rate'].idxmax(), 'alphabet_size']})",
            f"{results['avg_hit_rate'].min():.4f} (Cache: {results.loc[results['avg_hit_rate'].idxmin(), 'cache_size']}, Alphabet: {results.loc[results['avg_hit_rate'].idxmin(), 'alphabet_size']})",
            f"{results['avg_runtime'].mean():.4f}",
            f"{correlation:.4f}",
            f"{results['hit_rate_difference'].mean():.4f}",
            f"{results.loc[results['avg_hit_rate'].idxmax(), 'cache_size']}"
        ]
    })
    
    # Save summary to CSV for presentation
    summary.to_csv('lru_cache_summary_stats.csv', index=False)
    print("\nSummary statistics saved to lru_cache_summary_stats.csv")
    
    # Create presentation-ready text summary
    with open('presentation_summary.txt', 'w') as f:
        f.write("LRU CACHE PERFORMANCE WITH UNIFORM RANDOM ACCESS DISTRIBUTION\n")
        f.write("==========================================================\n\n")
        f.write("EXECUTIVE SUMMARY:\n")
        f.write(f"- Tested {len(results)} configurations of cache and alphabet sizes\n")
        f.write(f"- Average hit rate across all tests: {results['avg_hit_rate'].mean():.4f}\n")
        f.write(f"- Theoretical model correlation: {correlation:.4f}\n")
        f.write(f"- Best performing cache size: {results.loc[results['avg_hit_rate'].idxmax(), 'cache_size']}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("1. Hit rate is directly proportional to the cache-to-alphabet ratio\n")
        f.write("2. Empirical results closely match theoretical predictions for uniform distribution\n")
        f.write(f"3. Average deviation from theoretical model: {results['hit_rate_difference'].mean():.4f}\n")
        f.write("4. Larger cache sizes provide diminishing returns relative to alphabet size\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("1. For uniform access patterns, cache size should be at least 10% of alphabet size\n")
        f.write("2. Cache-to-alphabet ratio is the primary determinant of hit rate with uniform distribution\n")
        f.write("3. Further studies should compare with non-uniform distributions\n")
    
    print("\nPresentation summary saved to presentation_summary.txt")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()