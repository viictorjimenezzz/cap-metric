from typing import Optional
import torch
from torch import Tensor
import numpy as np
from scipy.stats import pearsonr, spearmanr, ks_2samp

from capmetric.binary import get_pairing_fn, get_energy_fn

def compute_paired_mutual_information(
        prob1: Tensor,
        prob2: Tensor,
        energy_func,
        beta: float = 1.0
    ) -> Tensor:
        """
        Compute MI with paired probabilities.

        Args:
            prob1, prob2: Your original probability vectors [batch_size]
            energy_func: Your energy function (baseline, focal, etc.)
            beta: Inverse temperature
        """
        n = len(prob1)
        y_zeros = torch.zeros(n, dtype=torch.float)
        y_ones = torch.ones(n, dtype=torch.float)
        
        # Compute energies with regularized probabilities
        e1_0 = energy_func(prob1, y_zeros)
        e1_1 = energy_func(prob1, y_ones)
        e2_0 = energy_func(prob2, y_zeros)
        e2_1 = energy_func(prob2, y_ones)
        
        # Standard factorized MI computation
        num = torch.log(
            torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1))
        )
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
            * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
    
        return torch.sum(num - den, dim=0)

def analyze_pairing_quality(tensor1, tensor2, idx1, idx2, method_name):
    """Analyze the quality of a pairing method."""
    paired1 = tensor1[idx1]
    paired2 = tensor2[idx2]
    
    # Convert to numpy for scipy functions
    p1_np = paired1.detach().cpu().numpy()
    p2_np = paired2.detach().cpu().numpy()
    
    # Statistical measures
    pearson_corr, _ = pearsonr(p1_np, p2_np)
    spearman_corr, _ = spearmanr(p1_np, p2_np)
    
    # Mutual information
    energy_fn = get_energy_fn("baseline")
    mi = compute_paired_mutual_information(paired1, paired2, energy_func=energy_fn)
    
    # Distributional similarity (KS test)
    ks_stat, ks_pvalue = ks_2samp(p1_np, p2_np)
    
    # Basic statistics
    stats = {
        'method': method_name,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'mutual_info': mi,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'mean_diff': abs(paired1.mean() - paired2.mean()).item(),
        'std_diff': abs(paired1.std() - paired2.std()).item(),
        'paired1_range': (paired1.min().item(), paired1.max().item()),
        'paired2_range': (paired2.min().item(), paired2.max().item()),
    }
    
    return stats, paired1, paired2


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE PAIRING STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate test data: same distribution, different sizes
    n_large = 1000
    n_small = 300
    
    # Both from same normal distribution
    mean, std = 2.5, 1.2
    mean_shifted = mean + 3
    
    # Generate data: try shifting the large and then the small
    losses_large = torch.normal(mean + mean_shifted, std, (n_large,))
    losses_small = torch.normal(mean, std, (n_small,))
    
    print(f"Dataset sizes: Large={n_large}, Small={n_small}")
    print(f"True distribution: N({mean}, {std}²)")
    print(f"Large dataset stats: mean={losses_large.mean():.4f}, std={losses_large.std():.4f}")
    print(f"Small dataset stats: mean={losses_small.mean():.4f}, std={losses_small.std():.4f}")
    print()
    
    # Test all pairing methods
    pairing_methods = ["none", "random", "absolute", "cdf"]
    
    print("PAIRING METHODS COMPARISON:")
    print("-" * 95)
    print(f"{'Method':<12} {'Pearson':<10} {'Spearman':<10} {'MI':<10} {'KS-stat':<10} {'KS-pval':<10} {'Mean-diff':<10} {'Std-diff':<10}")
    print("-" * 95)
    
    results = []
    
    for method in pairing_methods:
        pairing_fn = get_pairing_fn(method)
        idx1, idx2 = pairing_fn(losses_small, losses_large)
        
        # Analyze pairing quality
        stats, paired1, paired2 = analyze_pairing_quality(
            losses_small, losses_large, idx1, idx2, method
        )
        
        # Print summary
        print(f"{method:<12} {stats['pearson_corr']:<10.4f} {stats['spearman_corr']:<10.4f} "
              f"{stats['mutual_info']:<10.4f} {stats['ks_statistic']:<10.4f} "
              f"{stats['ks_pvalue']:<10.4f} {stats['mean_diff']:<10.4f} {stats['std_diff']:<10.4f}")
        
        results.append({
            'stats': stats,
            'paired1': paired1,
            'paired2': paired2,
            'indices': (idx1, idx2)
        })
    
    print("-" * 95)
    print()
    
    # Detailed analysis of each method
    print("DETAILED ANALYSIS OF EACH METHOD:")
    print("-" * 50)
    
    for i, method in enumerate(pairing_methods):
        result = results[i]
        stats = result['stats']
        paired1 = result['paired1']
        paired2 = result['paired2']
        idx1, idx2 = result['indices']
        
        print(f"\n{method.upper()} METHOD:")

        print(f"  Paired Data Statistics:")
        print(f"    Dataset 1 range: [{stats['paired1_range'][0]:.3f}, {stats['paired1_range'][1]:.3f}]")
        print(f"    Dataset 2 range: [{stats['paired2_range'][0]:.3f}, {stats['paired2_range'][1]:.3f}]")
        print(f"    Dataset 1 mean: {paired1.mean():.4f} ± {paired1.std():.4f}")
        print(f"    Dataset 2 mean: {paired2.mean():.4f} ± {paired2.std():.4f}")
        
        # Analyze pairing patterns
        if method == "none":
            print(f"  Pairing Pattern: First {len(idx1)} elements from both datasets")
        elif method == "random":
            print(f"  Pairing Pattern: Random selection from large dataset")
        elif method == "absolute":
            print(f"  Pairing Pattern: Value-based matching")
            print(f"  Example pairings (small->large):")
            for j in range(min(5, len(paired1))):
                print(f"    {paired1[j]:.3f} -> {paired2[j]:.3f}")
        elif method == "cdf":
            print(f"  Pairing Pattern: Percentile-based matching")

            print(f"  Example pairings (small->large):")
            for j in range(min(5, len(paired1))):
                print(f"    {paired1[j]:.3f} -> {paired2[j]:.3f}")

            # Show percentile preservation
            print(f"  Percentile preservation check:")
            small_ranks = torch.argsort(torch.argsort(paired1)).float()
            large_ranks = torch.argsort(torch.argsort(paired2)).float()
            rank_corr = torch.corrcoef(torch.stack([small_ranks, large_ranks]))[0, 1]
            print(f"    Rank correlation: {rank_corr:.4f}")