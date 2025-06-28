from typing import Optional
import torch
from torch import Tensor

from capmetric.binary import get_regularizer_fn


def compute_constrained_mutual_information(
        prob1: Tensor,
        prob2: Tensor,
        energy_func,  # Your energy function
        regularization_type: str = "threshold_max",
        regularization_params: Optional[dict] = None,
        beta: float = 1.0
    ) -> Tensor:
        """
        Compute MI with implicit constraint enforcement via probability regularization.
        
        This approach modifies the input probabilities to naturally guide the energy function
        toward configurations with exactly maxcount positive classifications.
        
        Args:
            prob1, prob2: Your original probability vectors [batch_size]
            energy_func: Your energy function (baseline, focal, etc.)
            maxcount: Maximum number of positive classifications
            regularization_type: Type of probability regularization
            regularization_params: Parameters for regularization
            beta: Inverse temperature
            
        Returns:
            Mutual information with implicit constraint enforcement
        """
        # Regularize probabilities to implicitly enforce constraint
        regularize_prob = get_regularizer_fn(
            regularization_type,
            regularization_params
        )
        reg_prob1 = regularize_prob(prob1)
        reg_prob2 = regularize_prob(prob2)
        
        n = len(prob1)
        y_zeros = torch.zeros(n, dtype=torch.float)
        y_ones = torch.ones(n, dtype=torch.float)
        
        # Compute energies with regularized probabilities
        e1_0 = energy_func(reg_prob1, y_zeros)
        e1_1 = energy_func(reg_prob1, y_ones)
        e2_0 = energy_func(reg_prob2, y_zeros)
        e2_1 = energy_func(reg_prob2, y_ones)
        
        # Standard factorized MI computation
        num = torch.log(
            torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1))
        )
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
            * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
    
        return torch.sum(num - den, dim=0)


if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE PROBABILITY REGULARIZATION ANALYSIS")
    print("="*80)
    
    # Test data setup
    torch.manual_seed(42)  # For reproducibility
    n = 200
    max_counts = 20
    prob1 = torch.rand(n)
    prob2 = torch.rand(n)
    
    # Simple baseline energy for testing
    def baseline_energy(prob, y):
        return y * (1 - prob) + (1 - y) * prob
    
    print(f"Dataset size: {n}")
    print(f"Maximum positive classifications: {max_counts}")
    print(f"Original prob1 range: [{prob1.min():.4f}, {prob1.max():.4f}]")
    print(f"Original prob1 mean: {prob1.mean():.4f} ± {prob1.std():.4f}")
    print()
    
    # Identify original top-k and bottom-(n-k) elements
    top_k_orig, top_k_indices = torch.topk(prob1, max_counts, largest=True)
    bottom_orig, bottom_indices = torch.topk(prob1, n - max_counts, largest=False)
    
    print("ORIGINAL DISTRIBUTION ANALYSIS:")
    print(f"Top-{max_counts} values: min={top_k_orig.min():.4f}, max={top_k_orig.max():.4f}, mean={top_k_orig.mean():.4f}")
    print(f"Bottom-{n-max_counts} values: min={bottom_orig.min():.4f}, max={bottom_orig.max():.4f}, mean={bottom_orig.mean():.4f}")
    print(f"Separation gap: {top_k_orig.min():.4f} - {bottom_orig.max():.4f} = {(top_k_orig.min() - bottom_orig.max()):.4f}")
    print()
    
    # Test all regularization methods
    regularization_configs = [
        ("none", {}),
        ("threshold_max", {}),
        ("threshold_mean", {}), 
        ("threshold_zero", {}),
        ("smooth_sigmoid", {"temperature": 0.1}),
        ("smooth_sigmoid", {"temperature": 1.0}),
        ("smooth_exponential", {"temperature": 0.5}),
        ("percentile", {"suppression_percentile": 0.05, "enhancement_percentile": 0.95})
    ]
    
    print("REGULARIZATION METHODS COMPARISON:")
    print("-" * 80)
    print(f"{'Method':<25} {'MI':<12} {'Top Mean':<10} {'Bot Mean':<10} {'Separation':<12} {'Suppression':<12}")
    print("-" * 80)
    
    results = []
    
    for reg_type, params in regularization_configs:
        params.update({"maxcount": max_counts})
        
        # Apply regularization
        regularize_prob = get_regularizer_fn(
            reg_type,
            params
        )
        reg_prob1 = regularize_prob(prob1)
        
        # Compute MI
        mi = compute_constrained_mutual_information(
            prob1, prob2, baseline_energy, reg_type, params
        )
        
        # Analyze regularization effects
        top_k_reg = reg_prob1[top_k_indices]
        bottom_reg = reg_prob1[bottom_indices]
        
        top_mean = top_k_reg.mean().item()
        bottom_mean = bottom_reg.mean().item()
        separation = top_k_reg.min().item() - bottom_reg.max().item()
        
        # Calculate suppression ratio
        if reg_type != "none":
            suppression_ratio = (bottom_reg.mean() / bottom_orig.mean()).item()
        else:
            suppression_ratio = 1.0
        
        # Format method name with params
        if params:
            method_name = f"{reg_type}({list(params.values())[0]:.1f})"
        else:
            method_name = reg_type
        
        print(f"{method_name:<25} {mi.item():<12.6f} {top_mean:<10.4f} {bottom_mean:<10.4f} {separation:<12.4f} {suppression_ratio:<12.3f}")
        
        results.append({
            'method': method_name,
            'mi': mi.item(),
            'top_mean': top_mean,
            'bottom_mean': bottom_mean,
            'separation': separation,
            'suppression': suppression_ratio,
            'reg_prob1': reg_prob1
        })
    
    print("-" * 80)
    print()
    
    # Detailed analysis of selected methods
    print("DETAILED ANALYSIS OF KEY METHODS:")
    print("-" * 50)
    
    key_methods = ["threshold_zero", "threshold_max", "adaptive(2.0)", "smooth_sigmoid(0.1)"]
    
    for method_name in key_methods:
        result = next((r for r in results if r['method'] == method_name), None)
        if result is None:
            continue
            
        reg_prob1 = result['reg_prob1']
        
        print(f"\n{method_name.upper()}:")
        print(f"  Mutual Information: {result['mi']:.6f}")
        print(f"  Top-{max_counts} probabilities:")
        print(f"    Range: [{reg_prob1[top_k_indices].min():.4f}, {reg_prob1[top_k_indices].max():.4f}]")
        print(f"    Mean: {reg_prob1[top_k_indices].mean():.4f}")
        print(f"    Sample: {reg_prob1[top_k_indices][:5].tolist()}")
        print(f"  Bottom-{n-max_counts} probabilities:")
        print(f"    Range: [{reg_prob1[bottom_indices].min():.4f}, {reg_prob1[bottom_indices].max():.4f}]") 
        print(f"    Mean: {reg_prob1[bottom_indices].mean():.4f}")
        print(f"    Sample: {reg_prob1[bottom_indices][:5].tolist()}")
        print(f"  Constraint effectiveness:")
        print(f"    Clear separation: {'YES' if result['separation'] > 0 else 'NO'}")
        print(f"    Suppression ratio: {result['suppression']:.3f} ({'Strong' if result['suppression'] < 0.5 else 'Moderate' if result['suppression'] < 0.8 else 'Weak'})")