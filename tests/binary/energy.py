from typing import Optional, Dict, Any
import torch
from torch import Tensor

from capmetric.binary import get_energy_fn

def compute_mutual_information_with_energy(
    prob1: Tensor,
    prob2: Tensor,
    energy_type: str,
    energy_params: Optional[Dict[str, Any]] = None,
    beta: float = 1.0
) -> Tensor:
    """
    Compute mutual information using specified energy function.
    """
    if energy_params is None:
        energy_params = {}
    
    n = len(prob1)
    y_zeros = torch.zeros(n, dtype=torch.float)
    y_ones = torch.ones(n, dtype=torch.float)
    
    # Compute energies
    compute_energy = get_energy_fn(
        energy_type,
        energy_params
    )
    e1_0 = compute_energy(prob1, y_zeros)
    e1_1 = compute_energy(prob1, y_ones)
    e2_0 = compute_energy(prob2, y_zeros)
    e2_1 = compute_energy(prob2, y_ones)
    
    # Factorized MI computation
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
    print("COMPREHENSIVE ENERGY FUNCTIONS ANALYSIS")
    print("="*80)
    
    # Test data setup
    torch.manual_seed(42)  # For reproducibility
    n = 1000
    prob1 = torch.rand(n)
    prob2 = torch.rand(n)
    
    print(f"Dataset size: {n}")
    print(f"prob1 range: [{prob1.min():.4f}, {prob1.max():.4f}]")
    print(f"prob1 mean: {prob1.mean():.4f} ± {prob1.std():.4f}")
    print(f"prob2 range: [{prob2.min():.4f}, {prob2.max():.4f}]")
    print(f"prob2 mean: {prob2.mean():.4f} ± {prob2.std():.4f}")
    print()
    
    # Create test scenarios for energy analysis
    # Test cases: (prob_value, label) combinations
    test_cases = [
        (0.1, 0),  # Low prob, normal (good match)
        (0.1, 1),  # Low prob, anomaly (bad match)  
        (0.5, 0),  # Medium prob, normal
        (0.5, 1),  # Medium prob, anomaly
        (0.9, 0),  # High prob, normal (bad match)
        (0.9, 1),  # High prob, anomaly (good match)
    ]
    
    print("ENERGY FUNCTION BEHAVIOR ON TEST CASES:")
    print("-" * 80)
    print(f"{'Energy Type':<15} {'(0.1,0)':<8} {'(0.1,1)':<8} {'(0.5,0)':<8} {'(0.5,1)':<8} {'(0.9,0)':<8} {'(0.9,1)':<8}")
    print("-" * 80)
    
    # Energy function configurations
    energy_configs = [
        ("baseline", {}),
        ("focal", {"gamma": 2.0}),
        ("focal", {"gamma": 0.5}),
        ("exponential", {"gamma": 2.0}),
        ("exponential", {"gamma": 0.5}),
        ("margin", {"margin": 0.1, "threshold": 0.5}),
        ("margin", {"margin": 0.3, "threshold": 0.5}),
        ("contrastive", {"center": 0.7, "margin": 0.1}),
        ("adaptive", {"mean": 0.5, "std": 0.2, "scale": 0.5})
    ]
    
    energy_results = []
    
    for energy_type, params in energy_configs:
        # Compute energies for test cases
        test_energies = []
        for prob_val, label in test_cases:
            prob_tensor = torch.tensor([prob_val])
            y_tensor = torch.tensor([float(label)])

            compute_energy = get_energy_fn(
                energy_type,
                params
            )
            energy_val = compute_energy(prob_tensor, y_tensor)
            test_energies.append(energy_val.item())
        
        # Format method name with key parameters
        if energy_type == "focal" and "gamma" in params:
            method_name = f"focal(γ={params['gamma']})"
        elif energy_type == "exponential" and "gamma" in params:
            method_name = f"exp(γ={params['gamma']})"
        elif energy_type == "margin" and "margin" in params:
            method_name = f"margin(m={params['margin']})"
        else:
            method_name = energy_type
        
        # Print test case results
        energy_str = " ".join([f"{e:<8.3f}" for e in test_energies])
        print(f"{method_name:<15} {energy_str}")
        
        # Compute mutual information
        mi = compute_mutual_information_with_energy(prob1, prob2, energy_type, params)
        
        energy_results.append({
            'method': method_name,
            'type': energy_type,
            'params': params,
            'mi': mi.item(),
            'test_energies': test_energies
        })
    
    print("-" * 80)
    print()
    
    # Mutual information comparison
    print("MUTUAL INFORMATION COMPARISON:")
    print("-" * 60)
    print(f"{'Energy Type':<20} {'MI Value':<12} {'Good Match Avg':<15} {'Bad Match Avg':<15}")
    print("-" * 60)
    
    for result in energy_results:
        # Calculate average energy for good vs bad matches
        # Good matches: (0.1,0), (0.9,1) - prob matches expectation
        # Bad matches: (0.1,1), (0.9,0) - prob contradicts expectation
        good_match_energies = [result['test_energies'][0], result['test_energies'][5]]  # (0.1,0), (0.9,1)
        bad_match_energies = [result['test_energies'][1], result['test_energies'][4]]   # (0.1,1), (0.9,0)
        
        good_avg = sum(good_match_energies) / len(good_match_energies)
        bad_avg = sum(bad_match_energies) / len(bad_match_energies)
        
        print(f"{result['method']:<20} {result['mi']:<12.6f} {good_avg:<15.3f} {bad_avg:<15.3f}")
    
    print("-" * 60)
    print()
    
    # Detailed analysis of key energy functions
    print("DETAILED ANALYSIS OF KEY ENERGY FUNCTIONS:")
    print("-" * 60)
    
    key_methods = ["baseline", "focal(γ=2.0)", "exp(γ=2.0)", "contrastive"]
    
    for method_name in key_methods:
        result = next((r for r in energy_results if r['method'] == method_name), None)
        if result is None:
            continue
        
        print(f"\n{method_name.upper()}:")
        print(f"  Mutual Information: {result['mi']:.6f}")
        print(f"  Energy behavior:")
        
        test_descriptions = [
            "Low prob, Normal (0.1, 0)",     # Should be low energy (good)
            "Low prob, Anomaly (0.1, 1)",   # Should be high energy (bad)
            "Med prob, Normal (0.5, 0)",    # Medium energy
            "Med prob, Anomaly (0.5, 1)",   # Medium energy  
            "High prob, Normal (0.9, 0)",   # Should be high energy (bad)
            "High prob, Anomaly (0.9, 1)"   # Should be low energy (good)
        ]
        
        for i, (desc, energy_val) in enumerate(zip(test_descriptions, result['test_energies'])):
            classification = "GOOD" if i in [0, 5] else "BAD" if i in [1, 4] else "MEDIUM"
            print(f"    {desc:<25}: {energy_val:>8.3f} ({classification})")
        
        # Energy discrimination analysis
        good_energies = [result['test_energies'][0], result['test_energies'][5]]
        bad_energies = [result['test_energies'][1], result['test_energies'][4]]
        discrimination = (sum(bad_energies) - sum(good_energies)) / 2  # Average difference
        
        print(f"  Discrimination power: {discrimination:.3f} (higher = better separation)")
        
        # Analyze special properties
        if "focal" in method_name:
            print(f"  Focus effect: Emphasizes hard examples (low confidence predictions)")
        elif "exp" in method_name:
            print(f"  Exponential penalty: Heavy punishment for misalignments")
        elif "contrastive" in method_name:
            print(f"  Class separation: Promotes clustering around centers")
        elif method_name == "baseline":
            print(f"  Linear penalty: Simple misalignment cost")