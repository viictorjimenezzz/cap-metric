import torch
from capmetric.binary import get_normalizer_fn


if __name__ == "__main__":
    print("TENSOR NORMALIZATION COMPARISON")
    print("="*50)
    
    # Test data
    torch.manual_seed(42)
    n = 1000
    
    # Different distribution tensors
    tensor1 = torch.randn(n) * 2.0 + 5.0  # Normal: mean=5, std=2
    tensor2 = torch.exp(torch.randn(n) * 0.5) * 2.0  # Exponential-like
    
    print(f"Original tensor1: range=[{tensor1.min():.2f}, {tensor1.max():.2f}], mean={tensor1.mean():.2f}")
    print(f"Original tensor2: range=[{tensor2.min():.2f}, {tensor2.max():.2f}], mean={tensor2.mean():.2f}")
    print()
    
    # Test all combinations
    methods = ["minmax", "softmax", "sigmoid"]
    modes = ["split", "joint"]
    
    print(f"{'Method':<10} {'Mode':<6} {'T1 Range':<15} {'T2 Range':<15} {'Preserves Scale':<15}")
    print("-" * 70)
    
    for method in methods:
        for mode in modes:
            params = {"mode": mode}
            normalize_tensors = get_normalizer_fn(method, params)
            norm1, norm2 = normalize_tensors(tensor1, tensor2)
            
            t1_range = f"[{norm1.min():.3f}, {norm1.max():.3f}]"
            t2_range = f"[{norm2.min():.3f}, {norm2.max():.3f}]"
            
            # Check if relative scale is preserved (only meaningful for joint mode)
            if mode == "joint":
                orig_ratio = tensor1.max().item() / tensor2.max().item()
                norm_ratio = norm1.max().item() / norm2.max().item()
                preserves = f"{norm_ratio/orig_ratio:.2f}" if orig_ratio != 0 else "N/A"
            else:
                preserves = "No (split)"
            
            print(f"{method:<10} {mode:<6} {t1_range:<15} {t2_range:<15} {preserves:<15}")
    
    print("-" * 70)
    print()
    
    # Detailed analysis of recommended method
    print("RECOMMENDED: MIN-MAX JOINT NORMALIZATION")
    print("-" * 40)
    
    params.update({"mode": "joint"})
    normalize_tensors = get_normalizer_fn("minmax", params)
    norm1, norm2 = normalize_tensors(tensor1, tensor2)
    
    print(f"Tensor1 normalized: range=[{norm1.min():.4f}, {norm1.max():.4f}], mean={norm1.mean():.4f}")
    print(f"Tensor2 normalized: range=[{norm2.min():.4f}, {norm2.max():.4f}], mean={norm2.mean():.4f}")
    
    # Show distribution preservation
    print("\nCORRELATION CHECK:")
    print(f"Tensor1 - Original vs Normalized correlation: {torch.corrcoef(torch.stack([tensor1, norm1]))[0,1]:.4f}")
    print(f"Tensor2 - Original vs Normalized correlation: {torch.corrcoef(torch.stack([tensor2, norm2]))[0,1]:.4f}")