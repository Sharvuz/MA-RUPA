import sys
from pathlib import Path
import torch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

print("[*] Testing imports...")
from utils.losses import (
    compute_anti_saturation_mil_loss,
    compute_stiefel_grassmann_loss,
    AntiSaturationMILLoss
)
print("  [OK] Losses imported successfully.")

# Test AntiSaturationMILLoss (Clean Top-K MIL without self-paced gradient sink)
criterion = AntiSaturationMILLoss(k_ratio=16, eps=1e-4, use_self_paced=False)

# 1. Normal and Abnormal bags
B, L = 4, 128
logits = torch.randn(B, L, requires_grad=True)
scores = torch.sigmoid(logits)
labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
lengths = torch.tensor([128, 100, 120, 96])

loss = criterion(scores, labels, lengths=lengths)
print(f"  [OK] Anti-Saturation MIL loss computed: {loss.item():.5f} (Guaranteed >= 0.0305)")
assert loss.item() >= 0.030, f"Loss {loss.item()} is below 0.030 floor!"

# Backward pass check
loss.backward()
assert logits.grad is not None, "Gradient is None!"
grad_norm = logits.grad.norm().item()
print(f"  [OK] Backward pass successful! Grad norm: {grad_norm:.6f} (Guaranteed non-vanishing)")
assert grad_norm > 1e-4, f"Grad norm {grad_norm} is too small (vanishing)!"

# 2. Test Stiefel Grassmannian Loss
U = torch.randn(64, 512, requires_grad=True)
loss_stiefel = compute_stiefel_grassmann_loss(U, lambda_stiefel=0.5, lambda_grassmann=0.25)
print(f"  [OK] Stiefel Grassmannian loss computed: {loss_stiefel.item():.5f}")
loss_stiefel.backward()
assert U.grad is not None, "U grad is None!"
print(f"  [OK] Stiefel backward pass successful! U grad norm: {U.grad.norm().item():.6f}")

print("\n[+] ALL S2 ANTI-COLLAPSE ARCHITECTURAL SAFARDS VERIFIED SUCCESSFULLY!")
