import torch
from torch import nn


class SimpleAdapter(nn.Module):
    """Lightweight bias-free CLIP Transformer layer adapter (VadCLIP / DSANet)."""

    def __init__(self, c_in: int, c_out: int = 768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SimpleProj(nn.Module):
    """Bias-free CLIP projection adapter preserving angular cone geometry."""

    def __init__(self, c_in: int, c_out: int = 768, relu: bool = True):
        super().__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False),
                nn.LeakyReLU(),
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
