import torch
from torch import nn


class SimpleAdapter(nn.Module):
    """Bottleneck / Residual-ready Adapter for CLIP Transformer layers."""
    def __init__(self, c_in: int, c_out: int = 768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SimpleProj(nn.Module):
    """Linear projection adapter without orthant truncation by default."""
    def __init__(self, c_in: int, c_out: int = 768, relu: bool = False):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_out, bias=False),
                nn.LeakyReLU(negative_slope=0.1, inplace=False)
            )
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
