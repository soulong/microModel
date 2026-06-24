import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048,
                 bottleneck_dim: int = 256, out_dim: int = 65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
