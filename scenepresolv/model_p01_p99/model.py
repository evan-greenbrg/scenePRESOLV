import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        b,
        hidden=256,
        **kwargs
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(b, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden)
        )

        self.p1_head = nn.Sequential(
            # nn.LayerNorm(3 * hidden),
            # nn.Linear(3 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        self.p2_head = nn.Sequential(
            # nn.LayerNorm(3 * hidden),
            # nn.Linear(3 * hidden, hidden),
            # nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )

        self.tau_min_raw = nn.Parameter(torch.tensor(0.0))
        self.tau_max_raw = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def bounded_output(x, low=0.04, high=6.0):
        return low + (high - low) * torch.sigmoid(x)

    def forward(self, x):
        # x = self.pool(self.mlp(x))
        x = self.mlp(x)
        # x = self.base(x)

        tau_min = nn.Softplus()(self.tau_min_raw) + 1e-2
        tau_max = nn.Softplus()(self.tau_max_raw) + 1e-2

        w_min = torch.softmax(-x / tau_min, dim=1)
        w_max = torch.softmax( x / tau_max, dim=1)

        x_min = (w_min * x).sum(dim=1)
        x_mean = x.mean(dim=1)
        x_max = (w_max * x).sum(dim=1)

        low  = self.bounded_output(self.p1_head(x_mean - x_min), 0.0, 6.0)
        delta = self.bounded_output(self.p2_head(x_max - x_mean), 0.0, 6.0)
        high = low + delta

        return torch.cat([low, high], dim=1)
