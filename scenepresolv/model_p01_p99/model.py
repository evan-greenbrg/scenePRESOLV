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
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
        self.p2_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

        self.tau_min_raw = nn.Parameter(torch.tensor(0.0))

        self.residual_attn = nn.Linear(hidden, 1)

    @staticmethod
    def bounded_output(x, low=0.04, high=6.0):
        return low + (high - low) * torch.sigmoid(x)

    def forward(self, x):
        print(x.shape)
        x = self.mlp(x)
        print(x.shape)

        tau_min = nn.Softplus()(self.tau_min_raw) + 1e-2

        # Low aggregation
        x_std = x.std(dim=1, keepdim=True).detach() + 1e-5
        tau_min = nn.Softplus()(self.tau_min_raw) + 1e-2
        w_min = torch.softmax(-x / (tau_min * x_std), dim=1)
        x_min = (w_min * x).sum(dim=1)

        # High aggregation
        x_mean = x.mean(dim=1)
        scores = self.residual_attn(x).squeeze(-1)
        x_attn = (torch.softmax(scores, dim=1).unsqueeze(-1) * x).sum(dim=1)

        # Targets
        low  = self.bounded_output(self.p1_head(x_min), 0.0, 6.0)
        delta = nn.Softplus()(self.p2_head(x_mean + x_attn))
        high = low.detach() + delta

        return torch.cat([low, high], dim=1)
