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

        self.base = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
        )
        self.p1_head = nn.Linear(hidden, 1)
        self.p2_head = nn.Linear(hidden, 1)

    def pool(self, x, tau=0.1):
        w_min = torch.softmax(-x / tau, dim=1)
        w_max = torch.softmax( x / tau, dim=1)
        min_term = torch.sum(w_min * x, dim=1)
        mean_term = torch.mean(x, dim=1)
        max_term = torch.sum(w_max * x, dim=1)
        return torch.cat([min_term, mean_term, max_term], dim=-1)
    

    def forward(self, x):
        x = self.pool(self.mlp(x))
        x = self.base(x)
        low  = nn.Softplus()(
            self.p1_head(x)
        )
        high = nn.Softplus()(
            self.p2_head(x)
        )

        return torch.cat([low, high], dim=1).squeeze(1) 
