import torch
from torch import nn


class BandAttentionReducer(nn.Module):
    def __init__(self, hidden=256, nheads=4):
        super().__init__()

        # Hidden shared encoding across wavelength grids
        self.hidden = hidden
        
        # Project each scalar band value to hidden dim
        self.band_proj = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU()
        )
        # nn.init.normal_(self.band_proj.Linear.weight, std=0.02)
        # nn.init.zeros_(self.band_proj.Linear.bias)
        
        # Project known wavelength (scalar) to hidden dim
        self.wavelength_proj = nn.Linear(1, hidden)
        
        # Cross-attention readout
        self.readout = nn.Parameter(torch.randn(1, 1, hidden))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=nheads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)

        self.wl_mean = 1440
        self.wl_std = 600

    def forward(self, x, wl):
        """
        x:           (s, n, b)
        wavelengths: (b,) in nm e.g. tensor([443, 490, 560, ...])
        """
        batch, samples, bands = x.shape
        
        # Project band values
        x = x.view(batch * samples, bands, 1)
        tokens = self.band_proj(x)
        
        # Normalize to [0, 1] range — keeps inputs well-scaled
        wl_norm = (wl.float() - self.wl_mean) / self.wl_std
        wl_enc = self.wavelength_proj(
            wl_norm.view(-1, 1)
        ).unsqueeze(0)
        tokens= tokens + wl_enc
        
        # Cross-attend
        query = self.readout.expand(batch * samples, -1, -1)
        out, _ = self.cross_attn(query, tokens, tokens)
        out = self.norm(out.squeeze(1))
        
        return out.view(batch, samples, self.hidden)


class Model(nn.Module):
    def __init__(
        self,
        b,
        hidden=256,
        **kwargs
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.attn_encoder = BandAttentionReducer(hidden)

        # self.low_mlp = nn.Sequential(
        #     nn.LayerNorm(hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.GELU()
        # )

        # self.high_mlp = nn.Sequential(
        #     nn.LayerNorm(hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.GELU()
        # )

        self.p1_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )
        self.p2_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )

        self.q_queries = nn.Parameter(torch.randn(2, hidden))
        self.hidden = hidden


    @staticmethod
    def bounded_output(x, low=0.04, high=6.0):
        return low + (high - low) * torch.sigmoid(x)
    
    def soft_pool(self, x, q, beta=2.0):
        scores = x.norm(dim=-1, keepdim=True)
        w = torch.softmax((q * beta) * scores, dim=1)
        return (w * x).sum(dim=1)

    def quantile_pool(self, x):
        # x: (batch, n, hidden)
        q = self.q_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        scores = torch.bmm(q, x.transpose(1, 2)) / (self.hidden ** 0.5)
        w = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(w, x)
        return pooled[:, 0], pooled[:, 1]

    def forward(self, x, wl=[]):
        x = self.attn_encoder(x, wl)
        x = self.mlp(x)

        # x_mean = x.mean(dim=1, keepdim=True)
        # x = x + 0.5 * x_mean

        # x_low  = x + self.low_mlp(x)
        # x_high = x + self.high_mlp(x)

        x_min, x_max = self.quantile_pool(x)

        # Low aggregation
        # x_min = self.soft_pool(x_low, -.99)
        # # High aggregation
        # x_max = self.soft_pool(x_high, .99)

        # Targets
        # low  = self.bounded_output(self.p1_head(x_min), 0.0, 7.0)
        # high = self.bounded_output(self.p2_head(x_max), 0.0, 7.0)
        low  = self.p1_head(x_min)
        high = self.p2_head(x_max)

        return torch.cat([low, high], dim=1)
