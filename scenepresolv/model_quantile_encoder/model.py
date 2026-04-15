import torch
from torch import nn

from spectf.model import SpecTfEncoder


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
        nn.init.normal_(self.readout, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=nheads,
            batch_first=True
        )
        self.token_norm = nn.LayerNorm(hidden)
        self.out_norm = nn.LayerNorm(hidden)

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

        tokens = tokens + wl_enc
        tokens = tokens * (self.hidden ** 0.5)
        tokens = self.token_norm(tokens)
        
        # Cross-attend
        query = self.readout.expand(batch * samples, -1, -1)
        out, _ = self.cross_attn(query, tokens, tokens)
        # Residual connection
        out = query + out
        out = self.out_norm(out.squeeze(1))
        
        return out.view(batch, samples, self.hidden)


class Model(nn.Module):
    def __init__(
        self,
        banddef,
        hidden=256,
        **kwargs
    ):
        super().__init__()
        # self.attn_encoder = BandAttentionReducer(hidden)
        self.attn_encoder = SpecTfEncoder(
            banddef,
            dim_output=hidden,
            num_heads=8,
            dim_proj=64,
            dim_ff=64,
            dropout=0.1,
            agg='max',
            use_residual=False,
            num_layers=1
        )
        # self.mlp = nn.Sequential(
        #     nn.LayerNorm(hidden),
        #     nn.Linear(hidden, hidden),
        #     nn.GELU(),
        #     nn.Linear(hidden, hidden),
        # )

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

        self.beta_low = nn.Parameter(torch.tensor(1.0))
        self.beta_high = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def bounded_output(x, low=0.04, high=6.0):
        return low + (high - low) * torch.sigmoid(x)
    
    def soft_pool(self, x, q, beta=5.0):
        scores = x.norm(dim=-1, keepdim=True)
        scores = (
            (scores - scores.mean(dim=1, keepdim=True))
            / (scores.std(dim=1, keepdim=True) + 1e-6)
        )
        logits = q * beta * scores
        logits = logits.clamp(-10, 10)
        w = torch.softmax(logits, dim=1)

        return (w * x).sum(dim=1)

    def forward(self, x, wl=[]):
        b, s, n = x.shape
        x = self.attn_encoder(x.view(b * s, n).unsqueeze(-1))
        x = x.view(b, s, self.hidden)
        # x = self.mlp(x)

        # Pooling
        beta_low = torch.relu(self.beta_low) + 1e-3
        beta_high = torch.relu(self.beta_high) + 1e-3

        beta_low = beta_low.clamp(0.5, 20.0)
        beta_high = beta_high.clamp(0.5, 20.0)

        # x_low  = x + self.low_mlp(x)
        x_low = self.soft_pool(x, q=-1, beta=beta_low)

        x_high = x + self.high_mlp(x)
        x_high = self.soft_pool(x, q=1, beta=beta_high)

        # Targets
        low = self.p1_head(x_low)
        delta = self.p2_head(x_high)
        high = low + delta

        return torch.cat([low, high], dim=1)
