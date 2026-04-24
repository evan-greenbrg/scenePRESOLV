import torch
from torch import nn

from spectf.model import SpecTfEncoder


class BandAttentionReducer(nn.Module):
    def __init__(self, hidden=256, nheads=4):
        super().__init__()

        # Hidden shared encoding across wavelength grids
        self.hidden = hidden

        self.input_norm = nn.LayerNorm(1)
        
        # Project each scalar band value to hidden dim
        self.band_proj = nn.Sequential(
            nn.Linear(1, hidden, bias=False),
            nn.GELU()
        )
        
        # Project known wavelength (scalar) to hidden dim
        self.wavelength_proj = nn.Linear(1, hidden, bias=False)
        
        # Cross-attention readout
        self.readout = nn.Parameter(torch.randn(1, 1, hidden))
        # nn.init.normal_(self.readout, std=0.02)
        nn.init.orthogonal_(self.readout[0])

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=nheads,
            batch_first=True
        )
        self.token_norm = nn.LayerNorm(hidden)
        self.out_norm = nn.LayerNorm(hidden)

        self.dropout = nn.Dropout(0.4)

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
        x = self.input_norm(x)
        tokens = self.band_proj(x)
        
        # Normalize to [0, 1] range — keeps inputs well-scaled
        wl_norm = (wl.float() - self.wl_mean) / self.wl_std
        wl_enc = self.wavelength_proj(
            wl_norm.view(-1, 1)
        ).unsqueeze(0)

        tokens = tokens + wl_enc
        tokens = self.token_norm(tokens)
        tokens = self.dropout(tokens)
        
        # Cross-attend
        query = self.readout.expand(batch * samples, -1, -1)
        out, _ = self.cross_attn(query, tokens, tokens)
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
        self.hidden = hidden
        self.attn_encoder = BandAttentionReducer(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )

        self.low_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.mid_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.high_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

        self.beta_low = nn.Parameter(torch.tensor(2.0))
        self.beta_high = nn.Parameter(torch.tensor(2.0))

    def soft_pool(self, x, q, beta):
        scores = x.norm(dim=-1, keepdim=True)

        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True) + 1e-5

        normalized_scores = (scores - mean) / std
        logits = q * beta * normalized_scores
        weights = torch.softmax(logits, dim=1)

        return (weights * x).sum(dim=1)

    def forward(self, x, wl=[]):
        x = self.attn_encoder(x, wl)
        x = self.mlp(x)

        # Pooling
        beta_low  = nn.functional.softplus(self.beta_low) + 1.0
        beta_high = nn.functional.softplus(self.beta_high) + 1.0

        x_low = self.soft_pool(
            x,
            q=-1, beta=beta_low
        )
        x_mid = x.mean(dim=1)
        x_high = self.soft_pool(
            x,
            q=1,  beta=beta_high
        )

        # Targets
        mid = self.mid_head(x_mid)

        low_delta = nn.functional.softplus(self.low_head(x_low))
        high_delta = nn.functional.softplus(self.high_head(x_high))

        mid_anchor = mid.detach()

        low = mid_anchor - low_delta
        high = mid_anchor + high_delta

        return torch.cat([low, mid, high], dim=1)
