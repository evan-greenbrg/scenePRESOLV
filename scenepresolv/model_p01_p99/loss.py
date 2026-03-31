import torch
from torch import nn


def pinball_loss(pred, target, quantiles):
    if target.dim() == 1:
        target = target.unsqueeze(1)
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device).view(1, -1)
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    
    # Normalize to keep magnitudes consistent
    per_q = loss.mean(dim=0)
    per_q = per_q / q.squeeze()

    return per_q.mean()
