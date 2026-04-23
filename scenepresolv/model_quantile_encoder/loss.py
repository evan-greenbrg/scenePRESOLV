import numpy as np
import torch
from torch import nn


def pinball_loss(
    pred,
    target,
    quantiles=[0.05, 0.5, 0.95]
):
    t = target.unsqueeze(2) 
    p = pred.unsqueeze(1)
    print(t.shape)
    print(p.shape)

    err = t - p
    q = torch.tensor(quantiles, device=pred.device).view(1, 1, -1)
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    loss_per_quantile = loss.mean(dim=(0, 1))

    return loss_per_quantile[0], loss_per_quantile[1], loss_per_quantile[2]


def mse_loss(pred, target, quantiles=[]):
    raw_low = nn.MSELoss()(pred[:, 0], target[:, 0])
    raw_mid = nn.MSELoss()(pred[:, 1], target[:, 1])
    raw_high = nn.MSELoss()(pred[:, 2], target[:, 2])

    return raw_low, raw_mid, raw_high 

