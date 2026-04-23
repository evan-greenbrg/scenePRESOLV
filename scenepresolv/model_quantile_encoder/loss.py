import numpy as np
import torch
from torch import nn


def pinball_loss(
    pred,
    target,
    quantiles=[0.05, 0.5, 0.95]
):
    print(pred.shape)
    print(target.shape)
    if target.dim() == 1:
        target = target.unsqueeze(1)
    
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device).view(1, -1)
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    
    loss_low  = loss[:, 0].mean()
    loss_mid = loss[:, 1].mean()
    loss_high = loss[:, 2].mean()
    
    return loss_low, loss_mid, loss_high 


def mse_loss(pred, target, quantiles=[]):
    raw_low = nn.MSELoss()(pred[:, 0], target[:, 0])
    raw_mid = nn.MSELoss()(pred[:, 1], target[:, 1])
    raw_high = nn.MSELoss()(pred[:, 2], target[:, 2])

    return raw_low, raw_mid, raw_high 

