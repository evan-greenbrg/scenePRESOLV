import numpy as np
import torch
from torch import nn


def pinball_loss(
    pred,
    target,
    quantiles=[]
):
    if target.dim() == 1:
        target = target.unsqueeze(1)
    
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device).view(1, -1)
    loss = torch.where(err >= 0, q * err, (q - 1) * err)
    
    loss_low  = loss[:, 0].mean()
    loss_high = loss[:, 1].mean()
    
    return loss_low, loss_high


def mape_loss(
    pred,
    target,
    quantiles=[]
):
    
    loss_p1 = (
        (pred[:, 0] - target[:, 0]) / target[:, 0]
    ).pow(2).mean()

    loss_p2 = (
        (pred[:, 1] - target[:, 1]) / target[:, 1]
    ).pow(2).mean()
    
    return loss_p1, loss_p2
