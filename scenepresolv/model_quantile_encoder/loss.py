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


def log_loss(
    pred,
    target,
    quantiles=[]
    ):

    loss_p1 = (
        torch.log(pred[:, 0] + 1e-6) - torch.log(target[:, 0] + 1e-6)
    ).pow(2).mean()

    loss_p2 = (
        torch.log(pred[:, 1] + 1e-6) - torch.log(target[:, 1] + 1e-6)
    ).pow(2).mean()

    return loss_p1, loss_p2


def mse_loss(pred, target, quantiles=[]):
    raw_p1 = nn.MSELoss()(pred[:, 0], target[:, 0])
    raw_p2 = nn.MSELoss()(pred[:, 1], target[:, 1])

    # Scale p2 loss to match p1's magnitude
    scale = (raw_p1.detach() / (raw_p2.detach() + 1e-8))
    loss_p2_scaled = scale * raw_p2

    return raw_p1, loss_p2_scaled

