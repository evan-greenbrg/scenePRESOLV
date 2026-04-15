import torch
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)


def quantile_coverage(pred, target, quantiles=[0.01, 0.99]):
    low_coverage  = (target[:, 0] < pred[:, 0]).float().mean()
    high_coverage = (target[:, 1] > pred[:, 1]).float().mean()
    return low_coverage.item(), high_coverage.item()


def interval_width(pred):
    return (pred[:, 1] - pred[:, 0]).mean().item()


def quantile_mape(pred, target):
    mape_low = (
        (pred[:, 0] - target[:, 0]).abs() / (target[:, 0].abs() + 1e-6)
    ).mean()
    mape_high = (
        (pred[:, 1] - target[:, 1]).abs() / (target[:, 1].abs() + 1e-6)
    ).mean()

    return mape_low.item(), mape_high.item()


def evaluation(dataloader, model, device, loss_fn):
    all_pred = []
    all_target = []
    wl = torch.tensor(dataloader.dataset.wl).type(torch.float32).to(device)
    with torch.no_grad():
        for batch in dataloader:
            x = batch['toa'].to(device)
            target = batch['atmosphere'].to(device)
            pred = model(x, wl)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
    
    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    
    loss_low, loss_high = loss_fn(pred, target)
    low_cov, high_cov = quantile_coverage(pred, target)
    mape_low, mape_high = quantile_mape(pred, target)
    width = interval_width(pred)
    
    return {
        'loss_low': loss_low.item(),
        'loss_high': loss_high.item(),
        'coverage_low': low_cov,
        'coverage_high': high_cov,
        'mape_low': mape_low,
        'mape_high': mape_high,
        'interval_width': width,
    }
