import torch
import numpy as np
from torch import nn
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)

from scenepresolv.model_quantile_encoder.loss import pinball_loss, mse_loss


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


def quantile_mae(pred, target):
    mae_low = (pred[:, 0] - target[:, 0]).mean()
    mae_high = (pred[:, 1] - target[:, 1]).mean()

    return mae_low.item(), mae_high.item()


def attn_similarity(model, x, wl):
    """
    If similarity is >0.9 or norm std is near zero,
    the encoder is collapsing all samples to similar representations 
    regardless of their spectral content.
    """
    with torch.no_grad():
        emb = model.attn_encoder(x, wl)
        sim = nn.functional.cosine_similarity(
            emb[:, :, None], emb[:, None, :], dim=-1
        )
        return sim.mean().item(), emb.norm(dim=-1).std().item()


def print_beta(model, x, wl):
    return nn.Softplus()(model.beta_low).item(), nn.Softplus()(model.beta_high).item()


def evaluation(dataloader, model, device, epoch, weight_pinball):
    quantiles = [0.25, 0.75]
    all_pred = []
    all_target = []
    similarities = []
    emb_norms = []
    low_betas = []
    high_betas = []
    wl = torch.tensor(dataloader.dataset.wl).type(torch.float32).to(device)
    with torch.no_grad():
        for batch in dataloader:
            x = batch['toa'].to(device)
            target = batch['atmosphere'].to(device)
            pred = model(x, wl)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())

            similarity, emb_norm = attn_similarity(model, x, wl)
            similarities.append(similarity)
            emb_norms.append(emb_norm)

            low_beta, high_beta = print_beta(model, x, wl)
            low_betas.append(low_beta)
            high_betas.append(high_beta)
    
    pred = torch.cat(all_pred)
    target = torch.cat(all_target)
    
    mse_loss_low, mse_loss_hi = mse_loss(pred, target, quantiles=quantiles)
    mse_loss_total = mse_loss_low + (2 * mse_loss_hi)
    pinball_loss_low, pinball_loss_hi = pinball_loss(pred, target, quantiles=quantiles)
    pinball_loss_total = pinball_loss_low + (2 * pinball_loss_hi)
    loss = (1 - weight_pinball) * mse_loss_total + weight_pinball * pinball_loss_total

    loss_low = (1 - weight_pinball) * mse_loss_low + weight_pinball * pinball_loss_low
    loss_high = (1 - weight_pinball) * mse_loss_hi + weight_pinball * pinball_loss_hi

    low_cov, high_cov = quantile_coverage(pred, target)
    mape_low, mape_high = quantile_mape(pred, target)
    mae_low, mae_high = quantile_mae(pred, target)
    width = interval_width(pred)
    
    return {
        'loss_low': loss_low.item(),
        'loss_high': loss_high.item(),
        'loss_total': loss.item(),
        'coverage_low': low_cov,
        'coverage_high': high_cov,
        'mape_low': mape_low,
        'mape_high': mape_high,
        'mae_low': mae_low,
        'mae_high': mae_high,
        'interval_width': width,
        'similarity': np.mean(similarities),
        'attention_embed_norm': np.mean(emb_norms),
        'low_beta': np.mean(low_betas),
        'high_beta': np.mean(high_betas),
    }
