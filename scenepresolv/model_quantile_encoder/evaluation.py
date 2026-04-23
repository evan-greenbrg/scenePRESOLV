import torch
import numpy as np
from torch import nn
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)

from scenepresolv.model_quantile_encoder.loss import pinball_loss, mse_loss


def quantile_coverage(pred, target):
    low_coverage  = (target < pred[:, 0][:, None]).float().mean()
    mid_coverage  = (target < pred[:, 1][:, None]).float().mean()
    high_coverage = (target < pred[:, 2][:, None]).float().mean()
    return low_coverage.item(), mid_coverage.item(), high_coverage.item()


def interval_width(pred):
    return (pred[:, 2] - pred[:, 0]).mean().item()


def quantile_mae(pred, target, quantiles=[0.05, 0.5, 0.95]):
    qs = [torch.quantile(target, q, axis=1) for q in quantiles]
    maes = [pred[:, i] - qs[i] for i in range(len(quantiles))]
    maes = [m.mean() for m in maes]

    mae_low, mae_mid, mae_high = [m.item() for m in maes]

    return mae_low, mae_mid, mae_high


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


def evaluation(dataloader, model, wl, device, epoch, quantiles=[0.05, 0.5, 0.95]):
    all_pred = []
    all_target = []
    similarities = []
    emb_norms = []
    low_betas = []
    high_betas = []
    wl = torch.tensor(wl).type(torch.float32).to(device)
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
    
    pinball_loss_low, pinball_loss_mid, pinball_loss_hi = pinball_loss(
        pred, target, quantiles=quantiles
    )
    pinball_loss_total = pinball_loss_low + pinball_loss_mid + pinball_loss_hi

    loss = pinball_loss_total
    loss_low = pinball_loss_low
    loss_mid = pinball_loss_mid
    loss_high = pinball_loss_hi

    low_cov, mid_cov, high_cov = quantile_coverage(pred, target)
    mae_low, mae_mid, mae_high = quantile_mae(pred, target, quantiles)
    width = interval_width(pred)
    
    return {
        'loss_low': loss_low.item(),
        'loss_mid': loss_mid.item(),
        'loss_high': loss_high.item(),
        'loss_total': loss.item(),
        'coverage_low': low_cov,
        'coverage_mid': mid_cov,
        'coverage_high': high_cov,
        'mae_low': mae_low,
        'mae_mid': mae_mid,
        'mae_high': mae_high,
        'interval_width': width,
        'similarity': np.mean(similarities),
        'attention_embed_norm': np.mean(emb_norms),
        'low_beta': np.mean(low_betas),
        'high_beta': np.mean(high_betas),
    }
