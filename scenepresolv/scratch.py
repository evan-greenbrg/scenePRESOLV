import math
from functools import partial
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from spectral import envi
import wandb
from matplotlib import pyplot as plt
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)

from spectf.utils import envi_header
from spectf.utils import get_device

from spectf.utils import seed as useed
from scenepresolv.dataset import ImageDataset
from scenepresolv.model_p01_p99.model import Model as Model_p99
from scenepresolv.model_p01_p99.trainer import Trainer as Trainer_p99
from scenepresolv.model_p01_p99.evaluation import evaluation as evaluation_p99


def gradient_summary(model):
    rows = []
    total_params = 0
    total_with_grad = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        has_grad = param.grad is not None

        if has_grad:
            total_with_grad += param.numel()
            grad = param.grad
            rows.append({
                'name':     name,
                'shape':    tuple(param.shape),
                'mean':     grad.mean().item(),
                'std':      grad.std().item(),
                'max':      grad.abs().max().item(),
                'dead':     (grad.abs() < 1e-7).float().mean().item(),
                'exploding': grad.abs().max().item() > 10.0,
                'vanishing': grad.abs().max().item() < 1e-5,
            })
        else:
            rows.append({
                'name':     name,
                'shape':    tuple(param.shape),
                'mean':     None,
                'std':      None,
                'max':      None,
                'dead':     None,
                'exploding': False,
                'vanishing': False,
            })

    # ── Header ────────────────────────────────────────────────────────────────
    col_w = 42
    print(f"\n{'─'*100}")
    print(f"  GRADIENT SUMMARY")
    print(f"{'─'*100}")
    print(f"  {'LAYER':<{col_w}} {'SHAPE':<18} {'MEAN':>10} {'STD':>10} {'MAX':>10} {'DEAD%':>7}  FLAGS")
    print(f"{'─'*100}")

    for r in rows:
        flags = []
        if r['exploding']: flags.append('EXPLODING')
        if r['vanishing']: flags.append('vanishing')
        flag_str = ' '.join(flags)

        if r['mean'] is None:
            print(f"  {r['name']:<{col_w}} {str(r['shape']):<18} {'no grad':>10}")
        else:
            print(
                f"  {r['name']:<{col_w}} {str(r['shape']):<18}"
                f" {r['mean']:>10.2e} {r['std']:>10.2e}"
                f" {r['max']:>10.2e} {r['dead']*100:>6.1f}%"
                f"  {flag_str}"
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"{'─'*100}")
    exploding = sum(r['exploding'] for r in rows)
    vanishing = sum(r['vanishing'] for r in rows)
    dead_layers = [r['name'] for r in rows if r['dead'] is not None and r['dead'] > 0.5]

    print(f"  Params total: {total_params:,}  |  with grad: {total_with_grad:,}")
    print(f"  Exploding: {exploding}  |  Vanishing: {vanishing}  |  >50% dead: {len(dead_layers)}")
    if dead_layers:
        print(f"  Dead layers: {', '.join(dead_layers)}")
    print(f"{'─'*100}\n")


class PinballTrainer:
    def __init__(self, loss_name='pinball', gpu=0, **kwargs):

        self.device = get_device(gpu)

        self.loss_fn = partial(
            pinball_loss,
            quantiles=kwargs['quantiles']
        )

    def step(self, x, target, model, opt):
        pred = model(x)

        err = target - pred
        q = torch.tensor([0.01, 0.99], device=pred.device).view(1, -1)
        loss_per_q = torch.where(err >= 0, q * err, (q - 1) * err).mean(dim=0)
        loss = self.loss_fn(pred, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=0.5)
        # nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=0.05)
        opt.step()

        return loss, model, opt


def evaluation(_dataloader, model, device, loss_fn, calc_epoch_loss=False):
    train_true_p1 = []
    train_pred_p1 = []
    train_true_p2 = []
    train_pred_p2 = []
    epoch_total_loss = 0
    for idx, batch_ in enumerate(_dataloader):
        print(idx)
        rdn = batch_['toa'].to(device)
        target = batch_['atmosphere'].to(device)

        with torch.no_grad():
            pred = model(rdn)

        loss = loss_fn(pred, target)

        pred_cpu = pred.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()

        train_pred_p1 += list(pred_cpu[..., 0].flatten())
        train_pred_p2 += list(pred_cpu[..., 1].flatten())
        train_true_p1 += list(target_cpu[..., 0].flatten())
        train_true_p2 += list(target_cpu[..., 1].flatten())

        if calc_epoch_loss:
            epoch_total_loss += loss 
            epoch_total_loss /= len(_dataloader)

    train_pred_p1 = np.array(train_pred_p1)
    train_pred_p2 = np.array(train_pred_p2)
    train_true_p1 = np.array(train_true_p1)
    train_true_p2 = np.array(train_true_p2)

    train_r2_p1 = r2_score(
        train_true_p1,
        train_pred_p1
    )
    train_r2_p2 = r2_score(
        train_true_p2,
        train_pred_p2
    )
    train_mape_p1 = mean_absolute_percentage_error(
        train_true_p1,
        train_pred_p1
    )
    train_mape_p2 = mean_absolute_percentage_error(
        train_true_p2,
        train_pred_p2
    )

    return (
        train_r2_p1, train_r2_p2,
        train_mape_p1, train_mape_p2,
        epoch_total_loss
    )


def init_wandb(
    wandb_project,
    wandb_entity,
    wandb_name,
    model_type,
    **kwargs
):
    timestamp = datetime.now().strftime(
        f"%Y%m%d_%H%M%S_%f_{wandb_name}"
    )
    try:
        wandb_config = {
            'lr': kwargs.get('lr'),
            'epochs': kwargs.get('epochs'),
            'batch_size': kwargs.get('batch_size'),
            'model_type': model_type,
        }
        wandb_config.update(kwargs)

        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=timestamp,
            dir='./',
            config=wandb_config,
            settings=wandb.Settings(_service_wait=300)
        )
        return run

    except Exception as e:
        print("WandB error!")
        print(e)
        sys.exit(1)


device = get_device(0)

rdn_path='/Users/bgreenbe/Projects/PresolveScrape/0326/files/emit20241029t023412_o30302_s005_l1b_rdn_b0106_v01.img'
atm_path='/Users/bgreenbe/Projects/PresolveScrape/0326/files/emit20241029t023412_o30302_s005_l2a_mask_b0106_v01.img'
loc_path='/Users/bgreenbe/Projects/PresolveScrape/0326/files/emit20241029t023412_o30302_s005_l1b_loc_b0106_v01.img'
obs_path='/Users/bgreenbe/Projects/PresolveScrape/0326/files/emit20241029t023412_o30302_s005_l1b_obs_b0106_v01.img'
wl_grid = '/Users/bgreenbe/Projects/H2O_AOD_Model/wavelength_grid.txt'

n = 10
dataset = ImageDataset(
    [rdn_path for i in range(n)],
    [atm_path for i in range(n)],
    [obs_path for i in range(n)],
    nsamples=100,
    wl_grid=wl_grid,
    target_fun='p99',
    cache_cube=True
)

# Check that cube filters properly
cube = np.reshape(
    dataset.toa_cube,
    (
        dataset.toa_cube.shape[0] * dataset.toa_cube.shape[1],
        dataset.toa_cube.shape[2]
    )
)

# plt.hist(cube[:, 10], bins=30)
# plt.show()

dataloader = DataLoader(
    dataset,
    batch_size=20,
    shuffle=False
)

wandb_name: str = 'Test'
wandb_entity: str = 'ev-ben-green-uc-santa-barbara'
wandb_project: str = 'ModelDev'
lr = 1e-3
epochs = 1000
model: str='p99'
batch_size = 10

run = init_wandb(
        wandb_project,
        wandb_entity,
        wandb_name,
        model,
        epochs=epochs,
        batch_size=batch_size,
    )

use_wl = dataset.wl
b = len(use_wl)
model = Model_p99(
    b,
    hidden=512,
).to(device)

opt = torch.optim.AdamW([
    {"params": model.p1_head.parameters(), "lr": 1e-3},
    {"params": model.p2_head.parameters(), "lr": 1e-3},
    {"params": model.mlp.parameters(), "lr": 1e-3},
], lr=1e-3)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=epochs,
    eta_min=1e-7,
)

trainer = Trainer_p99(quantiles=[.01, .99])

for epoch in range(epochs):
    print()
    print()
    print()
    print()
    print()
    print(epoch)
    print()
    print()
    print()
    print()
    print()
    model.train()

    # training loop
    train_epoch_total_loss = 0
    for ite, batch_ in enumerate(dataloader):
        x = batch_['toa'].to(device)
        target = batch_['atmosphere'].to(device)

        loss, model, opt = trainer.step(
            x, target, model, opt
        )

        run.log({"train/total_loss": loss})
        train_epoch_total_loss += loss
        train_epoch_total_loss /= len(dataloader)

    scheduler.step() 

    model.eval()
    train_eval_dict = evaluation_p99(
        dataloader, model, device, trainer.loss_fn
    )
    for key, value in train_eval_dict.items():
        run.log({f"train/{key}": value})

    
for idx, batch_ in enumerate(dataloader):
    rdn = batch_['toa'].to(device)
    target = batch_['atmosphere'].to(device)

    with torch.no_grad():
        pred = model(rdn)

    pred_cpu = pred.detach().cpu().numpy()
    target_cpu = target.detach().cpu().numpy()
    print(pred_cpu)
    print(target_cpu)

    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].axvline(0, color='black')
    axs[0].hist(pred_cpu[:, 0] - target_cpu[:, 0], bins=3)
    axs[0].set_xlim([-.1, .1])

    axs[1].axvline(0, color='black')
    axs[1].hist(pred_cpu[:, 1] - target_cpu[:, 1], bins=3)
    axs[1].set_xlim([-.1, .1])
    plt.show()


gradient_summary(model)

