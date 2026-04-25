import sys
import os
from datetime import datetime

import click
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import wandb

from scenepresolv.utils import get_device
from scenepresolv.utils import gradient_summary
from scenepresolv.dataset import ImagePoolDataset
from scenepresolv.model_quantile_encoder.model import Model
from scenepresolv.model_quantile_encoder.evaluation import evaluation

from scenepresolv.model_quantile_encoder.loss import pinball_loss, mse_loss


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
            'nsamples': kwargs.get('nsamples'),
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_dataloader(worker_id):
    """
    Necessary because workers inherit state (seed) from main
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


@click.command()
@click.argument('train_pool_path')
@click.argument('train_target_path')
@click.argument('test_pool_path')
@click.argument('test_target_path')
@click.argument('wavelength_grid')
@click.argument('outdir')
@click.argument('model')
@click.argument('wandb_name')
@click.argument('wandb_entity')
@click.argument('wandb_project')
@click.option('--quantiles', '-q', multiple=True, default=[0.05, 0.95])
@click.option('--hidden', default=128)
@click.option('--batch_size', default=20)
@click.option('--nsamples', default=500)
@click.option('--epochs', default=20)
@click.option('--ncores', default=1)
@click.option('--save_every_epoch', is_flag=True, default=False)
def train(
    train_pool_path: str,
    train_target_path: str,
    test_pool_path: str,
    test_target_path: str,
    wavelength_grid: str,
    outdir: str,
    model: str,
    wandb_name: str,
    wandb_entity: str,
    wandb_project: str,
    quantiles: list = [0.05, 0.95],
    hidden: int = 128,
    batch_size: int = 20,
    nsamples: int = 300,
    epochs: int = 20,
    ncores: int = 1,
    save_every_epoch: bool = False,
):
    print(quantiles)
    device = get_device()

    train_dataset = ImagePoolDataset(
        train_pool_path,
        train_target_path,
        nsamples
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=ncores,
        worker_init_fn=init_dataloader
    )
    test_dataset = ImagePoolDataset(
        test_pool_path,
        test_target_path,
        nsamples
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=ncores,
        worker_init_fn=init_dataloader
    )

    run = init_wandb(
        wandb_project,
        wandb_entity,
        wandb_name,
        model,
        epochs=epochs,
        batch_size=batch_size,
        nsamples=nsamples,
        hidden=hidden,
    )

    wl = np.loadtxt(wavelength_grid)[:, 0] * 1000
    banddef = torch.tensor(wl, dtype=torch.float32).to(device)
    model = Model(banddef, hidden=hidden).to(device)

    model.apply(init_weights)
    with torch.no_grad():
        model.low_head[3].bias.fill_(1.0)
        # model.mid_head[3].bias.fill_(2.0)
        model.high_head[3].bias.fill_(3.0)
        model.attn_encoder.wavelength_proj.weight.data *= 2.0

    opt = torch.optim.AdamW([
        {"params": model.attn_encoder.parameters(), "lr": 1e-3, "weight_decay": 1e-3},
        {"params": model.mlp.parameters(), "lr": 5e-4, "weight_decay": 1e-3},
        {"params": model.low_head.parameters(), "lr": 1e-3},
        # {"params": model.mid_head.parameters(), "lr": 1e-2},
        {"params": model.high_head.parameters(), "lr": 1e-3},
        {"params": [model.beta_high], "lr": 5e-4},
        {"params": [model.beta_low], "lr": 5e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=1e-4,
    )

    # TODO allow this to vary within batch?
    wl = torch.tensor(wl).type(torch.float32).to(device)

    for epoch in range(epochs):
        model.train()

        opt.zero_grad()
        for i, batch_ in enumerate(train_dataloader):
            x = batch_['toa'].to(device)
            target = batch_['atmosphere'].to(device)
            pred = model(x, wl)

            # pinball_loss_low, pinball_loss_mid, pinball_loss_hi = pinball_loss(
            pinball_loss_low, pinball_loss_hi = pinball_loss(
                pred, target,
                quantiles=quantiles
            )
            # width_penalty = (pred[:, 1] - pred[:, 0]).mean() * 0.1
            # loss = pinball_loss_low + pinball_loss_mid + pinball_loss_hi + width_penalty
            loss = pinball_loss_low + pinball_loss_hi

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            opt.step()

            if i == len(train_dataloader) - 1:
                gradient_summary(model)

            opt.zero_grad()

            run.log({"train/beta_low": model.beta_low.item()})
            run.log({"train/beta_high": model.beta_high.item()})

        model.eval()
        train_eval_dict = evaluation(
            train_dataloader, model, wl, device, epoch, quantiles
        )
        for key, value in train_eval_dict.items():
            run.log({f"train/{key}": value})

        test_eval_dict = evaluation(
            test_dataloader, model, wl, device, epoch, quantiles 
        )
        scheduler.step()
        for key, value in test_eval_dict.items():
            run.log({f"test/{key}": value})
        run.log({"epoch": epoch})

        timestamp = datetime.now().strftime(
            f"%Y%m%d_%H%M%S_%f_{wandb_name}"
        )
        if save_every_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(outdir, f"presolve_{timestamp}_{epoch}.pt")
            )
    if not save_every_epoch:
        torch.save(
            model.state_dict(),
            os.path.join(outdir, f"spectf_presolve_{timestamp}.pt")
        )
    run.finish()


if __name__ == '__main__':
    train()
