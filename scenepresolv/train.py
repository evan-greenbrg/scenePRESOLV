from pathlib import Path
import sys
import os
from datetime import datetime

import click
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from scenepresolv.utils import get_device
from scenepresolv.utils import seed as useed
from scenepresolv.utils import file_to_list
from scenepresolv.utils import gradient_summary
from scenepresolv.dataset import ImageDataset
from scenepresolv.model_p01_p99.model import Model as Model_p99
from scenepresolv.model_p01_p99.trainer import Trainer as Trainer_p99
from scenepresolv.model_p01_p99.evaluation import evaluation as evaluation_p99

from scenepresolv.model_quantile_encoder.model import Model as Model_attn
from scenepresolv.model_quantile_encoder.trainer import Trainer as Trainer_attn
from scenepresolv.model_quantile_encoder.evaluation import evaluation as evaluation_attn


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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@click.command()
@click.argument('train_rdn_path')
@click.argument('train_atm_path')
@click.argument('train_obs_path')
@click.argument('test_rdn_path')
@click.argument('test_atm_path')
@click.argument('test_obs_path')
@click.argument('wavelength_grid')
@click.argument('outdir')
@click.argument('model')
@click.argument('wandb_name')
@click.argument('wandb_entity')
@click.argument('wandb_project')
@click.option('--cube_cache_root', default='')
@click.option('--batch_size', default=20)
@click.option('--nsamples', default=100)
@click.option('--epochs', default=20)
@click.option('--seed', default=42)
@click.option('--save_every_epoch', is_flag=True, default=False)
@click.option('--just_cube', is_flag=True, default=False)
def train(
    train_rdn_path: str,
    train_atm_path: str,
    train_obs_path: str,
    test_rdn_path: str,
    test_atm_path: str,
    test_obs_path: str,
    wavelength_grid: str,
    outdir: str,
    model: str,
    wandb_name: str,
    wandb_entity: str,
    wandb_project: str,
    cube_cache_root: str = None,
    batch_size: int = 20,
    nsamples: int = 300,
    epochs: int = 20,
    seed: int = 42,
    save_every_epoch: bool = False,
    just_cube: bool = False
):
    useed(seed)
    device = get_device()

    cache_root = Path(cube_cache_root)
    train_dataset = ImageDataset(
        file_to_list(train_rdn_path),
        file_to_list(train_atm_path),
        file_to_list(train_obs_path),
        nsamples=nsamples,
        wl_grid=wavelength_grid,
        target_fun='IQR',
        cache_cube=True,
        save_to_disk=True,
        cache_root=os.path.join(
            cache_root.parent,
            'train_' + cache_root.stem
        )
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = ImageDataset(
        file_to_list(test_rdn_path),
        file_to_list(test_atm_path),
        file_to_list(test_obs_path),
        nsamples=nsamples,
        wl_grid=wavelength_grid,
        target_fun='IQR',
        cache_cube=True,
        save_to_disk=True,
        cache_root=os.path.join(
            cache_root.parent,
            'test_' + cache_root.stem
        )
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    if just_cube:
        print("Finished just building cube")
        return

    run = init_wandb(
        wandb_project,
        wandb_entity,
        wandb_name,
        model,
        epochs=epochs,
        batch_size=batch_size,
    )

    if model == 'p99':
        use_wl = train_dataset.wl
        b = len(use_wl)
        model = Model_p99(b, hidden=256).to(device)

        opt = torch.optim.AdamW([
            {"params": model.p1_head.parameters(), "lr": 1e-3},
            {"params": model.p2_head.parameters(), "lr": 1e-3},
            {"params": model.mlp.parameters(), "lr": 1e-3},
            {"params": [model.tau_min_raw], "lr": 1e-2},
            {"params": model.residual_attn.parameters(),"lr": 3e-3, "weight_decay": 0.0}
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs,
            eta_min=1e-4,
        )

        trainer = Trainer_p99(
            quantiles=[.01, .99],
            run=run
        )

        evaluation = evaluation_p99

    elif model == 'p95':
        use_wl = train_dataset.wl
        b = len(use_wl)
        model = Model_p99(b, hidden=512).to(device)

        opt = torch.optim.AdamW([
            {"params": model.p1_head.parameters(), "lr": 1e-3},
            {"params": model.p2_head.parameters(), "lr": 1e-3},
            {"params": model.mlp.parameters(), "lr": 1e-3},
            {"params": [model.tau_min_raw], "lr": 1e-2},
            {"params": model.residual_attn.parameters(),"lr": 3e-3, "weight_decay": 0.0}
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=epochs,
            eta_min=1e-4,
        )

        trainer = Trainer_p99(
            quantiles=[.01, .99],
            run=run
        )

        evaluation = evaluation_p99

    elif model == 'attn':
        use_wl = train_dataset.wl
        b = len(use_wl)
        banddef = torch.tensor(use_wl, dtype=torch.float32).to(device)
        model = Model_attn(banddef, hidden=128).to(device)

        model.apply(init_weights)
        # nn.init.xavier_uniform_(model.attn_encoder.readout)

        run.watch(model, log="all", log_freq=100)

        opt = torch.optim.AdamW([
            {"params": model.p1_head.parameters(), "lr": 1e-4},
            {"params": model.p2_head.parameters(), "lr": 2e-4},
            {"params": model.mlp.parameters(), "lr": 1e-4},
            {"params": model.attn_encoder.parameters(), "lr": 5e-5},
            {"params": [model.beta_low, model.beta_high], "lr": 1e-3},
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            patience=3,
            factor=0.5,
            min_lr=1e-6
        )

        # TODO allow this to vary within batch?
        wl = torch.tensor(
            train_dataset.wl
        ).type(torch.float32).to(device)

        trainer = Trainer_attn(
            quantiles=[.25, .75],
            run=run,
            wl=wl
        )

        evaluation = evaluation_attn

    else:
        raise ValueError("Model string is invalid")

    for epoch in range(epochs):
        model.train()
        train_epoch_total_loss = 0

        opt.zero_grad()
        for i, batch_ in enumerate(train_dataloader):
            x = batch_['toa'].to(device)
            target = batch_['atmosphere'].to(device)

            loss_low, loss_high = trainer.loss_fn(model(x, wl), target)
            loss = (loss_low + loss_high)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            opt.step()
            opt.zero_grad()

            run.log({"train/total_loss": loss_low + loss_high})
            run.log({"train/low_loss": loss_low})
            run.log({"train/high_loss": loss_high})

            run.log({"train/beta_low": model.beta_low.item()})
            run.log({"train/beta_high": model.beta_high.item()})
            train_epoch_total_loss += (loss_low + loss_high)
            train_epoch_total_loss /= len(train_dataloader)

        model.eval()
        run.log({"train/epoch_total_loss": train_epoch_total_loss})
        train_eval_dict = evaluation(
            train_dataloader, model, device, trainer.loss_fn
        )
        for key, value in train_eval_dict.items():
            run.log({f"train/{key}": value})

        test_eval_dict = evaluation(
            test_dataloader, model, device, trainer.loss_fn,
        )

        model.eval()
        run.log({"train/epoch_total_loss": train_epoch_total_loss})
        train_eval_dict = evaluation(
            train_dataloader, model, device, trainer.loss_fn
        )
        for key, value in train_eval_dict.items():
            run.log({f"train/{key}": value})

        test_eval_dict = evaluation(
            test_dataloader, model, device, trainer.loss_fn,
        )
        scheduler.step(test_eval_dict['loss_high'])
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

    gradient_summary(model)
    run.finish()


if __name__ == '__main__':
    train()
