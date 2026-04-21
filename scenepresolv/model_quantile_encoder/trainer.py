from functools import partial

import torch
from torch import nn

from scenepresolv.model_quantile_encoder.loss import pinball_loss, mape_loss, log_loss, mse_loss
from scenepresolv.utils import get_device


class Trainer:
    def __init__(self, quantiles, run, wl):

        self.device = get_device()

        self.loss_fn = partial(
            # pinball_loss,
            # mape_loss,
            mse_loss,
            # log_loss,
            quantiles=quantiles
        )

        self.run = run

        # TODO allow this to vary within batch?
        self.wl = wl

    def step(self, x, target, model, opt):
        pred = model(x, self.wl)

        loss_low, loss_high = self.loss_fn(pred, target)
        loss = loss_low + loss_high

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step()

        return loss_low, loss_high, model, opt
