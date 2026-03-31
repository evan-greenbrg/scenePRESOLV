from functools import partial

import torch
from torch import nn

from scenepresolv.model_p01_p99.loss import pinball_loss
from scenepresolv.utils import get_device


class Trainer:
    def __init__(self, quantiles):

        self.device = get_device()

        self.loss_fn = partial(
            pinball_loss,
            quantiles=quantiles
        )

    def step(self, x, target, model, opt):
        pred = model(x)

        loss = self.loss_fn(pred, target)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        return loss, model, opt
