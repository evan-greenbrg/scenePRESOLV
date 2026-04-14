import torch
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error
)


def evaluation(_dataloader, model, device, loss_fn):
    train_true_p1 = []
    train_pred_p1 = []
    train_true_p2 = []
    train_pred_p2 = []
    for idx, batch_ in enumerate(_dataloader):
        rdn = batch_['toa'].to(device)
        target = batch_['atmosphere'].to(device)

        with torch.no_grad():
            pred = model(rdn)

        pred_cpu = pred.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()

        train_pred_p1 += list(pred_cpu[..., 0].flatten())
        train_pred_p2 += list(pred_cpu[..., 1].flatten())
        train_true_p1 += list(target_cpu[..., 0].flatten())
        train_true_p2 += list(target_cpu[..., 1].flatten())

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

    return {
        "r2_p1": train_r2_p1,
        "r2_p2": train_r2_p2,
        "mape_p1": train_mape_p1,
        "mape_p2": train_mape_p2
    }
