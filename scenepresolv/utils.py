import random
from typing import List, Optional, Tuple

import numpy as np
import torch


def get_device(gpu:Optional[int]=None) -> torch.device:
    """From https://github.com/emit-sds/SpecTf/blob/main/spectf/utils.py"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu else "cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps") # Apple silicon
    else:
        return torch.device("cpu")


def seed(i:int = 42) -> None:
    """From https://github.com/emit-sds/SpecTf/blob/main/spectf/utils.py"""
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)


def file_to_list(path):
    with open(path, 'r') as f:
        paths = [l.strip() for l in f.readlines()]
    return paths


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

    print(f"{'─'*100}")
    exploding = sum(r['exploding'] for r in rows)
    vanishing = sum(r['vanishing'] for r in rows)
    dead_layers = [r['name'] for r in rows if r['dead'] is not None and r['dead'] > 0.5]

    print(f"  Params total: {total_params:,}  |  with grad: {total_with_grad:,}")
    print(f"  Exploding: {exploding}  |  Vanishing: {vanishing}  |  >50% dead: {len(dead_layers)}")
    if dead_layers:
        print(f"  Dead layers: {', '.join(dead_layers)}")
    print(f"{'─'*100}\n")
