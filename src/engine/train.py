"""The training loop, written by hand.

Flat functions, not a Trainer class. victoresque's BaseTrainer buries these five lines
under three layers of indirection (train.py -> Trainer -> BaseTrainer.train ->
abstract _train_epoch -> Trainer._train_epoch). The loop is the thing being learned
here, so it stays visible (D-003).

THE FIVE STEPS, and what each actually does to the weights:

    optimizer.zero_grad()   Clear .grad on every parameter. PyTorch ACCUMULATES
                            gradients by default -- it adds to .grad rather than
                            replacing it -- so without this, step N's gradient is the
                            sum of steps 1..N. See scripts/06_gradient_demo.py, which
                            removes this line and shows the accumulation.

    logits = model(x)       Forward pass. Every operation is recorded in an autograd
                            graph: each output tensor keeps a reference to the function
                            that produced it and to its inputs.

    loss = criterion(...)   Reduces the whole batch to ONE scalar. It must be a scalar,
                            because backward() computes d(scalar)/d(parameter) -- the
                            derivative of a vector w.r.t. a vector is a matrix, and
                            there is no single "direction to move".

    loss.backward()         Walks that graph BACKWARDS from the scalar, applying the
                            chain rule at each node, and ACCUMULATES d(loss)/d(w) into
                            w.grad for every parameter with requires_grad=True. It does
                            NOT change any weight. After this call the weights are
                            byte-identical; only .grad has changed.

    optimizer.step()        Reads .grad and writes new values into the weights. For
                            AdamW: maintains two exponential moving averages per
                            parameter -- exp_avg (of the gradient) and exp_avg_sq (of
                            its square) -- and moves each weight by
                            lr * exp_avg_hat / (sqrt(exp_avg_sq_hat) + eps), plus a
                            decoupled weight-decay term. Those two buffers are the
                            optimizer STATE, which is why the checkpoint must save
                            optimizer.state_dict() and not just the weights: resuming
                            without them restarts the moment estimates from zero and
                            produces a visible loss spike on the first resumed epoch.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.engine.metrics import RunningAverage, accuracy, macro_f1


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    show_progress: bool = True) -> dict[str, float]:
    """One pass over the training set."""
    model.train()          # dropout ON, BatchNorm uses BATCH statistics and updates
                           # its running estimates. Forgetting this is a classic bug:
                           # the model trains with eval-mode BN and never learns the
                           # population statistics it will use at inference.
    meter = RunningAverage()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []

    it = tqdm(loader, desc="train", leave=False, ncols=70) if show_progress else loader
    for x, y in it:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()      # 1. clear accumulated gradients
        logits = model(x)          # 2. forward -- builds the autograd graph
        loss = criterion(logits, y)  # 3. reduce the batch to one scalar
        loss.backward()            # 4. chain rule backwards -> populates .grad
        optimizer.step()           # 5. .grad -> new weight values

        # .item() detaches from the graph. Accumulating the tensor itself would keep
        # the whole graph alive for the epoch and leak memory.
        meter.update("loss", loss.item(), n=y.size(0))
        meter.update("acc", accuracy(logits, y), n=y.size(0))
        preds.append(logits.argmax(1).detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    out = meter.result()
    out["macro_f1"] = macro_f1(np.concatenate(trues), np.concatenate(preds))
    return out


@torch.no_grad()   # no graph is built at all -- less memory, and no accidental training
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, show_progress: bool = True,
             return_predictions: bool = False):
    """One pass over a validation or test set."""
    model.eval()           # dropout OFF, BatchNorm uses its RUNNING statistics.
                           # Without this, eval results depend on batch composition
                           # and are not reproducible.
    meter = RunningAverage()
    preds, trues, probs = [], [], []

    it = tqdm(loader, desc="eval", leave=False, ncols=70) if show_progress else loader
    for x, y in it:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        meter.update("loss", loss.item(), n=y.size(0))
        meter.update("acc", accuracy(logits, y), n=y.size(0))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())
        if return_predictions:
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)
    out = meter.result()
    out["macro_f1"] = macro_f1(y_true, y_pred)
    if return_predictions:
        return out, y_true, y_pred, np.concatenate(probs)
    return out


def epoch_time(start: float, end: float) -> str:
    m, s = divmod(end - start, 60)
    return f"{int(m)}m {s:04.1f}s"
