"""Phase 1.4 -- eval-mode demonstrations, printed from real runs.

  4. model.train() vs model.eval() on the IDENTICAL batch
  5. torch.no_grad(): peak memory, and grad_fn on the returned loss

Forgetting model.eval() is the most common validation bug there is, and it is exactly
as invisible in metrics as fill=0 was: validation accuracy comes out mysteriously low
and nothing errors. Show it once.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import yaml

from src.data.dataset import build_datasets, class_weights
from src.models.cnn import build_model
from src.utils.device import device_report, get_device
from src.utils.seed import set_seed


def rule(t: str = "") -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}" if t else "-" * 78)


def demo_4_train_vs_eval(model, x, y, criterion) -> None:
    rule("DEMO 4 -- model.train() vs model.eval(), identical batch and weights")
    print("  Same input. Same weights. Different outputs. Two causes:\n")
    print("    Dropout   : train() zeroes a random 50% of features and scales the rest")
    print("                by 1/(1-p); eval() passes everything through unchanged.")
    print("    BatchNorm : train() normalises using THIS BATCH's mean/var (and updates")
    print("                its running estimates); eval() uses the running estimates,")
    print("                so a sample's output no longer depends on its batch-mates.\n")

    model.train()
    set_seed(0)
    with torch.no_grad():
        out_tr1 = model(x)
    set_seed(1)
    with torch.no_grad():
        out_tr2 = model(x)

    model.eval()
    with torch.no_grad():
        out_ev1 = model(x)
        out_ev2 = model(x)

    print("  logits for sample 0, first 5 classes:")
    print(f"    train(), dropout mask A : {_row(out_tr1[0][:5])}")
    print(f"    train(), dropout mask B : {_row(out_tr2[0][:5])}")
    print(f"    eval()  , call 1        : {_row(out_ev1[0][:5])}")
    print(f"    eval()  , call 2        : {_row(out_ev2[0][:5])}")

    print(f"\n  train() is NON-deterministic : max|A - B| = "
          f"{(out_tr1 - out_tr2).abs().max():.4f}")
    print(f"  eval()  is deterministic     : max|1 - 2| = "
          f"{(out_ev1 - out_ev2).abs().max():.4f}")
    print(f"  train vs eval differ by      : max|diff|  = "
          f"{(out_tr1 - out_ev1).abs().max():.4f}")

    # The cost, in the currency that matters.
    print("\n  what it costs if you forget model.eval() at validation time:")
    print(f"    {'mode':<26}{'loss':>10}{'accuracy':>11}")
    print("    " + "-" * 47)
    model.eval()
    with torch.no_grad():
        l = criterion(model(x), y).item()
        a = (model(x).argmax(1) == y).float().mean().item()
    print(f"    {'eval()  -- correct':<26}{l:>10.4f}{a:>11.2%}")
    model.train()
    accs, losses = [], []
    for s in range(5):
        set_seed(s)
        with torch.no_grad():
            o = model(x)
            losses.append(criterion(o, y).item())
            accs.append((o.argmax(1) == y).float().mean().item())
    print(f"    {'train() -- WRONG, run 1':<26}{losses[0]:>10.4f}{accs[0]:>11.2%}")
    print(f"    {'train() -- WRONG, run 2':<26}{losses[1]:>10.4f}{accs[1]:>11.2%}")
    print(f"    {'train() -- WRONG, mean/5':<26}"
          f"{sum(losses)/5:>10.4f}{sum(accs)/5:>11.2%}")
    print("\n    Nothing errors. The number is just quietly wrong, and it varies run")
    print("    to run -- which also makes the four-arm ablation non-reproducible.")


def _row(t: torch.Tensor) -> str:
    return "[" + " ".join(f"{v:+.4f}" for v in t.detach().cpu()) + "]"


def demo_5_no_grad(model, x, y, criterion, device) -> None:
    rule("DEMO 5 -- torch.no_grad(): what the autograd graph costs when unused")
    model.eval()

    print("  (a) does the returned loss carry a graph?\n")
    loss_with = criterion(model(x), y)
    print(f"    without no_grad : requires_grad={loss_with.requires_grad}  "
          f"grad_fn={type(loss_with.grad_fn).__name__}")
    with torch.no_grad():
        loss_without = criterion(model(x), y)
    print(f"    with    no_grad : requires_grad={loss_without.requires_grad}  "
          f"grad_fn={loss_without.grad_fn}")
    print(f"    same value      : {loss_with.item():.6f} vs {loss_without.item():.6f}")

    print("\n  (b) peak memory over a 32-image batch\n")
    big = x[:1].repeat(32, 1, 1, 1)
    peaks = {}
    for label, use in (("without no_grad", False), ("with no_grad", True)):
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            base = torch.mps.current_allocated_memory()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = 0
        else:
            base = 0
        if use:
            with torch.no_grad():
                out = model(big)
                peak = _mem(device)
        else:
            out = model(big)
            peak = _mem(device)
        peaks[label] = max(peak - base, 0)
        del out
        gc.collect()
    for k, v in peaks.items():
        print(f"    {k:<20}: {v / 1024**2:>8.1f} MB")
    if peaks["without no_grad"] > 0:
        saved = peaks["without no_grad"] - peaks["with no_grad"]
        print(f"    {'saved':<20}: {saved / 1024**2:>8.1f} MB  "
              f"({saved / peaks['without no_grad']:.0%})")

    # Make the number interpretable rather than merely large.
    blocks = [(32, 112), (64, 56), (128, 28), (256, 14)]
    tot = sum(32 * c * s * s * 4 / 1024**2 for c, s in blocks)
    print(f"\n    for scale: the four BLOCK OUTPUTS alone are {tot:.1f} MB at this batch")
    print( "    size. The rest is the ~5 further intermediates each block saves (2 conv,")
    print( "    2 BN, 2 ReLU, 1 pool) -- roughly a 17x multiplier over the block outputs.")
    print( "    A 0.0 MB delta under no_grad is not a measurement failure: intermediates")
    print( "    are freed the moment they are consumed, so only the weights remain.")

    print("\n  The forward pass normally SAVES every intermediate activation, because")
    print("  backward() needs them to apply the chain rule. At validation there is no")
    print("  backward pass, so all of that is retained for nothing. no_grad() stops it")
    print("  being recorded at all -- which is why eval can use a larger batch than")
    print("  training on the same hardware.")


def _mem(device: torch.device) -> int:
    if device.type == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    return 0


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device: {device_report(device)}")

    train_ds, _, _ = build_datasets(cfg, REPO)
    labels_all = train_ds.cache.labels[train_ds.indices]
    picks = [int((labels_all == c).nonzero()[0][0]) for c in range(8)]
    x = torch.stack([train_ds[i][0] for i in picks]).to(device)
    y = torch.tensor([train_ds[i][1] for i in picks]).to(device)
    print(f"batch : x {tuple(x.shape)}  y {y.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds.class_counts).to(device))
    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)

    demo_4_train_vs_eval(model, x, y, criterion)
    demo_5_no_grad(model, x, y, criterion, device)
    rule()
    print("EVAL DEMOS COMPLETE")


if __name__ == "__main__":
    main()
