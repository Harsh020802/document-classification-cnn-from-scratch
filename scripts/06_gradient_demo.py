"""Phase 1.4 -- three demonstrations, printed from real runs.

  1. Watch a gradient appear:  .grad is None -> populated -> weights move
  2. Remove zero_grad() and watch gradients accumulate
  3. Overfit a single batch of 8 -- the standard sanity check

Everything printed is a real number from a real forward/backward pass.
"""
from __future__ import annotations

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


def demo_1_gradient_appears(model, x, y, criterion, device) -> None:
    rule("DEMO 1 -- watch a gradient appear, and watch the weights move")

    # One specific weight, tracked all the way through. First conv, first filter.
    w = model.backbone[0].body[0].weight          # (32, 1, 3, 3)
    print(f"  tracking: backbone[0].body[0].weight  shape {tuple(w.shape)}")
    print(f"  showing filter 0, the full 3x3 kernel\n")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    opt.zero_grad(set_to_none=True)

    print("  (a) BEFORE loss.backward()")
    print(f"      w.grad is  : {w.grad}")
    before = w[0, 0].detach().clone()
    print(f"      w[0,0]     :\n{_fmt(before)}")

    logits = model(x)
    loss = criterion(logits, y)
    print(f"\n      forward pass done. loss = {loss.item():.6f}")
    print(f"      loss is a scalar: shape {tuple(loss.shape)}, "
          f"requires_grad={loss.requires_grad}")
    print(f"      loss.grad_fn = {loss.grad_fn}   <- the autograd graph's last node")

    loss.backward()

    print("\n  (b) AFTER loss.backward()")
    print(f"      w.grad is  : a {tuple(w.grad.shape)} tensor, no longer None")
    print(f"      w.grad[0,0]:\n{_fmt(w.grad[0, 0])}")
    print(f"      |grad| mean = {w.grad.abs().mean():.3e}   max = {w.grad.abs().max():.3e}")
    print(f"      w[0,0] UNCHANGED by backward(): max|diff| = "
          f"{(w[0,0] - before).abs().max().item():.3e}")
    print("      -> backward() computes derivatives. It does not touch the weights.")

    opt.step()

    print("\n  (c) AFTER optimizer.step()")
    after = w[0, 0].detach().clone()
    print(f"      w[0,0]     :\n{_fmt(after)}")
    delta = (after - before)
    print(f"      change     :\n{_fmt(delta)}")
    print(f"      max|delta| = {delta.abs().max().item():.3e}  "
          f"(lr = 3e-4, so a first AdamW step is ~lr in magnitude)")
    print("      -> step() read .grad and wrote new values into the weights.")

    st = opt.state[w]
    print(f"\n  (d) the optimizer now has STATE for this tensor:")
    for k, v in st.items():
        print(f"      {k:<12}: {tuple(v.shape) if torch.is_tensor(v) else v}")
    print("      exp_avg / exp_avg_sq are AdamW's per-parameter moment estimates.")
    print("      They are why checkpoints must save optimizer.state_dict(): resuming")
    print("      without them restarts these at zero and spikes the loss (D-003).")


def _fmt(t: torch.Tensor) -> str:
    return "\n".join("        [" + "  ".join(f"{v:+.6f}" for v in row) + "]"
                     for row in t.detach().cpu())


def demo_2_zero_grad(model_fn, x, y, criterion, device) -> None:
    rule("DEMO 2 -- why zero_grad() exists: remove it and watch gradients accumulate")

    for label, use_zero in (("WITHOUT optimizer.zero_grad()", False),
                            ("WITH optimizer.zero_grad()", True)):
        set_seed(0)
        m = model_fn().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
        w = m.backbone[0].body[0].weight
        print(f"\n  {label}")
        print(f"    {'step':<6}{'loss':>10}{'|grad| mean':>15}{'|grad| max':>14}")
        print("    " + "-" * 45)
        for step in range(1, 4):
            if use_zero:
                opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            print(f"    {step:<6}{loss.item():>10.4f}"
                  f"{w.grad.abs().mean().item():>15.3e}{w.grad.abs().max().item():>14.3e}")
            opt.step()

    print("\n  PyTorch ACCUMULATES into .grad rather than overwriting it. Without")
    print("  zero_grad() step 3's gradient is the SUM of steps 1-3, so the effective")
    print("  step size grows every iteration and the update no longer points along the")
    print("  current batch's gradient. (The accumulation behaviour is deliberate -- it")
    print("  is how you simulate a large batch on small memory -- but it must be opted")
    print("  into, not inherited by forgetting a line.)")


def demo_3_overfit_batch(model_fn, x, y, criterion, device, steps: int = 300) -> bool:
    rule(f"DEMO 3 -- overfit a single batch of {x.size(0)} images (the sanity check)")
    print("  If the model cannot memorise 8 images, something is broken: wrong loss")
    print("  reduction, gradients not flowing, a detached tensor, misaligned labels.")
    print("  Better to find that out in 30 seconds than 20 minutes into a real epoch.\n")

    set_seed(0)
    m = model_fn().to(device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)   # higher lr: we WANT overfitting

    print(f"    {'step':<7}{'loss':>10}{'acc':>8}")
    print("    " + "-" * 25)
    losses = []
    for s in range(1, steps + 1):
        opt.zero_grad()
        logits = m(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if s in (1, 5, 10, 25, 50, 100, 200, steps):
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    {s:<7}{loss.item():>10.5f}{acc:>8.2f}")

    final = losses[-1]
    ok = final < 0.05
    print(f"\n    start {losses[0]:.4f}  ->  final {final:.5f}")
    print(f"    {'PASS' if ok else 'FAIL'}: loss {'reached' if ok else 'did NOT reach'} "
          f"near-zero (< 0.05)")
    if not ok:
        print("    STOP -- debug before running a real epoch.")
    return ok


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device: {device_report(device)}")

    train_ds, _, _ = build_datasets(cfg, REPO)

    # One image from each of 8 DIFFERENT classes.
    #
    # Taking train_ds[0..7] gives all class 0 -- the split indices are sorted, so the
    # first eight rows are the same class. That makes demo 3 vacuous: "memorise 8
    # images" collapses to "always predict class 0", a constant function reachable by
    # pushing a single logit up. It would converge in 5 steps while being incapable of
    # detecting misaligned labels, which is one of the main bugs the check exists for.
    labels_all = train_ds.cache.labels[train_ds.indices]
    picks: list[int] = []
    for c in range(8):
        where = (labels_all == c).nonzero()[0]
        if len(where):
            picks.append(int(where[0]))
    x = torch.stack([train_ds[i][0] for i in picks]).to(device)
    y = torch.tensor([train_ds[i][1] for i in picks]).to(device)
    print(f"batch  : x {tuple(x.shape)}  y {y.tolist()}   "
          f"({len(set(y.tolist()))} distinct classes)")

    w = class_weights(train_ds.class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    def model_fn():
        return build_model(cfg)

    set_seed(cfg["seed"])
    demo_1_gradient_appears(model_fn().to(device), x, y, criterion, device)
    demo_2_zero_grad(model_fn, x, y, criterion, device)
    ok = demo_3_overfit_batch(model_fn, x, y, criterion, device)
    rule()
    print("ALL THREE DEMOS COMPLETE" if ok else "SANITY CHECK FAILED -- do not proceed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
