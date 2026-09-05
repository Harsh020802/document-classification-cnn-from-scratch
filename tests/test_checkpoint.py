"""Checkpoint round-trip and resume-continuity tests.

This is where the victoresque-format decision (D-003) either pays off or turns out to be
words in a notes file.

TEST 1 -- round trip. Save, build FRESH model and optimizer, load, then assert:
  (a) identical outputs on a fixed batch
  (b) exp_avg / exp_avg_sq restored, NOT reinitialised to zero
  (c) the arch config dict survives, since telling the four ablation arms apart
      depends on it

TEST 2 -- resume continuity, the real check. Train N steps, save, load into fresh
objects, train N more, and assert NO LOSS SPIKE at the join. That spike is the visible
symptom of dropped optimizer state, and preventing it is the entire reason we adopted
the full checkpoint dict over torch.save(model.state_dict(), ...).

Run:  PYTHONPATH=. python tests/test_checkpoint.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import yaml

from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.models.cnn import build_model
from src.utils.seed import set_seed

_fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<54}{detail}")
    if not ok:
        _fails.append(label)


def _fixture(cfg: dict, seed: int = 0):
    set_seed(seed)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    return model, opt, sched


def _batch(n: int = 4):
    set_seed(123)
    return torch.randn(n, 1, 224, 224), torch.randint(0, 10, (n,))


def test_round_trip(cfg: dict, tmp: Path) -> None:
    print("\n--- TEST 1: round trip ---")
    x, y = _batch()
    crit = nn.CrossEntropyLoss()

    model, opt, sched = _fixture(cfg)
    # Two real steps, so the optimizer has non-trivial state to lose.
    for _ in range(2):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        out_before = model(x).clone()

    w = model.backbone[0].body[0].weight
    ea_before = opt.state[w]["exp_avg"].clone()
    eas_before = opt.state[w]["exp_avg_sq"].clone()
    check("optimizer state is non-trivial before saving",
          ea_before.abs().sum().item() > 0, f"|exp_avg| sum {ea_before.abs().sum():.4f}")

    path = tmp / "ckpt.pt"
    arch = {"type": "DocCNN", **cfg["model"]}
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, epoch=3,
                    monitor_best=0.7123, config=cfg, arch=arch)
    check("checkpoint file written", path.exists(),
          f"{path.stat().st_size / 1e6:.2f} MB")

    # FRESH objects with a DIFFERENT seed -- so anything not restored is visibly wrong.
    model2, opt2, sched2 = _fixture(cfg, seed=999)
    model2.eval()
    with torch.no_grad():
        out_fresh = model2(x)
    check("fresh model differs before loading (sanity)",
          (out_fresh - out_before).abs().max().item() > 1e-4,
          f"max|diff| {(out_fresh - out_before).abs().max():.4e}")

    ckpt = load_checkpoint(path, model=model2, optimizer=opt2, scheduler=sched2)
    model2.eval()
    with torch.no_grad():
        out_after = model2(x)

    d = (out_after - out_before).abs().max().item()
    check("(a) identical outputs after load", d < 1e-6, f"max|diff| {d:.3e}")

    w2 = model2.backbone[0].body[0].weight
    st = opt2.state.get(w2, {})
    has = "exp_avg" in st and "exp_avg_sq" in st
    check("(b) exp_avg present after load", has)
    if has:
        d1 = (st["exp_avg"] - ea_before).abs().max().item()
        d2 = (st["exp_avg_sq"] - eas_before).abs().max().item()
        check("(b) exp_avg restored exactly", d1 < 1e-9, f"max|diff| {d1:.3e}")
        check("(b) exp_avg_sq restored exactly", d2 < 1e-9, f"max|diff| {d2:.3e}")
        check("(b) exp_avg NOT reinitialised to zero",
              st["exp_avg"].abs().sum().item() > 0,
              f"|exp_avg| sum {st['exp_avg'].abs().sum():.4f}")

    check("(c) arch dict round-trips", ckpt["arch"] == arch,
          f"head={ckpt['arch']['head']['type']}")
    check("(c) epoch round-trips", ckpt["epoch"] == 3)
    check("(c) monitor_best round-trips", abs(ckpt["monitor_best"] - 0.7123) < 1e-9)
    check("(c) scheduler last_lr restored",
          abs(sched2.get_last_lr()[0] - sched.get_last_lr()[0]) < 1e-12,
          f"lr {sched2.get_last_lr()[0]:.6e}")

    # The arch guard: loading arm C's weights into arm B must raise, not silently work.
    cfg_b = yaml.safe_load(yaml.safe_dump(cfg))
    cfg_b["model"]["head"]["type"] = "gap1_hidden"
    set_seed(0)
    model_b = build_model(cfg_b)
    try:
        load_checkpoint(path, model=model_b)
        check("arch mismatch raises (gap3 ckpt -> gap1_hidden model)", False, "no error")
    except ValueError:
        check("arch mismatch raises (gap3 ckpt -> gap1_hidden model)", True)


def test_resume_continuity(cfg: dict, tmp: Path) -> None:
    """Resume must not spike the loss -- with the RNG stream held fixed.

    SUBTLETY that broke the first version of this test: the model has dropout, which
    draws a fresh random mask on every train()-mode forward. An interrupted run and an
    uninterrupted run therefore diverge from RNG alone, regardless of the checkpoint.
    The first version measured that drift and concluded the checkpoint was broken --
    the giveaway was that "full resume" and "weights only" produced BYTE-IDENTICAL
    curves, which is impossible if optimizer state mattered.

    Fix: reseed to a known value at the resume boundary in every arm, so the dropout
    masks after the join are identical across arms and the ONLY difference is whether
    the optimizer's moment estimates were restored.
    """
    print("\n--- TEST 2: resume continuity (no loss spike at the join) ---")
    print("    RNG reseeded identically at the boundary, so dropout masks match across")
    print("    arms and the optimizer state is the only variable.\n")
    x, y = _batch(8)
    crit = nn.CrossEntropyLoss()
    N, RESUME_SEED = 12, 4242

    def run(steps: int, model, opt, seed_at_start: int | None = None) -> list[float]:
        if seed_at_start is not None:
            set_seed(seed_at_start)
        model.train()
        out = []
        for _ in range(steps):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    # Arm 1 -- uninterrupted reference. Reseeded at step N so its post-join RNG
    # stream matches the resumed arms exactly.
    model, opt, _ = _fixture(cfg)
    ref = run(N, model, opt, seed_at_start=7)
    ref += run(N, model, opt, seed_at_start=RESUME_SEED)

    # Arm 2 -- interrupted, FULL checkpoint (weights + optimizer).
    model, opt, _ = _fixture(cfg)
    part = run(N, model, opt, seed_at_start=7)
    path = tmp / "resume.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=None, epoch=1,
                    monitor_best=0.0, config=cfg,
                    arch={"type": "DocCNN", **cfg["model"]})
    model2, opt2, _ = _fixture(cfg, seed=999)
    load_checkpoint(path, model=model2, optimizer=opt2)
    part += run(N, model2, opt2, seed_at_start=RESUME_SEED)

    # Arm 3 -- interrupted, WEIGHTS ONLY (what torch.save(model.state_dict()) gives).
    model3, opt3, _ = _fixture(cfg, seed=999)
    load_checkpoint(path, model=model3)
    naive = list(part[:N]) + run(N, model3, opt3, seed_at_start=RESUME_SEED)

    print(f"    {'step':<6}{'uninterrupted':>15}{'resumed (full)':>17}{'weights only':>16}")
    print("    " + "-" * 54)
    for i in range(N - 2, min(N + 4, 2 * N)):
        mark = "  <-- resume" if i == N else ""
        print(f"    {i+1:<6}{ref[i]:>15.5f}{part[i]:>17.5f}{naive[i]:>16.5f}{mark}")

    dev_full = max(abs(a - b) for a, b in zip(ref, part))
    dev_naive = max(abs(a - b) for a, b in zip(ref, naive))
    print(f"\n    max deviation from uninterrupted, full resume  : {dev_full:.3e}")
    print(f"    max deviation from uninterrupted, weights only : {dev_naive:.3e}")

    check("(a) full resume reproduces the uninterrupted curve exactly",
          dev_full < 1e-6, f"max|diff| {dev_full:.3e}")
    check("(b) weights-only resume does NOT reproduce it",
          dev_naive > 1e-4, f"max|diff| {dev_naive:.3e}")
    check("(c) full resume is strictly better than weights-only",
          dev_full < dev_naive, f"{dev_full:.3e} < {dev_naive:.3e}")


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_round_trip(cfg, tmp)
        test_resume_continuity(cfg, tmp)
    print("\n" + "=" * 74)
    if _fails:
        print(f"FAILED ({len(_fails)}):")
        for f in _fails:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKPOINT CHECKS PASSED")


if __name__ == "__main__":
    main()
