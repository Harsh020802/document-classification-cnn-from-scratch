"""Shape and parameter-count assertions.

A shape check that RUNS is worth more than one that is documented. Comments drift;
these fail loudly.

CORRELATION REFERENCE DISTRIBUTION -- read before changing POOL_CORR_MIN.
The Pool3x3MPS-vs-AdaptiveAvgPool2d(3) correlation depends entirely on what is fed
through the backbone, so the threshold is meaningless without naming the input:

    torch.randn straight into the pool (not a feature map) : 0.8323
    randn INPUT through a random-init backbone             : 0.9962
    REAL cached document images through the same backbone  : 0.9860   <- asserted
    synthetic document-like proxy                          : 0.9925

This test asserts against **real cached images from the training split**, because that
is the distribution the model actually sees. Measured 0.9851-0.9873 across 5 seeds, so
the 0.98 floor has ~0.005 of headroom. Random tensors flatter the number (0.9962) by
producing an unrealistically smooth feature map; raw noise into the pool understates it
badly (0.8323) because it has no spatial correlation at all.

If this test fails, first ask whether the input distribution changed -- not just the
pooling.

Run:  PYTHONPATH=. python tests/test_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
import yaml

from src.models.blocks import Pool3x3MPS
from src.models.cnn import DocCNN
from src.utils.seed import set_seed

# Expected values, MEASURED (scripts/05_shape_walkthrough.py), not asserted from arithmetic.
BACKBONE_PARAMS = 1_172_640          # bias=False on convs; BN carries the offset
HEAD_PARAMS = {"gap1": 2_570, "gap1_hidden": 22_972, "gap3": 23_050, "gap7": 125_450}
BLOCK_SHAPES = [(32, 112, 112), (64, 56, 56), (128, 28, 28), (256, 14, 14)]
# Floor for the pooling correlation, measured on REAL cached images (see docstring).
POOL_CORR_MIN = 0.98

_fails: list[str] = []


def _real_batch(cfg: dict, n: int) -> torch.Tensor | None:
    """A batch of real cached training images, or None if the cache is not built."""
    if not (REPO / cfg["data"]["cache_dir"] / "images.npy").exists():
        return None
    from src.data.dataset import build_datasets
    train, _, _ = build_datasets(cfg, REPO)
    return torch.stack([train[i][0] for i in range(n)])


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52} {got}"
          + ("" if ok else f"   != expected {want}"))
    if not ok:
        _fails.append(f"{label}: got {got}, expected {want}")


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    size = cfg["data"]["input_size"]
    set_seed(0)
    x = torch.randn(4, 1, size, size)

    print("\n--- backbone block shapes ---")
    model = DocCNN().eval()
    seen = []
    hs = [b.register_forward_hook(lambda m, i, o: seen.append(tuple(o.shape[1:])))
          for b in model.backbone]
    with torch.no_grad():
        feat = model.forward_features(x)
    for h in hs:
        h.remove()
    for k, (got, want) in enumerate(zip(seen, BLOCK_SHAPES), 1):
        check(f"block{k} output (C,H,W)", got, want)

    print("\n--- the 14x14 divisibility constraint (reason for D-024..D-026) ---")
    check("block4 spatial size", tuple(feat.shape[-2:]), (14, 14))
    check("14 % 3 == 0 (i.e. AdaptiveAvgPool2d(3) is legal on MPS)", 14 % 3 == 0, False)
    check("14 % 7 == 0 (gap7 needs no workaround)", 14 % 7 == 0, True)

    print("\n--- Pool3x3MPS ---")
    pooled = Pool3x3MPS(3)(feat)
    check("Pool3x3MPS output", tuple(pooled.shape[1:]), (256, 3, 3))
    check("flattened features", pooled.flatten(1).shape[1], 2304)

    # Correlation is asserted on REAL cached images, not the randn batch above --
    # the reference distribution is the whole meaning of the threshold (see docstring).
    real = _real_batch(cfg, 16)
    if real is None:
        print("  [SKIP] pooling correlation -- cache not built, run scripts/04_build_cache.py")
    else:
        set_seed(0)
        with torch.no_grad():
            rfeat = DocCNN().eval().forward_features(real)
        a = Pool3x3MPS(3)(rfeat).flatten()
        b = F.adaptive_avg_pool2d(rfeat, 3).flatten()   # the op we WANT
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        ok = corr > POOL_CORR_MIN
        print(f"  [{'PASS' if ok else 'FAIL'}] "
              f"{f'corr vs AdaptiveAvgPool2d(3), REAL images > {POOL_CORR_MIN}':<52} {corr:.4f}")
        if not ok:
            _fails.append(f"pool correlation {corr:.4f} <= {POOL_CORR_MIN}")

    print("\n--- per-arm parameter counts ---")
    counts = {}
    for ht in ("gap1", "gap1_hidden", "gap3", "gap7"):
        m = DocCNN(head_type=ht, hidden_dim=86).eval()
        with torch.no_grad():
            y = m(x)
        p = m.count_parameters()
        counts[ht] = p["head"]
        check(f"{ht} logits", tuple(y.shape), (4, 10))
        check(f"{ht} head params", p["head"], HEAD_PARAMS[ht])
        check(f"{ht} backbone params", p["backbone"], BACKBONE_PARAMS)

    print("\n--- PRE-REGISTERED capacity match (B vs C) ---")
    b, c = counts["gap1_hidden"], counts["gap3"]
    rel = abs(c - b) / c
    ok = rel < 0.01
    print(f"  [{'PASS' if ok else 'FAIL'}] {'|gap3 - gap1_hidden| / gap3 < 1%':<52} "
          f"{rel:.4%}  ({b:,} vs {c:,})")
    if not ok:
        _fails.append(f"capacity match drifted to {rel:.2%}")

    print("\n" + "=" * 74)
    if _fails:
        print(f"FAILED ({len(_fails)}):")
        for f in _fails:
            print("  -", f)
        sys.exit(1)
    print("ALL SHAPE AND PARAMETER CHECKS PASSED")


if __name__ == "__main__":
    main()
