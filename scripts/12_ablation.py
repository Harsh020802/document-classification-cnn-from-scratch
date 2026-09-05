"""The four-arm ablation. PRE-REGISTERED -- see configs/ablation.yaml.

Runs all arms from one config with only `model.head.type` varying, so nothing can drift
between them. Writes per-arm CSVs into a shared run directory and emits the comparison
table, the McNemar matrix, and PER-CLASS F1 for every arm.

Protocol (locked before any run):
  1. determinism check -- same seed, 3 epochs, x2. Diagnostic ONLY.
  2. seed-variance floor -- arm C x 3 seeds, full runs. Report the RANGE, not an SD.
  3. gate -- if the range is large, stop and discuss. Judgment call, not a threshold.
  4. four arms at seed 0, sealed test split.
  5. PRIMARY: McNemar exact, gap3 vs gap1_hidden(86), unadjusted alpha=0.05.
  6. SECONDARY: 5 remaining pairs, Holm-corrected, labelled exploratory.
  7. every number printed against the measured seed floor.
  8. inconclusive is a valid outcome.

PER-CLASS F1 (D-044): the Scientific hypothesis -- that its ~0.58 ceiling is a MODALITY
limit rather than a capacity or resolution limit -- is testable here for free. The arms
span no spatial grid (gap1) to a 7x7 grid (gap7). If Scientific stays flat across all
four, the ceiling is modality and the Week 3 OCR case is proven. If it moves with spatial
resolution, the claim needs weakening.

Usage:
  PYTHONPATH=. python scripts/12_ablation.py --stage floor
  PYTHONPATH=. python scripts/12_ablation.py --stage arms
  PYTHONPATH=. python scripts/12_ablation.py --stage analyse
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.stats import binomtest

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]
ARMS = {"A": "gap1", "B": "gap1_hidden", "C": "gap3", "D": "gap7"}


def run_training(head: str, seed: int, tag: str, epochs: int | None = None) -> Path:
    cmd = [sys.executable, "scripts/09_train.py", "--head", head,
           "--seed", str(seed), "--tag", tag]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO, check=True,
                   env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    return sorted((REPO / "outputs" / "runs").glob(f"*_{head}_seed{seed}_{tag}"))[-1]


def mcnemar(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Exact binomial McNemar on the DISCORDANT pairs only.

    Conditions on the same test images, so shared correctness cancels -- that is where
    the power comes from. b = model1 right & model2 wrong; c = the reverse. Everything
    both get right or both get wrong carries no information about which is better.

    Counterintuitive and worth remembering: MORE discordant pairs means a LARGER
    detectable difference, since SD(b-c) = sqrt(n_disc) while the denominator (n_test)
    is fixed. High disagreement means both models are individually noisy.
    """
    c1, c2 = p1 == y_true, p2 == y_true
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    n = b + c
    p = binomtest(b, n, 0.5, alternative="two-sided").pvalue if n else 1.0
    k = 2.80   # 1.96 + 0.84 -> 80% power at alpha=.05 two-sided. ONE k throughout.
    return {"b": b, "c": c, "n_discordant": n, "p": float(p),
            "detectable_pts": float(k * np.sqrt(n) / len(y_true) * 100) if n else float("nan"),
            "observed_pts": float((c1.mean() - c2.mean()) * 100)}


def holm(pairs: list[tuple[str, float]], alpha: float = 0.05) -> list[tuple[str, float, float, bool]]:
    """Holm step-down. More powerful than Bonferroni at the same FWER, and assumes no
    independence -- which matters here, since the tests share arms."""
    s = sorted(pairs, key=lambda x: x[1])
    m = len(s)
    out, rejected = [], True
    for i, (name, p) in enumerate(s):
        thr = alpha / (m - i)
        rejected = rejected and p <= thr
        out.append((name, p, thr, rejected))
    return out


def analyse(run_dirs: dict[str, Path], floor: dict | None) -> None:
    res = {a: json.loads((d / "results.json").read_text()) for a, d in run_dirs.items()}
    preds = {a: np.load(d / "test_predictions.npy", allow_pickle=True).item()
             for a, d in run_dirs.items()}
    y_true = preds["A"]["y_true"]

    print("\n" + "=" * 78)
    print("ABLATION RESULTS -- sealed test split, seed 0")
    print("=" * 78)
    if floor:
        print(f"  measured seed-variance floor (arm C x 3 seeds): "
              f"RANGE {floor['range_acc']*100:.2f} acc pts / "
              f"{floor['range_f1']:.4f} macro-F1")
        print("  every difference below must be read against that range.\n")
    else:
        print("  NO SEED FLOOR MEASURED -- run --stage floor first. Differences below\n"
              "  cannot be interpreted without it.\n")

    print(f"  {'arm':<5}{'head':<14}{'spatial':<9}{'head par':>10}{'test acc':>10}{'macro-F1':>10}")
    print("  " + "-" * 60)
    spatial = {"A": "none", "B": "none", "C": "3x3", "D": "7x7"}
    for a in "ABCD":
        r = res[a]
        print(f"  {a:<5}{r['head']:<14}{spatial[a]:<9}{r['params']['head']:>10,}"
              f"{r['test_acc']:>10.4f}{r['test_macro_f1']:>10.4f}")

    print("\n  PRIMARY (pre-registered, unadjusted alpha=0.05): C gap3 vs B gap1_hidden(86)")
    print("  matched capacity (23,050 vs 22,972 = 0.34%), spatial info the only difference")
    m = mcnemar(y_true, preds["C"]["y_pred"], preds["B"]["y_pred"])
    print(f"    b (C right, B wrong) = {m['b']}   c (B right, C wrong) = {m['c']}   "
          f"discordant = {m['n_discordant']}")
    print(f"    observed difference  = {m['observed_pts']:+.2f} accuracy points")
    print(f"    detectable at 80% power (k=2.80) = {m['detectable_pts']:.2f} points")
    print(f"    McNemar exact p = {m['p']:.4f}   -> "
          f"{'REJECT H0' if m['p'] < 0.05 else 'fail to reject H0'}")
    print("\n    SCOPE: this is a claim about these two FITTED MODELS at seed 0, not about")
    print("    the architectures (Dietterich 1998). The architectural claim is licensed")
    print("    ONLY if the effect also exceeds the measured seed floor.")
    if floor:
        ok = abs(m["observed_pts"]) > floor["range_acc"] * 100
        print(f"    effect {abs(m['observed_pts']):.2f} pts vs floor "
              f"{floor['range_acc']*100:.2f} pts -> architectural claim "
              f"{'SUPPORTED' if ok else 'NOT supported'}")

    print("\n  SECONDARY (exploratory, Holm-corrected)")
    sec = []
    for a, b in itertools.combinations("ABCD", 2):
        if {a, b} == {"B", "C"}:
            continue
        mm = mcnemar(y_true, preds[a]["y_pred"], preds[b]["y_pred"])
        sec.append((f"{a}-{b}", mm["p"]))
    print(f"    {'pair':<7}{'p':>9}{'Holm thr':>11}{'significant':>13}")
    for name, p, thr, rej in holm(sec):
        print(f"    {name:<7}{p:>9.4f}{thr:>11.4f}{str(rej):>13}")

    # PER-CLASS F1 -- the D-044 Scientific test.
    print("\n  PER-CLASS F1 BY ARM  (D-044: is Scientific's ceiling MODALITY or capacity?)")
    print(f"    {'class':<12}" + "".join(f"{a:>9}" for a in "ABCD") + f"{'range':>9}")
    print("    " + "-" * 57)
    for c in CLASSES:
        vals = [res[a]["per_class"][c]["f1"] for a in "ABCD"]
        rng = max(vals) - min(vals)
        star = "  <-- " if c == "Scientific" else ""
        print(f"    {c:<12}" + "".join(f"{v:>9.3f}" for v in vals) + f"{rng:>9.3f}{star}")
    sv = [res[a]["per_class"]["Scientific"]["f1"] for a in "ABCD"]
    srng = max(sv) - min(sv)
    print(f"\n    Scientific range across arms = {srng:.3f}")
    if floor:
        print(f"    seed-variance floor (macro-F1) = {floor['range_f1']:.3f}")
        verdict = ("MODALITY ceiling confirmed -- flat across spatial resolutions"
                   if srng <= floor["range_f1"]
                   else "moves with spatial resolution -- the modality claim needs weakening")
        print(f"    -> {verdict}")

    out = REPO / "outputs" / "ablation_results.json"
    out.write_text(json.dumps({
        "arms": {a: {k: res[a][k] for k in
                     ("head", "test_acc", "test_macro_f1", "per_class", "params")}
                 for a in "ABCD"},
        "primary": m, "secondary": sec, "floor": floor,
        "scientific_range": srng,
    }, indent=2))
    print(f"\n  saved -> {out.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["determinism", "floor", "arms", "analyse"])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    if a.stage == "determinism":
        print("Determinism check: arm C, same seed, 3 epochs, twice. DIAGNOSTIC ONLY --\n"
              "this is not the noise floor (see D-021).")
        d1 = run_training("gap3", 0, "det1", epochs=3)
        d2 = run_training("gap3", 0, "det2", epochs=3)
        import csv
        rows = []
        for d in (d1, d2):
            with (d / "metrics.csv").open() as f:
                rows.append(list(csv.DictReader(f))[-1])
        same = rows[0]["val_loss"] == rows[1]["val_loss"]
        print(f"\n  run1 final val_loss {float(rows[0]['val_loss']):.8f}")
        print(f"  run2 final val_loss {float(rows[1]['val_loss']):.8f}")
        print(f"  bit-identical: {same}")

    elif a.stage == "floor":
        print("Seed-variance floor: arm C (gap3) x seeds 0,1,2. THIS is the floor.\n"
              "Reported as an observed RANGE -- an SD from n=3 implies precision that\n"
              "does not exist.")
        accs, f1s = [], []
        for s in (0, 1, 2):
            d = run_training("gap3", s, f"floor{s}", epochs=a.epochs)
            r = json.loads((d / "results.json").read_text())
            accs.append(r["test_acc"]); f1s.append(r["test_macro_f1"])
            print(f"  seed {s}: acc {r['test_acc']:.4f}  macro-F1 {r['test_macro_f1']:.4f}")
        floor = {"seeds": [0, 1, 2], "accs": accs, "f1s": f1s,
                 "range_acc": max(accs) - min(accs), "range_f1": max(f1s) - min(f1s)}
        print(f"\n  RANGE: acc {floor['range_acc']*100:.2f} pts   "
              f"macro-F1 {floor['range_f1']:.4f}")
        (REPO / "outputs" / "seed_floor.json").write_text(json.dumps(floor, indent=2))

    elif a.stage == "arms":
        for arm, head in ARMS.items():
            run_training(head, 0, f"abl{arm}", epochs=a.epochs)

    else:
        dirs = {}
        for arm, head in ARMS.items():
            g = sorted((REPO / "outputs" / "runs").glob(f"*_{head}_seed0_abl{arm}"))
            if not g:
                sys.exit(f"missing run for arm {arm} ({head}) -- run --stage arms first")
            dirs[arm] = g[-1]
        fp = REPO / "outputs" / "seed_floor.json"
        analyse(dirs, json.loads(fp.read_text()) if fp.exists() else None)


if __name__ == "__main__":
    main()
