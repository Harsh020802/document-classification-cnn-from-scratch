# Results

Every measured number, with the config and commit that produced it. **No bare accuracy
figures** — every result is stated against the measured noise floor.

**Status: Phase 1.2 steps 1-6 in progress; nothing trained yet.** Dataset, dedup and split
sections below are MEASURED. Training tables remain placeholders.

---

## Reporting rules

1. Seed spread reported as an **observed range** (min / max / range + raw per-seed values),
   never an SD — an SD from n=3 implies precision that does not exist.
2. Every arm difference printed **against the measured seed-variance floor**.
3. All detection thresholds at **k = 2.80** (1.96 + 0.84; 80% power, α=0.05 two-sided).
   One consistent k throughout.
4. **Inconclusive is a valid result** and is written as inconclusive.
5. Test-set sampling variance (**1.90%** at p=0.75, n=522) is **irreducible** by more seeds.

---

## 0. Environment (measured 2026-09-05)

| Item | Value |
|---|---|
| Machine | MacBook Air M2 (Mac14,2), 8 GB unified, 8 CPU / 8 GPU cores |
| OS | macOS 26.2 (Darwin 25.2.0), arm64 |
| Backend | MPS (Metal 4) |
| Python | 3.11.16 (conda env `docclf`) |
| torch / torchvision | 2.8.0 / 0.23.0 |
| Disk free at setup | 22.7 GB |
| Swap at setup | 5.21 / 6.00 GB |

**MPS op support (measured):** `AdaptiveAvgPool2d(p)` on a 14×14 input requires 14 % p == 0.
`p=1` OK · `p=3` **FAILS** · `p=7` OK. See D-024.

---

## 1. Model parameter counts — MEASURED 2026-09-06

From a real forward pass (`scripts/05_shape_walkthrough.py`), asserted in
`tests/test_shapes.py`.

**Backbone (4 blocks, 32→64→128→256): 1,172,640** — not the 1,175,232 estimated in Phase 0.4.
The estimate assumed conv layers carry a bias; they use `bias=False`, since BatchNorm
immediately subtracts the batch mean and cancels it (D-034).

| Arm | Head | Spatial | Head params | **Total** |
|---|---|---|---|---|
| A | `gap1` | none | 2,570 | **1,175,210** |
| B | `gap1_hidden(86)` | none | 22,972 | **1,195,612** |
| C | `gap3` | 3×3 | 23,050 | **1,195,690** |
| D | `gap7` | 7×7 | 125,450 | **1,298,090** |

**B vs C capacity match: 0.3384%** (test fails if it exceeds 1%). Rejected alternatives:
flatten head 501,770 (21×); VGG11's head ~119.6M (~5,000×).

### 1a. Real tensor shapes through the backbone

| stage | shape (N,C,H,W) | spatial | channels |
|---|---|---|---|
| input | (8, 1, 224, 224) | 224×224 | 1 |
| block1 | (8, 32, 112, 112) | 112×112 | 32 |
| block2 | (8, 64, 56, 56) | 56×56 | 64 |
| block3 | (8, 128, 28, 28) | 28×28 | 128 |
| **block4** | **(8, 256, 14, 14)** | **14×14** | 256 |

### 1b. The gap3 path, confirmed step by step

| step | shape |
|---|---|
| block4 output | (8, 256, 14, 14) — **14 % 3 ≠ 0** |
| `AdaptiveAvgPool2d(3)` on **cpu** | OK |
| `AdaptiveAvgPool2d(3)` on **mps** | **RAISES** |
| replicate-pad 14→15 | (8, 256, 15, 15) |
| `AvgPool2d(k=5, s=5)` | (8, 256, 3, 3) |
| flatten | (8, 2304) = 256 × 9 cells |

Correlation with the true op on real activations: **0.9964** (test requires > 0.99).

### 1c. Receptive field

**76 px theoretical after block 4 = 33.9%** of a 224 px input; effective RF ~30–45 px.
Progression 6 → 16 → 36 → 76. Each of the 9 pooled cells averages a 5×5 region of the padded
map ≈ **75×75 px** of input. Global layout therefore enters the model **from the pooling grid,
not the receptive field** — which is precisely why gap1 makes "letterhead at top"
unrepresentable.

---

## 2. Dataset — MEASURED 2026-09-05

| Metric | Expected | **Measured** |
|---|---|---|
| Total images | 3,482 | **3,482** ✓ |
| Classes | 10 | **10** ✓ |
| Imbalance (Memo:Resume) | 5.16× | **5.17×** (620:120) ✓ |
| Source resolution (median) | ~750×1000 assumed | **2386×2292** |
| Aspect ratio h/w (min/med/max) | ~1.29 assumed | **0.68 / 1.33 / 1.64** |
| PIL mode | — | **`L`** (already grayscale, 200/200 sampled) |

### 2a. Class distribution — measured vs the published table

Class weights use the **measured** column (D-028).

| Class | Published | **Measured** | Δ |
|---|---|---|---|
| Memo | 619 | **620** | +1 |
| Email | 593 | **599** | +6 |
| Letter | 565 | **567** | +2 |
| Form | 372 | **431** | **+59** |
| Report | 261 | **265** | +4 |
| Scientific | 255 | **261** | +6 |
| ADVE | 162 | **230** | **+68** |
| Note | 189 | **201** | +12 |
| News | 169 | **188** | +19 |
| Resume | 120 | **120** | 0 |
| **Total** | 3,482 | **3,482** | 0 |

### 2b. Near-duplicate detection (phash, union-find)

Threshold sweep — the cliff between 3 and 4 is transitive chaining through sparse
near-blank emails, not real duplicates (D-030):

| threshold | clusters | in-cluster | % | largest |
|---|---|---|---|---|
| 0–1 | 7 | 18 | 0.5% | 5 |
| **3 (adopted)** | **31** | **116** | **3.3%** | **25** |
| 5 (initial) | 39 | 256 | 7.4% | **121** |
| 6 | 50 | 411 | 11.8% | 278 |
| 8 | 79 | 609 | 17.5% | 415 |

Evidence the 121-cluster was an artifact: median pairwise distance *within* it was **14**
(≈3× the threshold), and its members averaged **0.018** ink density against **0.124** for the
rest of the dataset — 7× less ink.

**Adopted (threshold 3):** 31 clusters, 116 images (3.3%), largest 25, **1 cross-label
cluster**.

That cluster is **not label noise** — it is 5 "IMAGE NOT AVAILABLE ONLINE" archive placeholder
pages (ADVE ×4, News ×1) at phash distance **0** from each other. Images with no
class-relevant content, carrying labels for documents that were never digitised. A full-dataset
sweep found **exactly 5**, all placed in **train** (val 0, test 0). Kept, not removed. See
D-033.

### 2c. Split — stratified, near-duplicate clusters kept intact

| Class | Total | Train | Val | Test | tr% | va% | te% |
|---|---|---|---|---|---|---|---|
| ADVE | 230 | 161 | 35 | 34 | 70.0 | 15.2 | 14.8 |
| Email | 599 | 419 | 90 | 90 | 69.9 | 15.0 | 15.0 |
| Form | 431 | 302 | 65 | 64 | 70.1 | 15.1 | 14.8 |
| Letter | 567 | 397 | 85 | 85 | 70.0 | 15.0 | 15.0 |
| Memo | 620 | 434 | 93 | 93 | 70.0 | 15.0 | 15.0 |
| News | 188 | 132 | 28 | 28 | 70.2 | 14.9 | 14.9 |
| Note | 201 | 141 | 30 | 30 | 70.1 | 14.9 | 14.9 |
| Report | 265 | 185 | 40 | 40 | 69.8 | 15.1 | 15.1 |
| Resume | 120 | 84 | 18 | 18 | 70.0 | 15.0 | 15.0 |
| Scientific | 261 | 183 | 39 | 39 | 70.1 | 14.9 | 14.9 |
| **TOTAL** | **3,482** | **2,438** | **523** | **521** | **70.0** | **15.0** | **15.0** |

Every class within **0.2%** of target. **Leakage check passed** — no near-duplicate cluster
spans two splits.

**Test set n = 521**, so the power analysis (computed for 522) holds unchanged.

### 2d. Cache and normalization

| Metric | Predicted | **Measured** |
|---|---|---|
| Cache size (256×256 uint8) | 217.6 MB | **217.6 MB** ✓ exact |
| Cache build time | — | **37 s** (one-off) |
| Content geometry (`boxes.npy`) | — | **27.2 KB** (4 int16 per image) |
| Padding fraction of cached pixels | — | **23.4%** |

**Normalization (train split, n=2,438) — D-010 confirmed by measurement:**

| | mean | std |
|---|---|---|
| **Content pixels only** | **0.9351** | **0.1726** ← used |
| Including white padding | 0.9502 | 0.1535 |
| Difference | **+0.0152** | **-0.0190** |

Including padding would shift the mean toward white **and deflate the std by 11%**
(0.1726 → 0.1535). The std effect is the damaging one: dividing by an under-estimated std
under-normalizes the real document content. Mean 0.935 also confirms pages are ~94% white,
consistent with the ink-density finding from the dedup investigation.

**Class weights** (inverse frequency, normalised to mean 1.000): Resume **2.197** (highest)
down to Memo **0.425** (lowest) — a 5.2× ratio matching the imbalance.

**Dataset smoke test:** train 2,438 / val 523 / test 521; items are `(1, 224, 224)` float32;
augmentation verified random (max|Δ| 5.57 across two reads of the same index), eval verified
deterministic (max|Δ| 0.0000).

### 2e. Visual checks

| Check | Figure | Result |
|---|---|---|
| Squash vs pad (D-007) | `01_squash_vs_pad.png` | **PASS** — squashed row distorts each document differently; padded row preserves proportions. Sampled aspects 1.04–1.33, missing the 0.68 extreme |
| Augmentation fill (D-008) | `02_augmentation_fill.png` | **PASS** — `fill=0` shows black corner wedges at larger rotations; `fill=255` clean white at identical rotation. Verified by opening the image |
| Cross-label cluster (D-033) | `03_cross_label_cluster.png` | 5 archive placeholder pages, not mislabelled documents |

**Not a bug:** ~2.6% of ADVE images are sideways-scanned (0% in other classes). **0 of 498
sampled images carry an EXIF orientation tag**, so no metadata is being ignored — the scans
are genuinely rotated in the corpus (D-031).

---

## 3. Timing (to be measured)

Per-epoch seconds. `num_workers` = 0 vs 2, cache vs no cache — four cells, one experiment.

**MEASURED 2026-09-06** (20 batches per config, scaled to 77 batches/epoch):

| | workers=0 | workers=2 |
|---|---|---|
| **cached** | **33.7 s** | 89.9 s |
| no cache | 34.9 s | 88.3 s |

**The cache gives ~1.0× on this hardware** — 33.7 vs 34.9 s is inside noise. The GPU is not
the bottleneck: at 438 ms/batch the compute dominates so heavily that JPEG decode hides inside
it. D-009's performance justification was wrong; the cache is kept for reproducibility and
because the baselines read it (D-046).

**`num_workers=2` is 2.7× SLOWER**, not marginally worse — worker processes on an 8 GB machine
already in swap cost more than they save, with no blocking I/O left to overlap.

Actual training run: **36.8 s/epoch**, consistent with 33.7 s plus a validation pass.

**Revised expectation, set BEFORE running.** The Phase 0.4 estimate of 20–45 s/epoch assumed
~750×1000 source images. Measured median is **2386×2292 — roughly 7× the pixels per JPEG
decode**. So the *uncached* arm should come in far worse than 20–45 s/epoch. **That is the
expected result, not a problem**: it means the cache is doing considerably more work than
projected. The cached arm is the one that should land near the original estimate.

Flag if swap thrashes — it was at 5.21/6 GB before any run, and thrashing would silently
distort these numbers.

---

## 3b. Training-loop verification (Phase 1.4, measured 2026-09-06)

**Gradient lifecycle**, one weight tensor (`backbone[0].body[0].weight`, shape (32,1,3,3)):

| moment | `w.grad` | `w[0,0]` |
|---|---|---|
| before `backward()` | **None** | unchanged baseline |
| after `backward()` | populated, \|grad\| mean 9.957e-02, max 4.825e-01 | **max\|diff\| = 0.000e+00** |
| after `step()` | — | every element moved by exactly **3.000e-04 = lr** |

Backward computes derivatives and changes no weight. The uniform `lr`-sized first step is
characteristic of Adam: on step 1 the update reduces to `lr × sign(g)` regardless of gradient
magnitude. Optimizer state after one step: `step`, `exp_avg`, `exp_avg_sq` — two full-size
buffers per parameter, which is why a checkpoint is 14.4 MB not 4.8 MB.

**`zero_grad()` ablation** (identical seed, 3 steps):

| step | no zero_grad — loss | \|grad\| mean | with zero_grad — loss | \|grad\| mean |
|---|---|---|---|---|
| 1 | 2.3238 | 1.686e-01 | 2.3238 | 1.686e-01 |
| 2 | 1.9156 | 1.942e-01 | 1.9156 | 1.329e-01 |
| 3 | 1.4383 | **2.229e-01 ↑** | 1.3853 | **8.023e-02 ↓** |

Gradient magnitude *rises* without `zero_grad()` (accumulating the sum of all prior steps) and
*falls* with it (the model fitting the batch). Opposite directions by step 3.

**Single-batch overfit — PASS.** 8 images spanning 8 distinct classes (D-037):

| step | 1 | 5 | 10 | 25 | 50 | 300 |
|---|---|---|---|---|---|---|
| loss | 2.32377 | 0.73245 | 0.17043 | 0.00966 | 0.00118 | **0.00012** |
| acc | 0.00 | 0.62 | 0.88 | 1.00 | 1.00 | 1.00 |

Starting loss **2.324 ≈ ln(10) = 2.303**, as required for an untrained 10-class model.

---

## 4. Determinism check (diagnostic only)

Same seed, arm C, 3 epochs, ×2. Answers "is MPS bit-reproducible for our ops."
**This is not the noise floor.**

| Run | Final train loss | Final val loss | Bit-identical? |
|---|---|---|---|
| 1 | — | — | — |
| 2 | — | — | — |

---

## 5. Seed-variance floor — **the number everything is read against**

Arm C (`gap3`), seeds {0, 1, 2}, full runs.

| Seed | Test acc | Macro-F1 |
|---|---|---|
| 0 | — | — |
| 1 | — | — |
| 2 | — | — |
| **min / max / range** | — | — |

**Floor = observed range.** If it exceeds ~3 points the ablation likely cannot run — a
judgment call for the user, not an automatic gate.

---

## 5b. Baselines recomputed on the CORRECTED cache-space split

The cache-space dedup (D-043 fix) changed the split: 42 clusters / 206 images (5.9%) vs 31/116
in original space, and **0 test images within Hamming ≤3 of a train image** (was 13). Sizes are
unchanged at 2438/523/521, and **only 111 of 521 test images (21%) are shared with the old
split** — 410 moved out, 410 moved in. Scores across the two splits are therefore **not
directly comparable**.

| model | old split (orig-space) | **new split (cache-space)** |
|---|---|---|
| majority class | 17.85% | **17.85%** |
| logreg 8×8 | 58.73% (C=0.1) | **61.61%** (C=1.0) |
| logreg 32×32 | 61.61% (C=0.1) | **64.30%** (C=0.01) |
| 32×32 over 8×8 | +2.88 pts | **+2.69 pts** |

The 84.84% CNN figure was measured on the **old** split and stays labelled as such. The
ablation arms are all on the new split.

---

## 6. Baselines — the interpretability floor

All with `StandardScaler` (train-fit), `C` tuned on validation, `class_weight='balanced'`.

**MEASURED 2026-09-06**, sealed test split (n=521):

| Model | Features | Selected C | Test acc | Macro-F1 |
|---|---|---|---|---|
| Majority class (always Memo) | — | — | **17.85%** | **0.0303** |
| LogReg 8×8 | 64 | 0.1 | **58.73%** | 0.5491 |
| LogReg 32×32 | 1,024 | 0.1 | **61.61%** | 0.5599 |
| **CNN (arm C)** | — | — | *(running)* | *(running)* |

**The Phase 0.4 estimate of 30–45% for the linear baselines was wrong by ~20 points.** The CNN
target of 70–80% now sits only 8 points above the linear floor; 65% would be a failure, not a
marginal pass.

**16× more features buys +2.88%** (8×8 → 32×32). Both linear models are reading coarse ink
distribution by region, so the CNN's margin over **61.61%** is what demonstrates it learned
structure rather than density.

Majority-class macro-F1 **0.0303** against 17.85% accuracy — the case for macro-F1 as the
primary metric, in one row.

The CNN must clear LogReg 32×32 by a clear margin for the result to mean anything.
If 32×32 ≈ 8×8, both are using coarse ink density and the CNN's margin over *that* is what
shows it learned structure.

---

## 7. Ablation — four arms, seed 0, sealed 522-image test set

| Arm | Head | Spatial | Head params | Test acc | Macro-F1 | Δ vs floor |
|---|---|---|---|---|---|---|
| A | `gap1` | none | 2,570 | — | — | — |
| B | `gap1_hidden(86)` | none | 22,972 | — | — | — |
| C | `gap3` | 3×3 | 23,050 | — | — | — |
| D | `gap7` | 7×7 | 125,450 | — | — | — |

### 7-ABL. FOUR-ARM ABLATION — MEASURED 2026-09-06

Seed 0, sealed test split (n=521), **corrected cache-space split**. Only `model.head.type`
differs across arms — verified by diffing the effective runtime configs (exactly one key).

| arm | head | spatial | head params | test acc | macro-F1 |
|---|---|---|---|---|---|
| A | `gap1` | none | 2,570 | 0.7582 | 0.7529 |
| B | `gap1_hidden(86)` | none | 22,972 | **0.6775** | 0.7001 |
| C | `gap3` | **3×3** | 23,050 | **0.8560** | **0.8435** |
| D | `gap7` | 7×7 | 125,450 | 0.8253 | 0.8188 |

**Seed-variance floor: 2.11 accuracy points** (D-049). Every difference below is read against it.

### The headline comparison — A vs C (clean, both converged)

**+9.78 accuracy points from adding a 3×3 pooling grid — 4.6× the seed floor.**
McNemar A–C: b=16, c=67, p=1.4e-08. Both arms converged (best epochs 37 and 35), identical
configs. This is the result the writeup leads with because it needs no caveat.

Caveat that does apply: A and C are **not** capacity-matched (2,570 vs 23,050 head params), so
A-vs-C alone cannot separate spatial information from head capacity. Isolating that was arm B's
job — see below.

### The pre-registered primary — C vs B, reported as registered

| quantity | value |
|---|---|
| head params | 23,050 vs 22,972 — **0.34% apart** |
| b (C right, B wrong) | **112** |
| c (B right, C wrong) | **19** |
| discordant pairs | 131 |
| observed difference | **+17.85 accuracy points** |
| detectable at 80% power (k=2.80) | 6.15 pts |
| McNemar exact p | **3.108e-17** — reject H₀ |
| vs 2.11-pt seed floor | **8.5× the floor** |

**Caveat, stated in the same breath: arm B did not converge** (D-051). Its best epoch was 40 —
the last one — its train−val accuracy gap is **−0.022** (validation *better* than training,
i.e. underfitting), and its final train loss is 2.7× arm C's. **17.85 points is therefore an
upper bound on the architectural effect, not an estimate of it.** Under a longer budget the gap
would narrow.

The pre-registration is left intact rather than rewritten (D-048).

### Secondary comparisons (exploratory, Holm-corrected) — all five significant

| pair | b | c | p | Holm threshold | significant |
|---|---|---|---|---|---|
| B–D | 26 | 103 | 4.9e-12 | 0.0100 | yes |
| A–C | 16 | 67 | 1.4e-08 | 0.0125 | yes |
| A–B | 71 | 29 | 3.2e-05 | 0.0167 | yes |
| A–D | 28 | 63 | 3.1e-04 | 0.0250 | yes |
| C–D | 27 | 11 | 1.4e-02 | 0.0500 | yes |

### gap7 loses to gap3 — more resolution is not monotonically better

**gap3 0.8560 vs gap7 0.8253: a 3.07-point drop** for 5.4× the head parameters and 5.4× the
spatial cells (C–D: b=27, c=11, p=0.0139). 1.5× the seed floor — the smallest margin in the set
and the closest to it, but a real reversal of the naive "more spatial resolution is better"
reading. See D-053.

### Per-class F1 by arm — the Scientific test

| class | A (none) | B (none) | C (3×3) | D (7×7) | range |
|---|---|---|---|---|---|
| ADVE | 0.971 | 0.943 | 0.971 | 0.985 | 0.042 |
| Email | 0.899 | 0.919 | 0.944 | 0.927 | 0.045 |
| News | 0.929 | 0.964 | 0.929 | 0.893 | 0.071 |
| Form | 0.816 | 0.773 | 0.870 | 0.849 | 0.098 |
| Note | 0.618 | 0.793 | 0.712 | 0.706 | 0.175 |
| **Scientific** | **0.485** | **0.458** | **0.648** | **0.545** | **0.190** |
| Report | 0.609 | 0.482 | 0.737 | 0.673 | 0.255 |
| Resume | 0.791 | 0.667 | 0.895 | 0.944 | 0.278 |
| Letter | 0.686 | 0.538 | 0.840 | 0.827 | 0.301 |
| Memo | 0.726 | 0.465 | 0.891 | 0.838 | 0.427 |

**Scientific range 0.190 against its 0.048 per-class seed floor — 4× above it. D-044 is
REFUTED** (D-052): Scientific F1 moves substantially with spatial resolution, so it is not a
modality ceiling. Vision demonstrably helps; it just plateaus at 0.648 while other classes
reach 0.85+.

Nearly every class exceeds the 0.048 threshold, so spatial position matters broadly rather than
specifically for Scientific.

Figure: `fig6_ablation.png`

---

## 7a. Primary comparison (pre-registered, unadjusted α=0.05)

**C vs B** — matched capacity, spatial info the only difference.

| Quantity | Value |
|---|---|
| Discordant b (C right, B wrong) | — |
| Discordant c (B right, C wrong) | — |
| Total discordant | — |
| McNemar exact p | — |
| Detectable difference at this discordance (k=2.80) | — |
| Observed difference | — |
| Exceeds seed floor? | — |

**Two separate claims, reported separately:**
- **Model-level** (licensed by McNemar): —
- **Architectural** (licensed *only* if the effect exceeds the seed floor): —

### 7b. Secondary comparisons (exploratory, Holm-corrected)

Holm ladder: 0.0100 / 0.0125 / 0.0167 / 0.0250 / 0.0500

| Pair | b | c | p | Holm threshold | Significant? |
|---|---|---|---|---|---|
| A–B | — | — | — | — | — |
| A–C | — | — | — | — | — |
| A–D | — | — | — | — | — |
| B–D | — | — | — | — | — |
| C–D | — | — | — | — | — |

---

## 7b. FIRST FULL TRAINING RUN — measured 2026-09-06

`gap3`, seed 0, 40 epochs configured, **early-stopped at 26** (10 epochs without val
macro-F1 improvement). **16m 12s total, median 36.8 s/epoch** on M2 MPS.

| metric | value |
|---|---|
| best val macro-F1 | 0.7857 (epoch 16) |
| **test accuracy (full 521)** | **0.8522** |
| **test accuracy (508 clean)** | **0.8484** ← the honest figure |
| **test macro-F1 (508 clean)** | **0.8282** |
| test loss | 0.4729 |

**Against the floors:**

| model | test acc | macro-F1 |
|---|---|---|
| majority class | 17.85% | 0.0303 |
| logreg 8×8 | 58.73% | 0.5491 |
| logreg 32×32 | 61.61% | 0.5599 |
| **CNN gap3 (clean)** | **84.84%** | **0.8282** |

**+23.2 accuracy points over the best linear baseline**, and macro-F1 +0.269. The CNN is
clearly learning structure, not just ink density.

**Leakage check fired and found a real defect (D-043).** 85.22% crossed the >85% tripwire.
13 test images (2.5%) are within phash distance 3 of a training image *as the model sees them* —
the dedup hashed originals, not the 256×256 cache. All 13 predicted correctly. Excluding them:
**0.8484 / 0.8282**. Small, real, and reported as the headline number.

**Per-class, test split:**

| class | n | precision | recall | F1 |
|---|---|---|---|---|
| Email | 90 | 0.957 | **1.000** | **0.978** |
| ADVE | 34 | 0.971 | 0.971 | 0.971 |
| News | 28 | 0.926 | 0.893 | 0.909 |
| Letter | 85 | 0.897 | 0.824 | 0.859 |
| Form | 64 | 0.826 | 0.891 | 0.857 |
| Note | 30 | 0.889 | 0.800 | 0.842 |
| Memo | 93 | 0.832 | 0.849 | 0.840 |
| Report | 40 | 0.769 | 0.750 | 0.759 |
| Resume | 18 | 0.548 | 0.944 | 0.694 |
| **Scientific** | 39 | 0.704 | **0.487** | **0.576** |

macro-F1 **0.8286** vs accuracy **0.8522** — a 2.4-point gap, smaller than the 5–10 predicted,
so the class weighting worked better than expected.

**Resume**: recall 0.944, precision 0.548 — the weighting pushed it to over-predict Resume
(7 Memos misread as Resume). The intended trade, visible in the numbers.

**Scientific is the real weakness** and it is not fixable with vision (D-044): the class is
defined by *topic*, not layout.

**Error triage — top 30 most-confident errors, classified by inspection (D-045):**

| category | count | share |
|---|---|---|
| label error (model right, GT wrong) | **6** | 20% |
| genuinely ambiguous | **13** | 43% |
| model error | **11** | 37% |
| **not the model's fault** | **19** | **63%** |

Consistent with the published 11.7% label-error rate. Most of the remaining ~3 points to the
~88% ceiling is not closeable by better modelling.

**Most-confused pairs:** Memo→Resume (7), Scientific→Memo (5), Scientific→Form (5),
Scientific→Report (4), Report→Letter (4), Letter→Memo (4).

**correlation(class size, F1) = +0.362** — weak. Scientific (n=39) scores 0.576 while News
(n=28) scores 0.909, so definition matters more than size.

Figures: `04_confusion_matrix.png`, `05_confident_errors.png`, `06_training_curves.png`.

---

## 8. Final model — per-class breakdown

| Class | n (test) | Precision | Recall | F1 |
|---|---|---|---|---|
| Advertisement | — | — | — | — |
| Email | — | — | — | — |
| Form | — | — | — | — |
| Letter | — | — | — | — |
| Memo | — | — | — | — |
| News | — | — | — | — |
| Note | — | — | — | — |
| Report | — | — | — | — |
| Resume | — | — | — | — |
| Scientific | — | — | — | — |
| **Macro avg** | 522 | — | — | — |

Expectation: macro-F1 lands ~5–10 points **below** accuracy, because Resume (120 total,
~18 in test) underperforms. That gap is itself a finding.

Confusion matrix → `outputs/figures/confusion_matrix.png`
Most-confident errors → `outputs/figures/confident_errors.png`
(some of these are expected to be *label* errors, not model errors — 11.7% known rate)

---

## 9. Targets, for reference

| Marker | Value |
|---|---|
| Majority-class floor | ~17.8% |
| CNN target | 70–80% |
| Investigate if | <65% |
| Check leakage if | >85% |
| Label-noise ceiling | ~88% |
