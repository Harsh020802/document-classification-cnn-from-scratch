# Design Decisions

One entry per non-obvious decision, **newest first**. Reversals get a new entry; nothing is
deleted. Corrections in both directions are recorded — the wrong turns are the useful part.

Status values: `settled` · `open` · `revisited` · `reversed`

---

## D-028 — Actual class counts differ from published figures; use measured values
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled`

**Decision:** Class weights and all reporting use the **measured** counts from
`maveriq/tobacco3482`, not the figures quoted during Phase 0.2 research.

| Class | Literature (Phase 0.2) | **Measured** | Δ |
|---|---|---|---|
| Memo | 619 | 620 | +1 |
| Email | 593 | 599 | +6 |
| Letter | 565 | 567 | +2 |
| Form | 372 | **431** | **+59** |
| Report | 261 | 265 | +4 |
| Scientific | 255 | 261 | +6 |
| ADVE | 162 | **230** | **+68** |
| Note | 189 | 201 | +12 |
| News | 169 | 188 | +19 |
| Resume | 120 | 120 | 0 |
| **Total** | 3,482 | **3,482** | 0 |

Totals agree exactly, so the discrepancies are internal to the published table. Measured
imbalance **5.17×** (620/120) against the predicted ~5.16× — the design conclusions
(weighted loss, macro-F1 as primary metric, stratified split) are unaffected.

**Why it matters anyway:** inverse-frequency class weights are computed from these counts.
Using published numbers that are wrong for Form (+59) and ADVE (+68) would mis-weight two
classes by ~15–30%.

**Who raised it:** Claude, step-1 inspection. Recorded because the Phase 0.2 table was
presented with more confidence than a secondary source deserved.

---

## D-034 — Backbone is 1,172,640 params, not 1,175,232 (conv bias=False)
**Date:** 2026-09-06 · **Phase:** 1.3 · **Status:** `settled` — corrects **D-002**

**Measured** by `scripts/05_shape_walkthrough.py` and asserted in `tests/test_shapes.py`:

| Block | conv1 | bn1 | conv2 | bn2 | total |
|---|---|---|---|---|---|
| 1→32 | 288 | 64 | 9,216 | 64 | 9,632 |
| 32→64 | 18,432 | 128 | 36,864 | 128 | 55,552 |
| 64→128 | 73,728 | 256 | 147,456 | 256 | 221,696 |
| 128→256 | 294,912 | 512 | 589,824 | 512 | 885,760 |
| | | | | **Backbone** | **1,172,640** |

**Why the earlier figure was 2,592 too high:** the Phase 0.4 arithmetic assumed conv layers
carry a bias. They do not — `ConvBlock` sets `bias=False`. (Even with bias the total would be
1,173,600, so part of the original estimate was independently off.)

**Why `bias=False` under BatchNorm — the parameter is unlearnable, not merely redundant.**
A conv bias adds a per-channel constant `b`. BatchNorm's next operation is to subtract the
batch mean of its input:

    y = (x + b) - mean(x + b) = (x + b) - (mean(x) + b) = x - mean(x)

The `b` cancels exactly. It has **no effect on the output**, so its gradient is identically
zero and it can never learn anything — it is not a parameter the optimiser could use even in
principle. BN's own `beta` provides the per-channel offset instead, which is why BN layers
keep their bias while the convs preceding them drop it. Carrying conv biases here would mean
2,592 parameters that occupy memory, appear in the checkpoint, and consume gradient
computation while being mathematically incapable of changing the loss.

**Per-arm totals, measured:**

| Arm | Head | Head params | **Total** |
|---|---|---|---|
| A `gap1` | 1×1 | 2,570 | **1,175,210** |
| B `gap1_hidden(86)` | 1×1 | 22,972 | **1,195,612** |
| C `gap3` | 3×3 | 23,050 | **1,195,690** |
| D `gap7` | 7×7 | 125,450 | **1,298,090** |

**Capacity match B vs C: 0.3384%** — verified by a test that fails if it drifts above 1%.
The pre-registration is unaffected: head parameter counts (22,972 / 23,050) are exactly as
registered; only the shared backbone figure was wrong, and it is identical across all arms.

**Who raised it:** Claude, comparing the measured walkthrough output against the figure
carried since Phase 0.4. Found by printing real numbers instead of restating the estimate —
which was the point of the user's instruction to print shapes from a real forward pass.

---

## D-038 — Why real images correlate WORSE than noise: the per-cell probe
**Date:** 2026-09-06 · **Phase:** 1.4 · **Status:** `settled` — closes the open question in **D-036**

**The anomaly:** real cached images give **0.9860** correlation between `Pool3x3MPS` and
`AdaptiveAvgPool2d(3)`, while random-noise input gives **0.9962**. Real data correlating *worse*
contradicted the prediction (made by the user, unchallenged by Claude) that spatially smooth
real feature maps would agree *better*.

**Per-cell correlation, real images** (rows top/mid/bottom, cols left/centre/right):

```
  1.0000  0.9978  0.9741
  0.9983  0.9955  0.9727
  0.9905  0.9864  0.9684
```

vs noise:

```
  1.0000  0.9999  0.9950
  0.9999  0.9998  0.9948
  0.9945  0.9941  0.9887
```

**Location prediction: CONFIRMED.** Disagreement is not uniform — it degrades monotonically
toward the **bottom-right**, exactly where replicate-padding acts. Top-left is 1.0000 (those
cells never touch padded rows/columns); bottom-right is 0.9684. Centre 0.9955 vs border mean
0.9860.

**Mechanism prediction: REFUTED, and backwards.** The hypothesis was that real features carry
*more* energy near the frame edges. Measured, they carry **less**:

| | frame-edge mean | interior mean | ratio |
|---|---|---|---|
| **real** | 0.0494 | **0.1019** | **0.485** |
| noise | 0.0705 | 0.0736 | 0.958 |

| | col13 / col_mean | row13 / row_mean |
|---|---|---|
| **real** | **0.358** | 0.708 |
| noise | 0.986 | 0.942 |

**The actual mechanism — a spatial gradient, not edge energy.** Real document feature maps have
a steep edge-to-interior falloff (column 13 carries 36% of an average column's energy: the right
edge of a padded page is white margin). Noise maps are flat by construction (99%).

Replicate-padding duplicates column 13, so the rightmost pooled cell averages
`[10, 11, 12, 13, 13]` — **double-weighting an atypically empty column**. On a flat map,
duplicating any column changes almost nothing. On a map with a steep gradient, duplicating the
extreme column shifts that cell measurably.

**Correlation is scale-relative, which is the second half:**

| | mean abs error | spread (std) | **err / spread** |
|---|---|---|---|
| real | 0.01146 | 0.12105 | **0.169** |
| noise | 0.00404 | 0.08435 | **0.110** |

Real data has 2.8× the absolute error but only 1.4× the spread, so the ratio is worse — and the
ratio is what correlation measures.

**Consequences:** none requiring action. Border cells at 0.968–0.978 still track the intended op
closely, and the effect is a property of documents having white margins — real structure, not a
defect. Recorded because the reasoning was wrong on both sides and "the number is fine" is not
the same as "the explanation is right."

**Who raised it:** User noticed their own earlier prediction had failed and nobody had flagged
it, and specified the probe. The location hypothesis was right; the energy hypothesis was
backwards.

---

## D-040 — Resume test v1 measured dropout RNG drift, not checkpoint fidelity
**Date:** 2026-09-06 · **Phase:** 1.4 · **Status:** `settled`

**Decision:** `tests/test_checkpoint.py` reseeds to a fixed value **at the resume boundary in
every arm**, so dropout masks after the join are identical and optimizer state is the only
variable.

**What was wrong with v1:** it trained N steps, saved, resumed in fresh objects, trained N more,
and compared against an uninterrupted run. It failed — and the failure was the test's, not the
checkpoint's.

The model has dropout, which draws a **fresh random mask on every train()-mode forward**. The
interrupted and uninterrupted runs therefore consume different RNG streams and diverge from
that alone. Measured: two forwards on identical input and weights differ by **1.87** in
train() mode.

**The giveaway was in the output.** "full resume" and "weights only" produced *byte-identical*
curves (both `+0.01816` at the join). That is impossible if optimizer state were the cause —
it proved the divergence came from something both arms shared, i.e. the RNG.

So the test measured RNG drift and reported it as dropped optimizer state. A failing test that
fails for the wrong reason is the mirror image of D-037's passing test that could not fail:
both give a confident signal about something they are not measuring.

**Corrected result — the D-003 decision is validated:**

| arm | max deviation from uninterrupted |
|---|---|
| **full checkpoint** (weights + optimizer) | **0.000e+00** — bit-exact |
| weights only (`torch.save(model.state_dict())`) | **2.079e-02** |

Round-trip checks all pass: outputs identical to 0.000e+00, `exp_avg` and `exp_avg_sq` restored
exactly and non-zero, arch dict / epoch / monitor_best / scheduler lr all round-trip, and
loading a `gap3` checkpoint into a `gap1_hidden` model **raises** rather than silently mixing
ablation arms.

**Note on reading the numbers:** the weights-only arm shows *lower* loss just after the join
(0.0304 vs 0.0376). That is not an improvement — it is a fresh optimizer taking uncalibrated
large steps on a single memorised batch. On a real run it manifests as instability.

**Who raised it:** Claude, on seeing the two resume columns print identical values.

---

## D-043 — Dedup hashed the ORIGINALS; the model sees the CACHE. 13 test images leak.
**Date:** 2026-09-06 · **Phase:** 1.5 · **Status:** `settled` — a real gap in **D-011**, quantified

**Triggered by:** test accuracy of **85.22%** crossing the >85% leakage-check threshold agreed in
D-012. The check fired, and it found something.

**Clean:** zero index overlap between splits, zero shared phash groups, zero byte-identical
cached images across train/test.

**Not clean:** re-hashing the **cached 256×256 padded images** — the representation the model
actually sees — finds **13 test images (2.5%) within phash distance 3 of a training image, and
4 at distance 0** (identical as the model sees them).

**Root cause.** `scripts/02_dedup_and_split.py` hashes `ds[i]["image"]`, the original scan
(median 2386×2292). The model is trained on the 256×256 grayscale padded cache. Downscaling by
~9× destroys the fine detail that separated those pairs: the four distance-0 pairs sit 4–12 bits
apart *as originals* and collapse to identical hashes *as cached images*.

The dedup answered **"are these the same document?"** when the question that matters for leakage
is **"are these identical to the model?"** Those are different questions, and only the second
one governs whether a test score is inflated.

**Impact — measured, and small:**

| | n | acc |
|---|---|---|
| contaminated (dist ≤ 3) | 13 | **1.0000** |
| clean remainder | 508 | **0.8484** |
| contaminated (dist ≤ 0) | 4 | 1.0000 |
| clean remainder | 517 | **0.8511** |
| **full test set** | **521** | **0.8522** |

12 of the 13 pairs share a label, so the contamination is the kind that *can* inflate accuracy —
and the model got all 13 right. Removing them drops accuracy **0.8522 → 0.8484 (−0.38 pts)** and
macro-F1 **0.8286 → 0.8282 (−0.0004)**.

**Reported figure: 84.84% accuracy / 0.8282 macro-F1 on the 508 uncontaminated test images.**
Both numbers appear in RESULTS.md; the headline is the clean one.

**Fix for the ablation (not yet applied):** hash the cached images, not the originals, and re-run
the split. Deferred deliberately — re-splitting now would invalidate this run's comparability
with the baselines, which were computed on the same split. The four-arm ablation is next week's
work and will use the corrected split, with baselines recomputed against it.

**Who raised it:** the >85% threshold, which the user required in D-012 specifically as a
leakage tripwire. It fired on the first run that crossed it and found a genuine defect. Had the
threshold not existed, 85.22% would have been reported as-is.

---

## D-046 — Timing: the cache gives ~1.0x, and num_workers=2 is 2.7x SLOWER
**Date:** 2026-09-06 · **Phase:** 1.2 step 7 · **Status:** `settled` — **contradicts** the D-009 rationale

**Measured** (20 batches per configuration, scaled to 77 batches/epoch):

| | workers=0 | workers=2 |
|---|---|---|
| **cached** | **33.7 s** | 89.9 s |
| no cache | 34.9 s | 88.3 s |

**Prediction 1 — WRONG. The cache buys ~1.0x, not the large speedup D-009 assumed.**
33.7 s vs 34.9 s is a 3% difference, well inside measurement noise. The argument for caching
was that decoding median-2386×2292 JPEGs per image per epoch would starve the GPU. It does not,
because **the GPU is not the bottleneck** — at 438 ms/batch for a 32×1×224×224 batch through a
1.2M-parameter model, the compute itself dominates so heavily that JPEG decode hides completely
inside it. On a faster accelerator the cache would matter; on an M2 it does not.

The cache is still worth keeping for a different reason than the one given: it makes the
preprocessing deterministic and inspectable, and it is what `content_mean_std` and the baselines
read. But **the stated performance justification in D-009 was wrong**, and the honest version is
"negligible speedup on this hardware, kept for reproducibility."

**Prediction 2 — RIGHT, and by a much larger margin than expected.** `num_workers=2` is
**2.7× slower** (89.9 s vs 33.7 s), not marginally worse. On an 8 GB machine already ~5 GB into
swap, two forked worker processes each carrying a copy-on-write view of the cache cost far more
in memory pressure and IPC than they save in overlap — and with no blocking I/O left to hide,
there is nothing for them to overlap with in the first place. `num_workers=0` stands.

**Cross-check:** the fastest configuration measures 33.7 s/epoch; the actual training run
measured **36.8 s/epoch**, which includes a validation pass over 523 images. Consistent.

**Who raised it:** User required this be measured rather than predicted (D-009). Doing so
refuted half of my own reasoning — which is the argument for measuring.

---

## D-048 — PRE-COMMITMENT: one seed per arm. Seeds will NOT be added afterward.
**Date:** 2026-09-06 · **Phase:** 2 (ablation) · **Status:** `settled` — **committed BEFORE the arm runs**

**Decision:** the four ablation arms run at **seed 0 only**. Regardless of what the results
show, additional seeds will not be added.

**Why this is committed blind.** If we run four arms, see gap3 edge ahead, and *then* decide to
add seeds, the design has been conditioned on the data and the pre-registration stops meaning
anything. The decision has to be made while it is still uninformed. This entry is committed to
git before the runs start; the timestamp is the evidence.

**Why more seeds would not rescue the comparison — arithmetic, verified.**
From the measured noise floor (D-049), the observed range of test accuracy over n=3 seeds is
**2.11 points**. For n=3 the expected range is ≈1.693σ, so **σ ≈ 1.25 points**. The minimum
detectable difference between two arm means at 80% power (k = 2.80) is 2.80 × √2 × σ/√k:

| seeds per arm | SE(mean) | SE(difference) | MDD at 80% power | runtime |
|---|---|---|---|---|
| **1** | 1.25 pts | 1.76 pts | **4.9 pts** | ~75 min |
| 3 | 0.72 pts | 1.02 pts | **2.8 pts** | ~5 h |
| 5 | 0.56 pts | 0.79 pts | **2.2 pts** | ~8 h |
| 10 | 0.39 pts | 0.56 pts | 1.6 pts | ~17 h |

Against an expected effect of **2–3 points**, three seeds per arm reaches 2.8 and five reaches
2.2 — both still at or above the low end of the expected effect. **Five hours buys a
marginally-less-weak answer, not a strong one.** Ten seeds would get there, at ~17 hours on
this hardware, which is out of scope.

**What this means for the result.** The primary McNemar test remains worth running: it is a
paired test on the same 521 images and asks whether these two *fitted models* differ, which is
answerable. The *architectural* claim requires the effect to exceed the 2.11-point floor
(D-022). At one seed per arm, that is the standard being applied, and it is unlikely to be met.
An inconclusive outcome is recorded as inconclusive.

**Who raised it:** User, requiring the seed count be fixed before any arm result exists, and
supplying the power arithmetic. Claude verified each step independently: σ = 2.11/1.693 = 1.25,
and the MDD column reproduces exactly.

---

## D-049 — Measured seed-variance floor: 2.11 accuracy points
**Date:** 2026-09-06 · **Phase:** 2 · **Status:** `settled`

**gap3, seeds 0/1/2, full 40-epoch runs on the corrected cache-space split:**

| seed | test acc | macro-F1 | Scientific F1 |
|---|---|---|---|
| 0 | 0.8560 | 0.8435 | 0.648 |
| 1 | 0.8676 | 0.8544 | 0.629 |
| 2 | 0.8464 | 0.8312 | 0.600 |
| **observed range** | **2.11 pts** | **0.0232** | **0.048** |

Reported as a **range**, not a standard deviation — three points do not support an SD (D-021).

**This is the most important number produced by the project.** The same architecture, trained
on the same data with only the random seed changed, varies by 2.11 accuracy points. The
expected gap3-vs-gap1_hidden effect is 2–3 points. **The architecture differs from itself by
about as much as it is expected to differ from the comparison arm.**

Most published reports of a comparison like this state a 2-point win without ever measuring
what the architecture does against itself. That is the finding here, not a caveat on it.

**Consequence for the Scientific hypothesis (D-044):** Scientific F1 ranges **0.048** across
seeds of one architecture. Movement below 0.048 across the four arms is seed noise, not spatial
resolution. That threshold is applied rather than eyeballed.

**Determinism check (diagnostic only, not the floor):** gap3 seed 0 run twice for 3 epochs gave
**bit-identical** values at every epoch — train loss 1.7377017484129795, val macro-F1
0.5132755559004305, matching to the last digit. MPS is reproducible for these operations at a
fixed seed, so the floor above measures *pure seed variance* with no implementation
non-determinism mixed in.

---

## D-053 — gap7 loses to gap3: more spatial resolution is not monotonically better
**Date:** 2026-09-06 · **Phase:** 2 · **Status:** `settled`

**Measured:** gap3 (3×3 grid, 23,050 head params) scores **0.8560**; gap7 (7×7 grid, 125,450
head params) scores **0.8253**. A **3.07-point drop** for 5.4× the head parameters and 5.4× the
spatial cells. McNemar C–D: b=27, c=11, p=0.0139, significant under Holm at the 0.0500 rung.

The drop is 1.5× the 2.11-point seed floor, so it is unlikely to be seed noise, though it is
the smallest margin in the set and closest to the floor.

**Why this matters on its own.** The intuition behind the whole ablation was "spatial position
carries signal", and the naive extension of that is "more spatial resolution is better". It is
not. Going 1×1 → 3×3 gains 9.78 points; going 3×3 → 7×7 loses 3.07.

**Plausible reading, not verified:** 49 cells over-fragment a 14×14 feature map — each cell
averages a 2×2 patch, which is small enough that a document's layout elements straddle cell
boundaries inconsistently across images. 9 cells at roughly 75×75 input pixels each capture
top/middle/bottom × left/centre/right, which matches how documents are actually organised.
Testing that would need intermediate grids (4×4, 5×5), which is not in scope.

**Who raised it:** User, requiring this not be buried under the primary comparison.

---

## D-052 — D-044 REFUTED: Scientific is not beyond the reach of vision
**Date:** 2026-09-06 · **Phase:** 2 · **Status:** `reversed` — supersedes the hypothesis in **D-044**

**D-044 hypothesised** that the Scientific class (F1 0.576) was limited by *modality*: the class
is defined by subject matter rather than page geometry, so a CNN reading layout would have no
mechanism to represent it, and its F1 would stay flat at ~0.58 across all four arms regardless
of spatial resolution.

**Measured, and it does not hold:**

| arm | A gap1 (no spatial) | B gap1_hidden (no spatial) | C gap3 (3×3) | D gap7 (7×7) | range |
|---|---|---|---|---|---|
| Scientific F1 | 0.485 | 0.458 | **0.648** | 0.545 | **0.190** |

The pre-registered threshold was the seed-variance floor for this class, **0.048** (D-049).
The observed range is **0.190 — 4× the threshold.** Scientific F1 moves substantially with
spatial resolution: adding a 3×3 grid lifts it from 0.485 to 0.648, a 16-point gain.

**The hypothesis was wrong. Layout carries real signal for this class.**

**What survives, and what the Week 3 OCR case now rests on.** Scientific is still the worst
class by a wide margin — 0.648 against 0.85+ for Email, ADVE, Memo and Letter — and the best
available spatial configuration does not close that gap. The honest framing is **"vision helps
but plateaus well below the other classes"**, not "vision cannot help". The OCR argument stands
on the residual gap, not on an impossibility claim. **Drop "unreachable by vision" entirely.**

**Also notable:** nearly every class moves more than the 0.048 threshold across arms — Memo
0.427, Letter 0.301, Resume 0.278, Report 0.255. Spatial position matters broadly, not
specifically for Scientific. Scientific is simply the hardest class, not a categorically
different one.

**Who raised it:** the ablation, run precisely to test this. Recorded as a reversal rather than
edited away — the original claim in D-044 stays exactly as written.

---

## D-051 — Arm B did not converge: the primary comparison is an upper bound
**Date:** 2026-09-06 · **Phase:** 2 · **Status:** `settled`

**Trigger:** the primary comparison returned **+17.85 points**, roughly 6× the pre-registered
2–3 point expectation, and arm B (gap1_hidden) scored **8 points below** arm A (gap1) — adding a
hidden layer over the same 256-vector should cost little or nothing. Three diagnostics were run
before anything was written up.

**Check 1 — did B train? It trained, but UNDERFIT. My overfitting explanation is refuted.**

| arm | train loss start → end | train F1 | val F1 | train−val acc gap | best epoch |
|---|---|---|---|---|---|
| A gap1 | 1.978 → 0.650 | 0.774 | 0.788 | **−0.011** | 37 |
| **B gap1_hidden** | 2.195 → **0.869** | **0.661** | 0.688 | **−0.022** | **40 (the last)** |
| C gap3 | 1.738 → **0.319** | 0.901 | 0.833 | **+0.062** | 35 |

**B's validation score is higher than its training score.** A model that has memorised its
training set cannot do that. This is underfitting, the opposite of the explanation I offered
when first reporting the result. B's final train loss is 2.7× C's, it climbed monotonically
from 0.166 to 0.688, and **its best epoch was the last one** — it was still improving when the
budget ran out. A and C both peaked before the end; only B never plateaued.

It did not stall at ln(10)=2.3026, so it was learning — just too slowly to converge in 40
epochs.

**Check 2 — the head is alive, but a third of it is dead.** `Linear(256→86)` weights are normal
(|w| mean 0.026, max 0.163), post-ReLU activations have mean +1.58 and std 2.86, so signal
flows. But **61% of activations are zero and 35 of 86 units are dead for every test image** —
51 effective units, not 86. Not a collapse; a degraded head, consistent with the slow
convergence.

**Check 3 — the configs are identical.** Diffing the effective runtime configs across all four
arms found **exactly one differing key: `model.head.type`.** Same lr 3e-4, weight decay 1e-4,
40 epochs, batch 32, dropout 0.5, seed 0, identical normalisation constants. All four ran the
full 40 epochs; none early-stopped. The experiment is mechanically clean.

**Decision on how to report it.** The pre-registered primary (C vs gap1_hidden) is reported as
registered — **b=112, c=19, 131 discordant, +17.85 points, McNemar exact p = 3.1e-17** — with
the caveat stated in the same breath: **B did not converge, so 17.85 points is an upper bound
on the architectural effect, not an estimate of it.** Under a longer budget the gap would
likely narrow. The pre-registration is left intact rather than rewritten (D-048).

**The writeup leads with A vs C instead**, which needs no such caveat: both arms converged
(best epochs 37 and 35), identical configs, and the only difference is 1×1 versus 3×3 pooling.
**+9.78 points, 4.6× the seed floor.**

**Limitation this creates, stated plainly.** A and C are *not* capacity-matched — 2,570 versus
23,050 head parameters — so A-vs-C alone cannot separate spatial information from head
capacity. Isolating that was arm B's entire purpose, and B's non-convergence weakens it. What
still holds: **B has 9× A's head capacity and scores 8 points worse**, so capacity alone plainly
does not drive the gain. But the clean isolation the design intended is not available from
these runs.

**Who raised it:** User, refusing to accept a 6×-over-prediction result without ruling out
mechanical causes first — the same discipline that caught the leakage bug and the hashing
proxy, applied here where the surprise was in our favour.

---

## D-050 — Two anomalies checked before proceeding; both benign
**Date:** 2026-09-06 · **Phase:** 2 · **Status:** `settled`

**Anomaly 1 — early stopping never fired across three runs**, where the earlier run stopped at
epoch 26. Verified: `monitor` is still `"max val_macro_f1"`, `early_stop` is 10, the validation
loop ran all 40 epochs (40 rows, 40 distinct val losses — not a stuck value). The trajectory
climbs 0.5133 → 0.8460 and **peaks at epoch 35**, leaving only 5 epochs of patience consumed
against a limit of 10. **Early stopping correctly did not fire** because the model was still
improving near the end. Not a bug.

**Anomaly 2 — accuracy rose (0.846–0.868) on a split that should be marginally harder** than
the one giving 0.8484. Verified: split sizes are unchanged at 2438/523/521, all 3,482 images are
still present (the dedup *regrouped*, it did not drop images), and every class sits within
14.8–15.1% of its total in the test split, so stratification holds.

**Explanation: only 111 of 521 test images (21%) are shared between the old and new splits.**
410 images moved out and 410 moved in. These are substantially different test sets, so the
scores are **not directly comparable** and the difference needs no further explanation.
Recorded as benign; not theorised past what the check shows.

**Who raised it:** User, requiring both be checked before spending another 75 minutes of
compute.

---

## D-047 — Model confidence is well calibrated; 99.99% on a single image is not anomalous
**Date:** 2026-09-06 · **Phase:** 1.6 · **Status:** `settled`

The first `scripts/predict.py` run returned **ADVE at 0.9999** — high enough to suspect
something wrong. Checked against the full test set rather than assuming either way.

**Confidence distribution** (521 predictions): median 0.9126, mean 0.8274, **25.5% above 0.99**,
11.1% above 0.999.

**Reliability by confidence bin:**

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.00, 0.50) | 51 | 0.4073 | 0.3725 | +0.0347 |
| [0.50, 0.70) | 84 | 0.6046 | 0.6786 | −0.0740 |
| [0.70, 0.90) | 109 | 0.8239 | 0.9083 | −0.0843 |
| [0.90, 0.99) | 144 | 0.9517 | 0.9514 | +0.0003 |
| **[0.99, 1.00]** | **133** | **0.9974** | **0.9925** | +0.0049 |

**Expected calibration error 0.0343.** Good for an uncalibrated network, and the model is
slightly *under*-confident in the mid range rather than over-confident. In the ≥0.99 bin it is
right 99.25% of the time — the confidence is earned, not inflated.

The second test image (Scientific, dataset index 3331) returns **0.3391 / 0.3201 / 0.1824**
across Form / Memo / Email — near-uniform, and wrong. That is the correct behaviour for a class
the model cannot resolve from layout (D-044): it expresses uncertainty rather than a confident
error.

**Who raised it:** User asked that a suspicious-looking confidence be flagged. It looked
suspicious and turned out to be justified — which required checking rather than asserting.

---

## D-045 — Error triage: 63% of confident errors are not the model's fault
**Date:** 2026-09-06 · **Phase:** 1.5 · **Status:** `settled` — quantifies **D-044**

**Method:** of 77 test errors, the **30 the model was most confident about** were rendered and
classified by inspecting each image. Three buckets, pre-defined.

| category | count | share |
|---|---|---|
| **Label error** — model right, ground truth wrong | **6** | 20% |
| **Genuinely ambiguous** — a reasonable annotator could go either way | **13** | 43% |
| **Model error** — the model was wrong | **11** | 37% |
| **not the model's fault (L + A)** | **19** | **63%** |

**The six clear label errors:**

| # | model said | labelled | what the page actually is |
|---|---|---|---|
| 2 | Memo (0.99) | Form | "PHILIP MORRIS MANAG / FACSIMILE TRANSM" with DATE/TO/FROM/RE |
| 5 | ADVE (0.97) | Scientific | "WORKPLACE SMOKING BANS" poster with a no-smoking symbol |
| 6 | Form (0.94) | Report | R.J. Reynolds PROMOTION INFORMATION SHEET, tabular |
| 8 | Letter (0.91) | Report | headed **"Inter-office Memorandum"**, To/From/Subject/Date |
| 12 | Memo (0.83) | Letter | headed **"INTEROFFICE MEMORANDUM"**, TO/FROM/DATE/SUBJECT |
| 17 | Scientific (0.77) | Note | "MEMORANDUM TO: Committee of Counsel" |

**Why this matters more than the accuracy figure.** 20% label errors in this sample is
consistent with the published 11.7% dataset-wide rate (confident errors should be *enriched*
for label problems, since the model is most certain when the visual evidence is unambiguous).
Combined with 43% genuinely ambiguous cases, **most of the remaining 3-point gap to the ~88%
ceiling is not closeable by better modelling.**

That converts a vague caveat into a defensible sentence: of the errors the model is most sure
about, roughly two thirds are the dataset's problem, not the architecture's.

**The 13 ambiguous cases cluster on the same boundaries** the confusion matrix flags:
Scientific↔Form (a data table is both), Scientific↔Report (an interim laboratory report is
both), Report↔Letter (study summaries under a letterhead). These are not annotation sloppiness
— they are genuine category overlap in the label scheme.

**Who raised it:** User, asking that "at least 3 of 12" be replaced by a counted estimate over
a larger sample.

---

## D-044 — "Scientific" is topic-defined, not layout-defined: a vision-only ceiling
**Date:** 2026-09-06 · **Phase:** 1.5 · **Status:** `reversed` — **REFUTED by measurement, see D-052.** The hypothesis below is left exactly as written; the ablation moved Scientific F1 from 0.485 to 0.648 with spatial resolution, a 0.190 range against a 0.048 noise threshold.

**Observation:** Scientific has by far the worst per-class F1 (**0.576**, recall **0.487** — the
model finds fewer than half of them) and accounts for **6 of the 12 most-confident errors**.

**Looking at those images explains it.** The class contains a CONFIDENTIAL research letter, a
numeric data-summary table, a contract-research-centre cover sheet, a workplace-smoking poster,
handwritten lab notes, and a research proposal. They share **subject matter**, nothing else.
Every other class in this dataset is defined by *layout* — an email has a header block, a memo
has To/From/Date/Subject, a resume has section headings, an advert is a poster.

A CNN reading page geometry has no mechanism to recognise "this document is about science." The
failure is not a defect in the model; it is a **mismatch between the class definition and the
input modality**.

**STATUS: this is a HYPOTHESIS with a named test, not yet a finding.** The argument is
convincing but it is an argument. The four-arm ablation settles it for free, because the arms
already vary spatial resolution from none (gap1) to a 7×7 grid (gap7):

- **Scientific F1 flat at ~0.58 across all four arms** → the ceiling is **modality**, and the
  Week 3 OCR case is proven rather than argued.
- **Scientific F1 moves substantially with spatial resolution** → the story is more complicated
  and this claim needs weakening.

`scripts/12_ablation.py` therefore reports **per-class F1 for all four arms**, with Scientific
called out. Until that runs, `SUMMARY.md` states this as a hypothesis with the test named.

If confirmed: OCR plus a text branch is not a generic improvement to chase — it is the *only*
thing that can lift Scientific, because the distinguishing information is in the words and
absent from the layout.

**Correlation(class size, F1) = +0.362** — a mild positive, so small classes underperform
somewhat despite inverse-frequency weighting, but it is weak. Scientific (n=39) scores 0.576
while News (n=28) scores 0.909, so class *size* is clearly not the driver. Class *definition* is.

**Also visible in the confident errors — the model is right and the label is wrong.** At least
three of twelve: a page headed "Inter-office Memorandum" labelled Report (model said Letter), a
page headed "INTEROFFICE MEMORANDUM" with To/From/Date/Subject labelled Letter (model said Memo),
and a no-smoking poster labelled Scientific (model said ADVE). Consistent with the published
11.7% label-error rate, and the reason the ~88% ceiling is real.

**Who raised it:** Claude, from the confident-error panel — the bentrevett-derived analysis
(D-003) doing exactly what it was adopted for.

---

## D-042 — Baselines measured: the linear floor is 61.6%, not the estimated 30–45%
**Date:** 2026-09-06 · **Phase:** 1.5 · **Status:** `settled` — revises the targets in **D-012**

**Measured on the sealed test split (n=521):**

| model | features | C | test acc | macro-F1 |
|---|---|---|---|---|
| majority class (always Memo) | — | — | **17.85%** | **0.0303** |
| logistic regression 8×8 | 64 | 0.1 | **58.73%** | 0.5491 |
| logistic regression 32×32 | 1024 | 0.1 | **61.61%** | **0.5599** |

All with `StandardScaler` fit on train only, `C` tuned on validation over
{0.001, 0.01, 0.1, 1, 10}, `class_weight='balanced'`. Both linear models selected **C=0.1**,
i.e. fairly strong regularisation — consistent with 1024 features on 2,438 samples.

**The Phase 0.4 estimate of 30–45% was wrong by ~20 points.** This materially raises the bar:
the CNN target was 70–80%, so a linear model on 32×32 pixels is already within **8 points** of
the low end of that range. A CNN scoring 65% would now be a *failure*, not a marginal pass —
it would be beaten by five lines of scikit-learn.

**The 8×8 arm is the diagnostic, and it pays off.** 16× more features buys only **+2.88%**
(58.73% → 61.61%). Both models are therefore reading essentially the same signal: **coarse ink
distribution by region**. Documents are separable to ~59% by where the ink is, at 8×8
resolution, with no notion of structure at all.

So the CNN's margin over **61.61%** is the number that shows it learned *structure* rather than
*density*. That margin, not the raw accuracy, is the result worth reporting.

**Majority-class macro-F1 of 0.0303** against 17.85% accuracy is the clearest possible
demonstration of why macro-F1 is the primary metric: the accuracy figure looks non-trivial and
the model is worthless.

**Who raised it:** User required baselines before the CNN (D-013) and required them to be
properly regularised. Both calls were correct — an unregularised baseline would have scored
lower and flattered the CNN.

---

## D-041 — Eval-mode demos: measured cost of forgetting `eval()` and `no_grad()`
**Date:** 2026-09-06 · **Phase:** 1.4 · **Status:** `settled`

**`model.train()` vs `model.eval()`, identical batch and weights:**

| | max deviation across two calls |
|---|---|
| `train()` | **1.5492** — non-deterministic |
| `eval()` | **0.0000** — deterministic |
| train vs eval | 1.6419 |

Cost of forgetting `eval()` at validation time on an untrained model: correct accuracy 12.50%,
wrong-mode runs 0.00%, 0.00%, mean 7.50% over 5 draws. **Nothing errors.** The number is
quietly wrong and varies run to run — which would also make the four-arm ablation
non-reproducible, since every arm's val score would be a random draw.

Two causes: dropout zeroes a random 50% of features in train() and passes everything in eval();
BatchNorm uses *this batch's* statistics in train() (so a sample's output depends on its
batch-mates) and its running estimates in eval().

**`torch.no_grad()`, 32-image batch, measured on MPS:**

| | allocated while output is alive | driver reserved |
|---|---|---|
| without `no_grad` | **1569.2 MB** | 2098.7 MB |
| with `no_grad` | **0.0 MB** | 1074.7 MB |

The 0.0 MB is correct, not a measurement failure: under `no_grad` intermediates are freed as
soon as they are consumed, so only the 10.7 MB of weights remain. Loss value is identical
(2.299889 both ways); only `requires_grad`/`grad_fn` differ.

For scale, the four block *outputs* alone are 91.9 MB at this batch size — the remaining ~1.5 GB
is the ~5 further intermediates each block saves (2 conv, 2 BN, 2 ReLU, 1 pool), a **~17×
multiplier**. That is what backward needs and validation does not.

**Who raised it:** User, asking that `evaluate()` get the same treatment the training loop got.

---

## D-039 — Standing rule: measure on the real object, or label the proxy
**Date:** 2026-09-06 · **Phase:** 1.4 · **Status:** `settled` — standing rule, applies for the project

**The pattern, three occurrences:**

1. **D-025/D-026** — pooling correlation of 0.816 measured on `torch.randn`, a distribution the
   model never sees. Real activations give 0.9860. Understated by 17 points and nearly drove the
   wrong choice between two workarounds.
2. **D-031** — an EXIF orientation bug diagnosed from a sideways-rendering image, reported as the
   cause. Measured afterwards: **0 of 498 images carry an orientation tag**. The observation was
   real; the mechanism was invented.
3. **D-036** — a 0.9993 correlation reported from a hand-rolled backbone in a scratch script.
   The same input through the real `DocCNN` gives 0.9925. Did not reproduce.

**Same shape every time: measure something adjacent to the question, report it as the answer.**
Each was caught, two unprompted — but the catches were luck-adjacent, and the pattern is now a
fact about how this project goes wrong rather than three separate slips.

**Rule, effective now:**

1. **Measure on the real object.** The real model, the real data, the real device. A scratch
   reimplementation is not the model; `torch.randn` is not the dataset; CPU is not MPS.
2. **Where a proxy is genuinely necessary, label it at the point of reporting** — name the
   distribution or model it came from, in the same sentence as the number. Not in a footnote,
   not in the commit message.
3. **Before reporting a mechanism as the cause of an anomaly, verify the mechanism is present.**
   One command. D-031 would have taken thirty seconds to check and was stated as fact instead.

**FOURTH OCCURRENCE — and the first the rule would have prevented (D-043).**
`scripts/02_dedup_and_split.py` hashed the **original scans** (median 2386×2292) to find
near-duplicates, but the model trains on the **256×256 cached images**. At ~9× downscaling those
diverge: four test/train pairs sit at phash distance **0 in cache space** while being 4–12 bits
apart as originals. The dedup measured "is this the same document?" when the question governing
leakage is "is this identical to the model?"

This is the same shape as the other three — measure something adjacent to the question — but it
is the first instance that arrived **after** the rule was written, and the rule states exactly
the check that would have caught it in advance: *the real object* here is the cached tensor the
model consumes, not the source file.

Recorded as evidence the rule earns its place: it names a failure mode that recurs, and it would
have prevented an instance it predates.

**Who raised it:** User, after the third occurrence, asking that it be promoted from incident to
named finding. User also identified the fourth instance as belonging to the pattern.

---

## D-037 — Single-batch overfit test must span multiple classes
**Date:** 2026-09-06 · **Phase:** 1.4 · **Status:** `settled`

**Decision:** the sanity-check batch takes **one image from each of 8 different classes**, not
`train_ds[0..7]`.

**Why the naive version is close to worthless:** split indices are sorted, so the first eight
training rows are all class 0 (ADVE). The batch was `y = [0,0,0,0,0,0,0,0]`. "Memorise 8
images" then collapses to "always predict class 0" — a **constant function**, reachable by
pushing a single logit up and ignoring the input entirely. Loss reached 0.00021 in 5 steps.

That is a passing number from a test that verifies almost nothing. In particular it **cannot
detect misaligned labels**, which is one of the specific bugs the overfit check exists to
catch: if `x[i]` were paired with `y[j]`, a constant-output model would still score perfectly.

**With 8 distinct classes** the model must actually use the input, and the curve becomes a real
one: 2.324 → 0.732 (step 5) → 0.170 (10) → 0.0097 (25) → 0.00012 (300), accuracy 0.00 → 1.00.

**Incidental confirmation:** starting loss **2.324 ≈ ln(10) = 2.303**, which is what an
untrained 10-class classifier must produce if its outputs are near-uniform. A starting loss far
from ln(n_classes) is itself a signal that something is wrong with the loss or the label
encoding.

**Who raised it:** Claude, on seeing `y = [0,0,0,0,0,0,0,0]` printed in the batch line. The
demo had already "passed" before the problem was visible — which is the point: a green result
from a test that cannot fail is worse than no test.

---

## D-036 — Pooling-correlation figures pinned to a named reference distribution
**Date:** 2026-09-06 · **Phase:** 1.3 · **Status:** `settled` — corrects a number in **D-026**

**Decision:** `tests/test_shapes.py` asserts the Pool3x3MPS-vs-`AdaptiveAvgPool2d(3)`
correlation on **real cached training images**, with the reference distribution stated in the
test docstring and the threshold pinned to it.

**Why:** the correlation depends entirely on what is fed through the backbone, so the number is
meaningless without naming the input. Measured on this model:

| input distribution | correlation |
|---|---|
| `torch.randn` straight into the pool (not a feature map) | 0.8323 |
| random *input* through a random-init backbone | 0.9962 |
| **real cached document images through the same backbone** | **0.9860** ← asserted |
| synthetic document-like proxy | 0.9925 |

Random tensors *flatter* the number (0.9962) by producing an unrealistically smooth feature
map; raw noise fed directly to the pool understates it badly (0.8323) because it has no spatial
correlation. Real images give 0.9860, stable at **0.9851–0.9873 across 5 seeds**, so the 0.98
floor carries ~0.005 headroom.

**Correction to D-026: the reported 0.9993 does not reproduce.** That figure came from a
hand-rolled backbone in a scratch script, not from `DocCNN`. Re-running the identical synthetic
input through the real model gives **0.9925**, and real images give **0.9860**. The decision is
unaffected — every measurement is far above the 0.99-class agreement the choice relied on — but
it is the third time a number has been reported from a proxy rather than the real object
(after the random-noise correlation and the fabricated EXIF mechanism). The pattern is
consistent: measuring something adjacent to the question instead of the question.

**Who raised it:** User noticed `test_shapes.py` asserting 0.9964 against a previously reported
0.9993 and asked which input each measured.

---

## D-035 — Two bugs in the first shape-walkthrough script
**Date:** 2026-09-06 · **Phase:** 1.3 · **Status:** `settled`

Both found by reading the script's own output against known-correct values, and both fixed.

**Bug 1 — the MPS probe ran on CPU.** The script tested `AdaptiveAvgPool2d(3)` on the
CPU-resident feature tensor and printed "succeeded on this device", one line after correctly
stating it RAISES on MPS. Directly self-contradictory output. The restriction is
MPS-specific; a CPU probe silently succeeds and hides the entire reason D-026 exists. Now
probes every available device and prints each:

```
AdaptiveAvgPool2d(3) on cpu  : OK
AdaptiveAvgPool2d(3) on mps  : RAISES -- Adaptive pool MPS: input sizes must be divisible...
```

**Bug 2 — receptive field computed as 61 px instead of 76 px.** The loop applied
`r += (k-1)*jump` for the two convs but only advanced the jump for the maxpool, **omitting the
pool's own `(2-1)*jump` contribution**. Max pooling widens the receptive field as well as
subsampling. Corrected loop reproduces the verified progression 6 → 16 → 36 → **76**, i.e.
**33.9%** of a 224 px input, matching D-006.

Had this gone unnoticed it would have understated the receptive field by 20% in the one
explanation where the number is load-bearing.

**Who raised it:** Claude. Both were visible only because the user required real printed
output rather than annotated comments — a comment claiming "RF = 76" would have stayed right
while the code was wrong.

---

## D-033 — The one cross-label cluster is 5 "IMAGE NOT AVAILABLE" placeholder pages
**Date:** 2026-09-06 · **Phase:** 1.2 · **Status:** `settled`

**Answer to "did any cluster span two labels at threshold 3?": yes, exactly one.** Cluster of
5: indices 68, 78, 94, 207 (ADVE) and 2571 (News).

**They are not documents.** Rendered
(`outputs/figures/03_cross_label_cluster.png`), all five read:

> **IMAGE NOT AVAILABLE ONLINE**
> *The material referenced in the associated index listing is available in the Minnesota
> Tobacco Document Depository. Please see this website's home page for additional information
> regarding the Depository.*

These are **archive placeholder pages**. The underlying scans were never digitised, and the
depository substituted a notice.

**This is irreducible error, not label noise — a distinct and more interesting concept.**
The published 11.7% error rate concerns documents given the *wrong* class: a correctable
annotation problem. These five are a **Bayes-error floor inside the dataset**. The pixels are
identical across two different labels, so *no function of the input* can separate them. The
optimal classifier — with unlimited capacity and unlimited data — is wrong on at least one of
these by construction. That is a property of the data, not of any model.

It sits alongside the ~88% label-noise ceiling as a **second, different reason** achievable
accuracy is bounded below 100%: one reason is that labels are wrong, the other is that the
information required to be right is absent from the input.

**Contrast with D-030, same tool and opposite outcomes:** here, **5 images at phash distance
0** — genuinely identical, correctly grouped. There, **121 images with median internal distance
14** — chained through near-blank pages, an artifact. The diagnostic that separates them is the
median *within-cluster* distance relative to the threshold, not the cluster size or count.

**Scope — measured, and contained.** Sweeping the full dataset against one placeholder as
template: **exactly 5 images at phash distance 0** (byte-identical content), and the count does
not grow out to distance 10. So there are five, not a hidden population.

| Metric | Value |
|---|---|
| Placeholder pages in dataset | **5** (0.14%) |
| Labels | ADVE ×4, News ×1 |
| phash distance to template | **0** for all five |
| Split placement | **train 5, val 0, test 0** |

**Decision: keep them, do not remove.** They are 0.2% of the training split, the grouped split
already placed all five in train so none contaminate val or test, and removing rows would
introduce a deviation from the published dataset for negligible benefit. Recorded so that if
they appear in error analysis they are recognised rather than rediscovered.

**Why phash found them and this is a success, not a failure:** unlike the D-030 chaining
artifact (median internal distance 14 on near-blank pages), these are genuinely identical —
distance 0. The dedup did exactly what it is for.

**Who raised it:** User asked explicitly whether any cluster spanned labels at threshold 3.
Rendering the cluster rather than reporting the count is what revealed they were placeholders.

---

## D-032 — Cache stores content geometry, not a per-pixel validity mask
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled` — refines **D-010**

**Decision:** `boxes.npy` holds `(left, top, w, h)` int16 per image — **27.2 KB** — instead of
a full uint8 mask array at **217.6 MB**.

**Why:** the first implementation wrote a per-pixel mask the same size as the image cache,
doubling cache disk from 218 MB to 435 MB. The mask is entirely determined by the paste
rectangle, so it was 218 MB encoding four numbers per image. On a machine with 16 GB free
that is worth not wasting.

`content_mean_std` now streams sums and sums-of-squares over each box rather than
materialising a boolean-indexed array, which also removes a large temporary allocation.

**Verified identical after the refactor:** mean 0.9351, std 0.1726 — unchanged to 4 decimals.

**Measured, confirming D-010:** padding is **23.4%** of cached pixels. Including it shifts the
mean +0.0152 toward white and **deflates the std by 11%** (0.1726 → 0.1535). The std effect is
the more damaging of the two, exactly as argued when D-010 was written.

**Who raised it:** Claude, on seeing `masks.npy` at 218 MB in the build output.

---

## D-031 — Visual checks passed; no EXIF bug (an inference I got wrong)
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled`

**Step 5, the `fill=255` check — PASSED, verified by looking at the image, not the log line.**
`outputs/figures/02_augmentation_fill.png` renders the same augmented document twice per
column, top row `fill=0` and bottom row `fill=255`. At the larger rotation angles the top row
shows **large black triangular wedges** across the corners; the bottom row is clean white at
the identical rotation. This is D-008 confirmed empirically rather than argued. The
near-zero-rotation column matches across both rows, which is the correct control.

**Step 4, squash vs pad — PASSED.** `01_squash_vs_pad.png` shows the middle (squashed) row
stretching each document by a different amount while the bottom (padded) row preserves
proportions. Caveat: the four images sampled span aspect 1.04–1.33, missing the 0.68 landscape
extreme where the effect is most dramatic. The point is made but the figure is not ideal.

**A wrong inference, recorded because the reasoning failure is the useful part.** The
augmentation panel's document renders sideways. I inferred that `resize_and_pad` was ignoring
EXIF orientation (PIL's `resize`/`convert` do not apply it without `ImageOps.exif_transpose`)
and stated that as a real bug affecting arm C's spatial signal.

**Measured: 0 of 498 sampled images carry a non-trivial EXIF orientation tag.** There is no
EXIF metadata being ignored, because there is none. The document is a **genuinely
sideways-scanned advertisement** — placed on the scanner rotated. Landscape images are 2.6% of
ADVE and 0.0% of every other class.

I jumped from an observation ("displayed sideways") to a mechanism ("EXIF is being ignored")
without checking the mechanism was present. The observation was right and the diagnosis was
invented. Same failure shape as the D-025 correlation measured on random noise: asserting a
cause instead of measuring it.

**What survives:** a small fraction of ADVE scans are rotated 90°, which adds genuine
orientation variance to the spatial signal arm C measures. A property of the corpus, not a
preprocessing defect. At ~2.6% of one class it needs no fix — recorded so it is not
rediscovered as a bug later.

**Minor, unfixed:** the corner-region boxes in the panel are drawn at 46px assuming a 224px
render, but the panel shows the 256px cached image. Indicative, not exact.

**Who raised it:** User's standing instruction to actually open the augmentation visual rather
than trust a "generated successfully" line. Looking at it confirmed the fix and surfaced the
sideways scan; measuring rather than assuming then corrected my own diagnosis of it.

---

## D-030 — phash threshold lowered 5 → 3: union-find chained sparse documents
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled` — revises **D-011**

**Decision:** `hamming_threshold: 3`, not 5. Union-find clustering kept.

**What went wrong at threshold 5:** the largest "near-duplicate" cluster contained **121
images** (3.5% of the dataset), labelled Email 119 / Letter 1 / News 1. Inspected visually:
they are **different emails**, not one document rescanned — different senders, subjects and
body text, sharing only a layout.

**Diagnosis — chaining, not similarity.** Median pairwise phash distance *within* the cluster
is **14**, nearly 3× the threshold. Most member pairs are not within threshold of each other;
the cluster formed transitively (A≈B, B≈C, … ) through union-find until images 30 bits apart
sat in one component.

**Mechanism — ink density.** Cluster members average **0.018** non-white pixels against
**0.124** for the rest of the dataset — **7× less ink**. phash thresholds DCT coefficients
against their median; on a near-blank page most coefficients sit near that median, so bits
flip on scanner noise and every sparse document hashes alike. It was detecting *emptiness*,
not duplication.

**Threshold sweep (measured):**

| thr | clusters | in-cluster | % | largest |
|---|---|---|---|---|
| 0–1 | 7 | 18 | 0.5% | 5 |
| **2–3** | **31** | **116** | **3.3%** | **25** |
| 4–5 | 39 | 256 | 7.4% | **121** |
| 6 | 50 | 411 | 11.8% | 278 |
| 8 | 79 | 609 | 17.5% | 415 |

The cliff between 3 and 4 (largest 25 → 121) is the chaining onset. Threshold 3 sits below
it. Runaway growth to 278 and 415 at higher thresholds confirms the diagnosis — real duplicate
counts do not grow like that.

**Composition of the discarded 7.4%:** the chained cluster alone was 121 images (3.5%); the
other 38 clusters together were 135 (3.9%). **About half the reported near-duplicate rate was
this single artifact.**

**Cost of leaving it:** not leakage — over-merging is the conservative direction. But 121
images is **20% of all 599 Emails** locked into one split, so val/test Emails would come from
a systematically different subpopulation, and "7.4% near-duplicates" would be a wrong number
to report.

**Options rejected:** capping cluster size (treats the symptom, leaves the threshold wrong);
requiring mutual similarity instead of transitive closure (most principled — directly fixes
non-transitivity — but O(n²) clique search for a problem a config value solves); keeping it
and documenting (accepts a known-distorted split).

**Correction recorded:** when writing `src/data/dedup.py` Claude noted union-find "may
over-merge, which is the safe direction for leakage" and treated that as costless. It is not
costless — it over-merged 121 images and would have skewed the Email class. The note
identified the risk and then dismissed it.

**The transferable finding is the mechanism, not the threshold value.** phash is unreliable on
sparse documents. It computes a DCT and thresholds each coefficient against the median; on a
page that is ~98% white, most coefficients sit near that median, so which side of it they fall
is decided by scanner noise rather than content. Every sparse document then hashes alike.
Measured here as **0.018 ink density in-cluster vs 0.124 elsewhere — 7×**. Any perceptual-hash
dedup on document corpora should expect this, and the diagnostic is the median *within-cluster*
distance: if it substantially exceeds the threshold, the cluster is chained, not similar.

**Who raised it:** Claude found it while inspecting the dedup output, and looked at the actual
images rather than accepting the cluster count. User chose the threshold, and asked that the
mechanism be recorded rather than only the new value.

---

## D-029 — Aspect ratios span 0.68–1.64; some documents are landscape
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled` — reinforces **D-007**

**Measured** (first 200 images): width 817 / **2386** / 3296 px, height 1089 / **2292** /
4192 px, aspect h/w **0.68 / 1.33 / 1.64** (min/median/max). All sampled images are PIL
mode `L`.

**Three consequences:**

1. **D-007 (pad-to-square) is more strongly justified than when it was argued.** The plan
   assumed ~1:1.29 portrait. The median is 1.33, close — but the **range** is 0.68 to 1.64,
   a **2.4× spread**, and **0.68 means genuinely landscape documents**. Squashing to square
   would distort these by wildly differing amounts, which is precisely the per-image-varying
   noise D-007 exists to remove.
2. **D-005 (grayscale) confirmed:** all sampled images are already mode `L`. The
   `.convert("L")` in `resize_and_pad` is a no-op safety net, not a conversion. The
   "scans are effectively grayscale already" reasoning holds.
3. **Source images are far larger than assumed** — median 2386×2292 against an assumed
   ~750×1000. This makes the cache (D-009) more valuable, not less: decoding a 2386×2292
   JPEG per image per epoch would badly starve the GPU.

**Who raised it:** Claude, step-1 inspection.

---

## D-027 — Stratified split moved before the cache build
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled`

**Decision:** Phase 1.2 order is download → dedup → **split** → visuals → cache+stats →
timing. The split moves from step 6 to step 3.

**Why:** `default.yaml` says `normalize.compute_from: train`, but the original plan built the
cache and computed normalization stats at step 4, with the split at step 6. At step 4 no
training split exists, so stats would have been computed over all 3,482 images — **val and
test pixel statistics leaking into the normalization applied to training data.**

The effect on accuracy would be small. It contradicts the config, and it is the same class of
leak the sealed test set and train-only `StandardScaler` (D-013) exist to prevent. Being
careful about leakage in the protocol and casual about it in the pipeline is not a coherent
position.

The visuals also move ahead of the cache build: if padding looks wrong, better to find out
before generating 217 MB.

**Who raised it:** User.

---

## D-026 — `gap3` = replicate-pad 14→15, then `AvgPool2d(kernel_size=5, stride=5)`
**Date:** 2026-09-05 · **Phase:** 1.2 · **Status:** `settled` — supersedes **D-025**

**Decision:** Arm C pools by replicate-padding the 14×14 map to 15×15, then
`nn.AvgPool2d(kernel_size=5, stride=5)`. Arms A/B/D keep exact `AdaptiveAvgPool2d`.

**Why D-025 was wrong — the reasoning that matters:** D-025 was chosen by comparing the two
workarounds against each other, without asking *which arms need a workaround at all*. Only
arm C does — 14 is divisible by 1 and 7, so A, B and D use the exact op. That made the
**pre-registered primary comparison (B vs C) the one comparison where the two arms are
implemented differently.** After spending four rounds removing a confound from the ablation
design, D-025 reintroduced one at the implementation layer.

Concretely, `AvgPool2d(k=5,s=4)` gives cells `[0-4][4-8][8-12]`: **row and column 13 unused,
columns 4 and 8 double-counted.** For documents the bottom edge carries signature blocks and
footers — precisely the spatial signal arm C exists to test. A loss for C could not be
separated from "C was denied the bottom 7% of the page."

**Fix (a), `PYTORCH_ENABLE_MPS_FALLBACK=1`: tested, does NOT work.** Identical failure with
and without the flag. The flag routes *unimplemented* ops to CPU; this op **is** implemented
and raises an explicit `RuntimeError` for non-divisible inputs, so the fallback path never
engages. A manual CPU round-trip works but costs 1.206 ms/batch vs 0.264 — 4.6× — and was
already rejected for making arm C's timing incomparable.

**Fix (b), adopted.** Measured on this machine:

| | D-025 (`k=5,s=4`) | **D-026** (pad→15, `k=5,s=5`) |
|---|---|---|
| Coverage | 13/14, col 13 dropped | **15/15, full** |
| Double-counted | cols 4, 8 | **none** |
| Cells | `[0-4][4-8][8-12]` overlapping | `[0-4][5-9][10-14]` uniform |
| corr vs `AdaptiveAvgPool2d(3)` (real activations) | 0.9993 | 0.9993 |
| Cost | 0.264 ms/batch | 0.410 ms/batch |
| Params | 23,050 | **23,050** |
| MPS fwd/bwd | OK | **OK** |

Cost of the fix: **+0.146 ms/batch = +0.011 s/epoch**. Eleven milliseconds to remove a
confound from the primary comparison.

**Caveat recorded:** the pad is `replicate`, so index 14 duplicates index 13 — row/col 13
carries double weight in the last cell. A mild edge emphasis, but it *includes* the bottom
edge rather than discarding it. Zero-padding was rejected: it would inject artificial dark
values into a white-page region.

**Correction recorded — the 0.816 figure in D-025 was measured on `torch.randn`.** Random
tensors have no spatial correlation, so windowing differences look maximal. Re-measured on
real block-4 activations from an initialized backbone fed document-like input (measured
adjacent-column smoothness **0.991**), the correlation is **0.9993**, mean |diff| 0.0001.
Benchmarking a pooling operator on white noise tests it on the one input distribution it will
never see. The corrected number does not change the decision — it makes clear the D-025
concern was about *coverage*, not numerical fidelity.

**Who raised it:** User spotted the arm asymmetry, proposed both fixes, and questioned what
the correlation was measured on. All three were right. Claude fixed the blocker without
checking whether the fix was symmetric across arms.

---

## D-025 — `gap3` implemented as `AvgPool2d(kernel_size=5, stride=4)`
**Date:** 2026-09-05 · **Phase:** 1.1 · **Status:** `reversed` — superseded by **D-026**

**Decision:** Arm C's 3×3 pooling is `nn.AvgPool2d(kernel_size=5, stride=4)`, not
`nn.AdaptiveAvgPool2d(3)`. Arms A (`gap1`) and D (`gap7`) keep `AdaptiveAvgPool2d`, which
works on MPS because 14 is divisible by 1 and 7.

**Measured on this machine** (torch 2.8.0, MPS), 14×14 input:

| option | output | cols used | coverage | corr. with true `AdaptiveAvgPool2d(3)` | MPS |
|---|---|---|---|---|---|
| `AvgPool2d(k=5, s=4)` | 3×3 | 0–12 | **92.9%** | **0.816** | OK |
| `AvgPool2d(k=4, s=5)` | 3×3 | 0–3, 5–8, 10–13 | 85.7% | 0.755 | OK |
| `AdaptiveAvgPool2d(3)` | 3×3 | — | 100% | 1.000 | **FAILS** |

Cells are `[0–4] [4–8] [8–12]` — a 1-column overlap between neighbours, and column 13 unused.

**Why this option:** it covers more of the feature map (13/14 vs 12/14) and correlates better
with the intended op. The overlap is not a defect — `AdaptiveAvgPool2d` on a non-divisible
input *also* produces uneven, overlapping windows, which is exactly why MPS refuses to
implement it. Option 2's clean partition costs 2 of 14 columns, a worse distortion than a
1-column overlap. Changing the input to 192 was rejected because 12 is not divisible by 7,
which merely moves the blocker to arm D and invalidates every size and timing figure computed
so far. A CPU fallback for the pool alone was rejected because a per-batch device transfer
inside the forward pass makes arm C's timing incomparable to the other three.

**Effect on the pre-registration:** none numerically. Output is 3×3 → 2,304 features →
`Linear(2304→10)` = **23,050 params**, unchanged, so the B-vs-C capacity match (22,972 vs
23,050, 0.3%) still holds. Logged **before** any training run, per D-020/D-022.

**Caveat to carry into the writeup:** arm C is now "3×3 average pooling with 92.9% coverage
and mild overlap," not textbook adaptive pooling. If arm C loses, this is a confound to name
— though a *small* one, since correlation with the intended op is 0.816 and the pooled cells
still encode top/middle/bottom × left/centre/right.

**Correction recorded:** Claude initially described this option as using "all 14 rows and
columns." It uses 13 of 14 — column 13 is dropped. Measured, not assumed.

**Who raised it:** Claude found the blocker in the environment smoke test and recommended
this option; user chose it.

---

## D-024 — `AdaptiveAvgPool2d(3)` does not run on MPS
**Date:** 2026-09-05 · **Phase:** 1.1 · **Status:** `reversed` — resolved by **D-025**

**Decision:** Superseded by **D-025** (`AvgPool2d(kernel_size=5, stride=4)`). Original entry kept below as the record of the blocker.

**What happened:** Environment smoke test raised
`RuntimeError: Adaptive pool MPS: input sizes must be divisible by output sizes.`
Block-4 emits a 14×14 feature map. 14 is divisible by 1, 2, 7, 14 — **not 3**.

Measured on this machine (torch 2.8.0, MPS):

| pool | 14 divisible? | MPS |
|---|---|---|
| `AdaptiveAvgPool2d(1)` | yes | OK |
| `AdaptiveAvgPool2d(3)` | **no** | **FAILS** |
| `AdaptiveAvgPool2d(7)` | yes | OK |

So arms A, B, D run; **arm C — half the pre-registered primary comparison — does not.**

**Options considered:**
1. `AvgPool2d(kernel_size=5, stride=4)` → verified 3×3 output, works on MPS. Overlapping
   windows (5>4), so cells overlap slightly rather than partitioning cleanly.
2. `AvgPool2d(kernel_size=4, stride=5)` → verified 3×3 output, works on MPS. Non-overlapping
   with a 2px gap; drops 2 of 14 rows/cols.
3. Change input 224 → 192, giving a 12×12 map (divisible by 3). But 12 is **not** divisible
   by 7, so this breaks arm D instead. Also changes every other measurement.
4. CPU fallback for the pooling op only — correct results, but a per-batch device transfer
   inside the forward pass, and it makes arm C's timing incomparable to the others.

**Why it matters beyond a workaround:** whichever option is chosen alters what "3×3 spatial
pooling" means, and arm C is the pre-registered primary. The change must be made and logged
**before** any run, or the pre-registration is compromised.

**Who raised it:** Claude, during the Phase 1.1 environment smoke test. Not anticipated by
either party during Phase 0 — the plan assumed `AdaptiveAvgPool2d` was portable. It is not.

---

## D-023 — Standing requirement: living project journal
**Date:** 2026-09-05 · **Phase:** 1.1 · **Status:** `settled`

**Decision:** Maintain `DECISIONS.md`, `JOURNAL.md`, `RESULTS.md`, `SUMMARY.md` under
`notes/`, tracked in git, for the remainder of the project (Weeks 1–5).

**Why:** Most of this project's value so far is reasoning, not implementation, and that
reasoning lived only in a chat log. Two audiences, both the user: the version of him opening
this repo cold in four months, and the version writing a CV entry or answering "walk me
through this project" in an interview.

**Who raised it:** User, `notes/reply-06.md`.

---

## D-022 — McNemar tests two fitted models, not two architectures
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Restate the primary hypothesis to match what the test actually measures, and
gate the architectural claim separately.

> **H₁:** the trained `gap3` model at seed 0 achieves higher test accuracy than the trained
> `gap1_hidden(86)` model at seed 0.
> **Scope:** a claim about these two fitted models, not about the architectures.
> **Architectural generalization** is supported only if the McNemar effect exceeds the
> measured seed-variance floor.

**Why:** McNemar conditions on two specific trained models and treats the *test images* as
the random variable. The original H₁ ("spatial position carries class signal independent of
head capacity") is a claim about *architectures*, which requires treating the *training run*
as the random variable. Different populations. Worse, the protocol measured seed variance in
step 2 and then conditioned it away in step 5 — incoherent. A significant McNemar at seed 0
could reflect one lucky initialization.

Dietterich (1998), *Approximate Statistical Tests for Comparing Supervised Classification
Learning Algorithms*, recommends McNemar precisely when only one training run per algorithm
is affordable, while noting it does not capture training variability. That is our situation
including the caveat.

The honest alternative — seeds as the unit of replication, 3+ per arm, paired across seeds —
is ~4 hours and blows the weekend. Not done; recorded in the README as the correct design
for a properly-powered architectural claim.

**Who raised it:** User. Claude wrote the mismatched hypothesis and did not notice that
step 2 and step 5 were measuring different things.

---

## D-021 — Seed floor measured on `gap3`, reported as a range
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited` (supersedes the arm-A plan)

**Decision:** Measure seed variance on **arm C (`gap3`)**, an arm in the primary comparison,
across seeds {0,1,2}. Report the **observed range**, not a standard deviation.

**Why:** Cost is essentially identical — the backbone (1,175,232 params) dominates and the
head differs by ~20k, under 2% of the model — so there is no reason to accept a proxy plus a
caveat when the real thing is free. An SD from n=3 implies precision that does not exist.

**Correction recorded:** Claude originally proposed arm A and wrote that its likely-lower
variance made the floor *"conservative in the direction of making differences look more
significant."* That is **anti-conservative** — an underestimated floor inflates apparent
effects. Getting the word backwards in a protocol whose whole purpose is not fooling
ourselves is worse than a vocabulary slip: it describes a bias toward false positives and
labels it caution.

The ~3-point kill switch is a **judgment call, not a computed threshold** — the range and a
read are reported, and the user decides.

**Who raised it:** User, both the design change and the terminology error.

---

## D-020 — Primary comparison sits outside the Holm family
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Primary (`gap3` vs `gap1_hidden(86)`) at **unadjusted α=0.05**. The 5 remaining
pairs form the exploratory family, Holm ladder `0.0100 / 0.0125 / 0.0167 / 0.0250 / 0.0500`.

**Why:** 4 arms → 6 pairs; at α=0.05 each, P(≥1 false positive) = 1 − 0.95⁶ = **26.5%**.
But pre-registering a primary and then Holm-correcting it across all 6 is having it both
ways — the entire benefit of pre-registration is that the primary is a single planned test
not subject to multiplicity. Holm over Bonferroni: uniformly more powerful at the same FWER
and assumes no independence, which matters since the tests share arms.

**Correction recorded:** Claude listed a 6-wide ladder starting at 0.05/6 while
simultaneously arguing the primary sat outside the family. Direct self-contradiction.

**Who raised it:** User.

---

## D-019 — Seed averaging cannot reduce test-set sampling variance
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Report the variance decomposition explicitly; never present more seeds as a
route to significance.

| Component | Source | Reducible by seeds? |
|---|---|---|
| Test-set sampling | 522 finite images | **No — irreducible** (1.90% at p=0.75) |
| Seed / init variance | weights, shuffle, dropout | Yes, by 1/√k |

**Why:** The same 522 images are evaluated by every seed; there is no resampling of the test
set. The distinction is the difference between *"run more seeds until it's significant"*
(p-hacking with extra compute) and *"this test set has a floor we cannot cross"* (the truth).

**Correction recorded:** Claude claimed 3 seeds would bring the interval to ±3.03 by applying
1/√3 to the **total** SE. Wrong — that SE is test-set sampling variance, which seeds cannot
touch. The incorrect table would have made "run more seeds" look like a valid next move.

**Who raised it:** User.

---

## D-018 — One consistent k throughout the power analysis
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** All detection thresholds at **k = 2.80** (= 1.96 + 0.84, 80% power at α=0.05
two-sided).

Corrected McNemar table:

| Discordant n | Detectable difference (k=2.80) |
|---|---|
| 30 | 2.9 pts |
| 50 | 3.8 pts |
| 80 | 4.8 pts |
| 120 | 5.9 pts |

Like-for-like: unpaired **7.5 pts**, paired **3.4–4.8 pts** at expected discordance →
**1.5–2× better**, not the 2.5× first claimed.

**Counterintuitive property worth recording:** more discordant pairs means a **larger**
detectable difference, since SD(b−c) = √n_disc while the 522 denominator is fixed
(20 disc → 2.4 pts; 320 disc → 9.6 pts). High disagreement means both models are individually
noisy and their disagreements are mostly coin-flips. **Low discordance is good news.** Arms B
and C share a backbone, so low discordance is expected — favourable.

**Correction recorded:** Claude argued ±5.25 was a confidence interval rather than a detection
threshold, then built the McNemar table at k=1.96 anyway — inconsistent, and in the direction
that flattered the proposed change, overstating the benefit by ~40%.

**Who raised it:** User, who reconstructed the table at both k values to locate the error.

---

## D-017 — McNemar's paired test instead of comparing independent proportions
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Compare arms with McNemar's exact binomial test on the same 522 test images.

**Why:** Both arms are evaluated on the *same* test set, so the comparison is paired.
Treating them as independent (√2 × SE) discards that. McNemar looks only at discordant pairs
— images where exactly one model is right — and everything both get right or both get wrong
carries no information about which is better. Free power, zero extra compute.

**Who raised it:** User raised the power problem; Claude identified that the runs were paired
and that McNemar was the appropriate test.

---

## D-016 — The ablation may be underpowered; inconclusive is a valid outcome
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Report the power analysis alongside every ablation number. If the result cannot
resolve the question, write it up as *"this dataset cannot resolve the question at this
effect size"* rather than presenting a decisive-looking table.

**Arithmetic:** test set = 15% × 3,482 = **522**. At p=0.75, SE = **1.90%**. Unpaired
SE_diff = 2.68%, 95% CI ±5.25 pts, and **detection at 80% power needs 7.5 pts**. The expected
GAP(1)-vs-GAP(3) effect is plausibly 2–3 points. **Underpowered by roughly 2.5×.**

**Who raised it:** User. Claude's initial framing treated ±5.25 as if it were the detection
threshold.

---

## D-015 — Pooling val+test for evaluation: proposed, then withdrawn
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `reversed`

**Decision:** Rejected. The test set stays sealed at 522 images.

**Why:** Pooling would halve SE to 1.34%, but `monitor: max val_macro_f1` means validation
*selects* the checkpoint. Scoring a model on data used to choose it is optimistically biased
— and biased **differently per arm**, depending on how much each overfits selection. That
systematic error contaminates exactly the comparison being made, and is worse than the random
error it removes. McNemar reaches further (~3.4–4.8 pts) without the bias.

Unbiased alternative for more evaluation data: 5-fold stratified CV on train+val with test
sealed (~2,960 eval samples, no selection bias) — 5× compute per arm. Noted for Week 2.

**Who raised it:** User proposed it, then withdrew it after Claude's objection.

---

## D-014 — Four ablation arms with a matched-capacity control
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited` (supersedes the 3-arm design)

**Decision:**

| Arm | Head | Spatial | Head params |
|---|---|---|---|
| A | `gap1` | none | 2,570 |
| B | `gap1_hidden(86)` | **none** | **22,972** |
| C | `gap3` | **3×3** | **23,050** |
| D | `gap7` | 7×7 | 125,450 |

**B vs C is the experiment**; A and D are context.

**Why:** The original 3-arm design (GAP 1/3/7) varied spatial resolution **and** head capacity
together, perfectly correlated. Two hypotheses — "spatial position helps" and "more head
capacity helps" — predict the *same* rise-then-fall curve. **The experiment had no
discriminating power at all**: three training runs that could not answer the question they
were built to ask.

**Arithmetic:** exact capacity match solves to h = 86.29. **h=86 → 22,972 params, 0.3% from
gap3's 23,050**; h=88 → 23,506, **+2.0%**. Since the arm exists to control for capacity, the
tighter value wins.

**Caveat recorded up front:** arm B has an extra ReLU that C lacks — matched on parameters,
not depth. Unavoidable: capacity cannot be added to a 256-vector without a layer. This gives
B a slight representational edge, so a B≈C tie *strengthens* the capacity explanation, and a
C win holds *despite* the handicap.

**Who raised it:** User identified the confound. Claude designed the confounded version and
conceded it was a non-experiment dressed as one. User proposed h=88; Claude computed h=86.

---

## D-013 — Three trivial baselines, run before the CNN
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** (1) majority class (~17.8%), (2) LogReg on 32×32 pixels, (3) LogReg on 8×8.
All with `StandardScaler` fit on **train only**, `C` tuned on **validation**, and
`class_weight='balanced'`. Selected `C` reported.

**Why:** Without a floor, "68%" is uninterpretable. The regularization matters: 1024 features
on ~2,400 samples overfits at sklearn defaults, and **a weak baseline is worse than no
baseline** — it gives false confidence that the CNN learned something. `class_weight='balanced'`
mirrors the CNN's weighted loss so the comparison isn't rigged in the CNN's favour. The 8×8
arm is near-pure ink-density-by-region: if 32×32 barely beats it, the CNN must clear a
density-only floor to have learned structure.

Majority class also gets its macro-F1 reported (~0.03) — a vivid demonstration of why
accuracy alone lies on imbalanced data.

**Who raised it:** User (the baselines, and the regularization requirement).

---

## D-012 — Literature comparison dropped; targets from first principles
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited`

**Decision:** Keep 70/15/15 and drop the published-baseline comparison.

| Marker | Value |
|---|---|
| Majority-class floor | ~17.8% |
| LogReg on raw pixels | ~30–45% (to be measured) |
| **CNN target** | **70–80%** |
| Investigate if | <65% |
| Check leakage if | >85% |
| Label-noise ceiling | ~88% |

**Why:** Published Tobacco-3482 baselines use ~100 images/class (~800 train); 70/15/15 gives
~2,400 — 3× the data, not comparable. The stated goal is understanding, not benchmark
placement, and the 100-per-class protocol would discard ~1,600 real images. D-013's baselines
provide a *better* reference point on our own split.

**Correction recorded:** Claude's original ">80% would be suspicious" was benchmarked against
the wrong protocol. **Counter-correction:** the user's "I may clear 80% legitimately" was also
overstated — published ~85% figures come from ImageNet-pretrained backbones, and more
in-domain data does not fully substitute for pretrained low-level features. User accepted this.

**Who raised it:** User caught the invalid comparison; Claude pushed back on the inference
drawn from it. Both were partly wrong.

---

## D-011 — Near-duplicate detection via perceptual hashing, before splitting
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** `imagehash.phash`, Hamming ≤ 5, run on raw images **before** splitting.
Cluster with **union-find**. Report cluster sizes with visual examples; flag clusters spanning
multiple class labels separately.

**Why:** Exact file hashing catches almost nothing in a scanned corpus — the same document
rescanned differs in noise, skew, and JPEG quantization, so different bytes, identical
content. Running before the split is what makes prevention possible rather than just
detection. Union-find because near-duplicate relations are **not transitive** (A≈B, B≈C does
not give A≈C); it may over-merge, which is the safe direction for leakage. Cross-label
clusters are either label errors or genuinely ambiguous documents — both interesting given
the 11.7% known label-error rate.

`imagehash` added to the allowed list: the list exists to prevent transfer-learning past the
hard parts, not to force hand-rolling a DCT. It pulls scipy, which sklearn already requires.

**Who raised it:** Claude proposed a duplicate check; user specified the method after noting
"check for duplicates" was underspecified.

---

## D-010 — Normalization stats over content pixels only
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Compute mean/std on the **train** split over **content pixels only**, using a
cache-time validity mask. Cache to `outputs/norm_stats.json`; apply identical values to
val/test. Print padded and unpadded stats side by side.

**Why:** Padding is white (255) and **the amount varies per image with aspect ratio**. Including
it pulls the mean toward white by a per-image-varying, class-irrelevant amount — and worse,
**deflates the apparent variance of real content**, so dividing by that std under-normalizes
the actual document pixels.

**Who raised it:** User, catching the interaction between two separately-correct decisions
(train-split stats, and white padding) before it was implemented.

---

## D-009 — Cache at 256, augment and crop to 224 at load time
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited`

**Decision:** Cache long-side-256, white-padded to 256×256, uint8 (**217.6 MB** measured).
Train: random crop → 224. Val/test: deterministic center crop → 224.

**Why:** Caching at 224 then augmenting means two interpolation passes, the second on
already-downscaled data — detail lost in pass one cannot return. Cost is 256²/224² = **1.306×**
(166.6 → 217.6 MB), trivial in 8 GB. The random crop is **free translation jitter** (±32 px),
and real scans genuinely vary in page position.

**Tension recorded:** random cropping is a translation augmentation, and arm C's premise is
that *absolute* spatial position carries signal — so jitter partially erodes what arm C tests.
Kept anyway: a spatial win *despite* jitter is the stronger, more honest result, and a model
needing pixel-exact registration would be overfit to a scanner artifact.

Deterministic center crop for eval is required, or the metric isn't reproducible across arms.

**Who raised it:** User.

---

## D-008 — `fill=255` on every geometric transform
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** A single module-level `PAD_VALUE = 255` used by both the pad **and** every
geometric transform's `fill`, applied in pixel space pre-normalization.

**Why:** torchvision's `RandomRotation`/`RandomAffine` default to `fill=0` — black. Against
white padding that paints **maximum-contrast black wedges** into the corners, and wedge area
scales with |rotation angle|, which is resampled every epoch. So it is random, class-irrelevant
noise injected into the corners — which are **4 of the 9 gap3 spatial cells**. Left unfixed it
would bias the ablation *against* the spatial hypothesis, since arm C's cells get polluted
while arm B's single pooled value averages it away.

One constant, not two copies: two places holding the same magic number will drift.

**Who raised it:** User. Claude specified white padding but missed the transform-fill default.

---

## D-007 — Pad to square instead of resizing to 224×224
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** Resize long side, white-pad short side to square. Rectangular 224×288 rejected.

**Why:** Tobacco-3482 pages are ~1:1.29 portrait; a direct square resize compresses ~29%. The
damage is not the distortion itself — a CNN can learn a consistently distorted space — but
that **the distortion factor varies per image with source aspect ratio**, injecting noise on
exactly the geometric signal GAP(3) exists to read. White matches document background, so
padding reads as "more margin," a variation real scans already show; black would create a hard
artificial edge for filters to latch onto.

224×288 rejected: 1.29× compute and activation memory (likely forcing batch 32→24), and a
14×18 feature map would make the GAP(3) grid non-square.

**Who raised it:** User. Claude's plan did not address aspect ratio at all.

---

## D-006 — Head is `AdaptiveAvgPool2d(3)`, not GAP(1)
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited` — see **D-024**, MPS blocks this

**Decision:** `AdaptiveAvgPool2d(3)` → 2,304 features → `Linear(2304→10)` = **23,050 params**.

**Why:** GAP(1) averages away every spatial coordinate. The original plan justified the design
with "document class is a global layout property" and then chose the one head that makes
global layout **unrepresentable** — a direct self-contradiction. A memo and a letter both
contain a header block, body text, and whitespace; under GAP(1) they differ only in the
*proportion* of local features, not their arrangement.

**Receptive field, verified:** 76 px theoretical after 4 blocks = **33.9%** of a 224 px input;
effective RF is smaller still (~30–46 px, 14–20%, per Luo et al. 2016 √-scaling). So a block-4
neuron sees a *patch*, not a page — the claim it "sees a substantial region" was wrong.
**Global aggregation comes from the pooling, not the receptive field**, which is precisely why
deleting spatial information at the pool is costly.

Cost: 23,050 vs GAP(1)'s 2,570 — and **21× smaller** than the 501,770-param flatten.

**Who raised it:** User identified the contradiction and computed the receptive field. Claude
confirmed 76 px and conceded the "substantial region" claim was wrong.

---

## D-005 — Grayscale input (right decision, corrected reasoning)
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `revisited`

**Decision:** 1-channel grayscale input.

**Original justification (WRONG):** "cuts the first layer's parameters by 3×." Measured: 3→32
channels at 3×3 is 864 params vs 288 for 1→32 — a **576-param difference out of 1.2M, 0.05%**.
Meaningless.

**Actual reasons:** (1) memory and dataloader throughput — 3× less data per image, which is
the real win on an 8 GB machine already 5 GB into swap; (2) Tobacco-3482 scans are effectively
grayscale already, so RGB would triplicate identical information; (3) no class signal in colour
— a memo and a letter are both black text on white paper.

**Who raised it:** User. Both agreed the decision was right and the reasoning was not.

---

## D-004 — Params-per-training-image heuristic dropped
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `reversed`

**Decision:** Removed, not replaced.

**Why:** The original "~350 params per image" divided 1.2M by all 3,482 images rather than the
~2,400 training images (~490). But the arithmetic was the smaller problem — the ratio has **no
theoretical basis**. Model capacity does not scale linearly with parameter count; effective
capacity depends on architecture, regularization, and data structure. It was an intuition pump
that pumps the wrong intuition.

**Who raised it:** User caught both the arithmetic and that the heuristic was not meaningful.

---

## D-003 — Engineering conventions from the reference repos
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** bentrevett's transparency wrapped in victoresque's durability; neither one's
abstraction layer.

**Adopted from victoresque** (`pytorch-template`):
- Checkpoint dict — `state_dict` + optimizer state + `monitor_best` + embedded config.
  **Changed:** store the full arch **config dict**, not victoresque's bare class name — every
  ablation arm is `DocCNN`, so a class-name check would never fire and the arms would be
  indistinguishable. This is what makes the ablation's checkpoints self-describing.
  Optimizer state matters: AdamW's moments, if dropped, cause a visible loss spike on resume.
- `monitor: "max val_macro_f1"` string convention; timestamped run dirs with a config snapshot.

**Adopted from bentrevett** (`pytorch-image-classification`):
- Flat `train()` / `evaluate()` functions with the five steps visible in sequence.
- Config-list → layer builder, so depth is a config value.
- `count_parameters` at startup; confident-wrong error analysis; sklearn confusion matrix.

**Rejected from victoresque:** `BaseTrainer`/`BaseModel` inheritance (the training loop is the
thing being learned; abstraction hides it behind 3 layers of indirection); `init_obj` reflective
instantiation (no IDE navigation, typos become deep runtime errors); TensorBoard (CSV is
diffable across arms, one less dependency); `n_gpu`/`DataParallel`; `log_softmax` + `nll_loss`;
`save_period: 1`.

**Rejected from bentrevett:** transfer learning (violates the core constraint — most of
notebook 4); the VGG11 head (**~119.6M params**, verified, vs our 23k — **~5,000×**; designed
for ImageNet's 1.2M images, would memorize 2,400 instantly); `RandomHorizontalFlip`; ImageNet
normalization constants; `LRFinder` (deferred to Week 2); the `copy.deepcopy` split trick
(**not stratified** — unacceptable at 5.2× imbalance).

**Who raised it:** User supplied the repos and set the boundary.

---

## D-002 — Architecture: 4-block VGG-style CNN, ~1.2M params
**Date:** 2026-09-05 · **Phase:** 0.4 · **Status:** `settled`

**Decision:** `[Conv3×3 → BN → ReLU] ×2 → MaxPool(2)`, channels **32 → 64 → 128 → 256**,
backbone **1,175,232** params, grayscale 224×224 input, raw logits out.

**Why VGG over ResNet:** no skip connections to explain away; every layer's purpose visible.
Residuals become a Week 2 controlled experiment — a better way to learn what they do than
starting with them.

**Why these choices:** 4 blocks takes 224→112→56→28→14, doubling channels each time; 3×3
kernels stacked in pairs match one 5×5's receptive field with fewer params and an extra
non-linearity; MaxPool is parameter-free and "is there a strong edge here" is the right
question for documents; BatchNorm permits higher learning rates and acts as a mild regularizer
(and is what YOLOv8's Conv+BN+SiLU block was doing all along); Dropout 0.5 is aggressive,
appropriate for a small dataset.

**Governing constraint:** ~2,400 training images, no pretrained weights. The dominant risk is
**overfitting**, not underfitting. Small model, strong regularization, aggressive augmentation.

**Who raised it:** Claude proposed; user verified the parameter arithmetic independently.

---

## D-001 — Scope: Tobacco-3482, from scratch, understanding over benchmark
**Date:** 2026-09-05 · **Phase:** 0.2 · **Status:** `settled`

**Decision:** Document-image classifier on **Tobacco-3482** (3,482 images, 10 classes), written
from scratch in PyTorch. No pretrained weights, no transfer learning, no one-liner wrappers.

**Options considered:** RVL-CDIP (400k images, 16 classes) — **38.8 GB against 21 GB free**,
out on disk before time even enters. DocLayNet / PubLayNet / FUNSD / SROIE / CORD — detection
and extraction tasks, deferred to Weeks 2–4.

**Why:** ~3.5k images means 20–60 s/epoch on M2 MPS and a full run in under 30 minutes — the
right regime for *learning*, because short runs mean you can break something and see the result
before losing the thread. Real scanned business documents, directly relevant to the user's work,
not a generic tutorial dataset. Extends into detection/OCR/fusion without rewrites.

**The 5.2× class imbalance is a feature** (Memo 619 → Resume 120): it forces learning why raw
accuracy misleads and why per-class P/R and a confusion matrix are what you actually read.

**Recorded caveat — label noise:** [Label Errors in the Tobacco3482 Dataset](https://arxiv.org/abs/2412.13140)
(WACV 2025) finds **11.7%** improperly annotated, **16.7%** with multiple valid labels, and
**35%** of a top model's errors attributable to label problems. This sets a **~88% ceiling**.
Stated up front so a 75% plateau isn't misread as a broken architecture. Data quality caps
model quality — no architecture change rescues bad labels.

**Compute:** local M2, free. Kaggle Notebooks (30 GPU-hr/week, P100) as the free fallback.
No paid GPU needed — renting an H100 for a 1.2M-param model on 3.5k images would leave the GPU
idle waiting on the dataloader.

**Who raised it:** User set the domain and constraints; Claude researched and recommended.
