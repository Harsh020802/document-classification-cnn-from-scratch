# Document Image Classifier — Summary

*Rewritten each phase. Last updated: 2026-09-06, after the four-arm ablation.*
*Status: pipeline complete, ablation run and analysed.*

---

## What it is

A convolutional neural network that looks at a scanned document and says what kind of
document it is — a memo, a letter, a form, a resume, and so on across ten categories. It is
written from scratch in PyTorch: the architecture, the training loop, the metrics, and the
evaluation are all hand-written rather than adapted from an existing model. The point of the
project is to understand every part of it, so a model I can fully explain beats a better model
I cannot.

---

## Headline result

**84.84% test accuracy, 0.8282 macro-F1** on Tobacco-3482, from scratch, no pretrained weights.

That number needs three pieces of context to mean anything, all of which are the point:

| reference | value | what it tells you |
|---|---|---|
| Majority-class baseline | 17.85% | the trivial floor |
| **Logistic regression on 32×32 pixels** | **61.61%** | what crude ink distribution alone achieves |
| **This CNN** | **84.84%** | **+23.2 points over the linear baseline** |
| Label-noise ceiling | ~88% | a perfect model scores about this, because 11.7% of labels are wrong |

The margin over the linear baseline is the real result. Going from 64 pixel features to 1024
bought logistic regression only **+2.88 points**, so both linear models are reading coarse ink
density by region. The CNN's 23-point margin over that cannot be explained by density — it is
evidence the model learned page *structure*.

The distance to the ~88% ceiling is about 3 points, and a manual review of the errors (below)
suggests most of what remains is not mine to close.

**On the headline figure.** The raw test score was 85.22%, which crossed a >85% leakage
tripwire set during planning. The check found a real defect: the near-duplicate detection
hashed the *original* scans while the model trains on 256×256 cached images, and at ~9×
downscaling four test/train pairs that were 4–12 bits apart as originals collapse to identical
hashes. Thirteen test images (2.5%) were affected. Excluding them gives **84.84% / 0.8282**,
which is the number reported everywhere here. The correction is small, and it is stated
up front rather than buried.

**Macro-F1 came in 2.4 points below accuracy** (0.8282 vs 0.8484) against a predicted gap of
5–10, so the inverse-frequency class weighting worked better than expected on a dataset with
5.17× imbalance. Its cost is visible in Resume: recall 0.944, precision 0.548 — the weighting
made the model over-predict the rarest class, which is the intended trade.

---

## Architecture

**Input** 1 × 224 × 224 grayscale. Documents are resized on the long side and white-padded to
square, then randomly cropped from a 256 px cache.

**Backbone** Four VGG-style blocks — each `[Conv3×3 → BatchNorm → ReLU] × 2 → MaxPool(2)` —
with channels 32 → 64 → 128 → 256. Spatial size halves each block: 224 → 112 → 56 → 28 → 14.
**1,172,640 parameters** (measured).

**Head** Average-pool the 256 × 14 × 14 feature map to 256 × 3 × 3 — keeping a coarse sense of
where things sit on the page — then a single linear layer to 10 classes. **23,050 parameters.**

**Total 1,195,690 parameters**, deliberately small. With ~2,438 training images and no
pretrained weights the dominant risk is memorisation, not lack of capacity. For contrast,
VGG11's classifier head alone is ~119.6M parameters — about 5,000× larger than ours.

**Training** Cross-entropy with inverse-frequency class weights, AdamW at 3e-4, cosine
annealing, batch 32. Early-stopped at epoch 26 of 40 on validation macro-F1. **16 minutes,
36.8 s/epoch** on an M2 MacBook Air.

---

## From scratch vs. used — the honest boundary

**Written from scratch:** the model class, every layer choice, the training loop (forward,
loss, backward, optimizer step), the validation loop, checkpointing, metrics, the augmentation
policy, the dataset class, the caching layer, the near-duplicate detection, the split logic,
and the ablation harness.

**Used off the shelf:** PyTorch's autograd, the `nn.Conv2d` / `BatchNorm2d` / `MaxPool2d`
primitives and AdamW; torchvision for transforms and `DataLoader` only — never
`torchvision.models`, never pretrained weights; scikit-learn for metrics and the two logistic
regression baselines; `imagehash` for perceptual hashing.

**Deliberately not used:** any pretrained backbone, any transfer learning, any high-level
training wrapper. This costs roughly 10 accuracy points against a fine-tuned ImageNet model,
and that trade is the point.

---

## What the errors actually are

Of 77 test errors, I reviewed the 30 the model was most confident about and classified each by
looking at the image:

| category | count | share |
|---|---|---|
| **Label error** — the model was right, the ground truth is wrong | **6** | 20% |
| **Genuinely ambiguous** — a reasonable annotator could go either way | **13** | 43% |
| **Model error** — the model was wrong | **11** | 37% |

**63% of the most-confident errors are not the model's fault.** Examples of the clear label
errors: a page headed *"Inter-office Memorandum"* with To/From/Subject/Date fields is labelled
**Report**; another headed *"INTEROFFICE MEMORANDUM"* is labelled **Letter**; a workplace
no-smoking poster is labelled **Scientific**. The model called all three correctly.

This is what the ~88% ceiling looks like from the inside, and it is why "just get the accuracy
higher" is the wrong goal on this dataset.

---

## The interesting problem

The project's most useful hour was spent discovering that an experiment I had designed could
not answer its own question.

The design decision was how to pool the network's final feature map before classification. The
conventional choice, global average pooling, collapses each channel to a single number — which
throws away *where* things are on the page. That seemed wrong for documents, where a letterhead
at the top and a signature block at the bottom are much of what distinguishes one type from
another. So I proposed keeping a coarse 3×3 spatial grid instead, and running an ablation: pool
to 1×1, 3×3, and 7×7, and see which wins.

The ablation was confounded. Those three options differ in spatial resolution **and** in the
number of parameters in the classifier head — 2.5k, 23k, 125k. If 3×3 won, there would be no
way to tell whether spatial position carried real signal or whether the head simply had nine
times more capacity. Both explanations predict the same result. It was three training runs that
could produce a confident-looking table and answer nothing.

The fix was a fourth arm: pool to 1×1, then add a hidden layer sized so the head has 22,972
parameters against the 3×3 head's 23,050 — a 0.34% match, with no spatial information at all.
Now the comparison isolates one variable.

Then the second problem. The test set is 521 images. At 85% accuracy, detecting a difference
between two arms at 80% statistical power requires a gap of about 7.5 points. The effect being
chased is plausibly 2–3 points. The experiment was underpowered by roughly 2.5×.

Part of that was recoverable. Both arms are evaluated on the *same* images, so the comparison is
paired, and McNemar's test — which looks only at the images where exactly one model is right —
improves the detectable difference to about 3.4–4.8 points. Roughly twice the resolution, for
free.

Part of it was not. Running more random seeds does not help: the uncertainty comes from having
only 521 test images, and every seed is evaluated on the same 521. That is the difference
between "run more seeds until it's significant," which is p-hacking, and "this test set has a
floor we cannot cross," which is the truth.

So the protocol now pre-registers a single primary comparison before any run, measures how much
the result moves across random seeds and reports every difference against that floor, and
treats "this dataset cannot resolve the question" as a valid outcome. One limitation is stated
rather than hidden: McNemar compares two *trained models*, not two *architectures*. The
architectural claim needs seeds as the unit of replication — about four hours of compute.

Knowing when an experiment cannot answer its question turned out to be more transferable than
any single architecture choice.

---

## The ablation, and the number that matters most

The pooling ablation ran: four heads on an identical backbone, differing only in whether and how
much spatial position survives before classification.

| arm | pooling | head params | test accuracy |
|---|---|---|---|
| gap1 | 1×1 (none) | 2,570 | 0.7582 |
| gap1_hidden(86) | 1×1 (none) | 22,972 | 0.6775 |
| **gap3** | **3×3** | 23,050 | **0.8560** |
| gap7 | 7×7 | 125,450 | 0.8253 |

**Adding a 3×3 grid over global average pooling is worth +9.78 accuracy points.** Spatial
position carries substantial signal for document classification — the design intuition held, and
by a much larger margin than the 2–3 points predicted.

**But the number I would put first is the noise floor.** Before running the arms, the same
architecture was trained three times with only the random seed changed. It varied by **2.11
accuracy points**. That measurement is what makes every other number here interpretable:

> The noise floor was measured rather than assumed, and it showed the expected effect was not
> resolvable at this test-set size and seed count. Most reports of this comparison would state a
> 2-point win without knowing the architecture differs from itself by 2.11 points across seeds.

As it happened the effect came in at 9.78 points — 4.6× the floor — so the conclusion survives.
Had it come in at 2 points, as predicted, the honest answer would have been "this experiment
cannot tell you", and the floor is what would have made that visible.

**Two things the ablation got wrong, both recorded.** The capacity-control arm did not converge
— its best epoch was its last, and its validation score exceeded its training score, which is
underfitting, not the overfitting I first reported. That makes the pre-registered primary
comparison an upper bound rather than an estimate, and it costs the clean separation of
"spatial information" from "head capacity" the design was built to provide. Separately, more
spatial resolution turned out not to be monotonically better: 7×7 pooling scored 3.07 points
*below* 3×3, for five times the parameters.

**And a hypothesis I had argued for was refuted.** I had claimed the Scientific class was
limited by modality — that a network reading page geometry could not represent a category
defined by subject matter — and predicted its F1 would stay flat across all four arms. It moved
from 0.485 to 0.648, a range four times the per-class noise floor. Layout carries real signal
for that class. What survives is narrower and still useful: Scientific plateaus at 0.648 while
other classes reach 0.85 or better, so the case for adding OCR rests on that residual gap rather
than on an impossibility claim.

## What I'd do differently

- **Measure on the real object, or label the proxy at the point of reporting.** The single
  recurring failure across this project, four times over: a correlation measured on random
  noise instead of real feature maps; an EXIF orientation bug diagnosed and reported as a cause
  when zero of 498 images carried the tag; a figure quoted from a scratch reimplementation that
  did not reproduce on the actual model; and near-duplicate detection run on source images
  rather than the cached tensors the model consumes — which is what caused the leakage. Same
  shape each time: measure something adjacent to the question, report it as the answer. The
  rule now: use the real model, the real data, the real device; where a proxy is unavoidable,
  name the distribution in the same sentence as the number; and verify a proposed mechanism
  exists before reporting it as the cause.
- **A green result from a test that cannot fail is worse than no test, and a red result from a
  test measuring the wrong thing is nearly as bad.** Both happened. The single-batch overfit
  check initially drew its 8 images from consecutive indices, which — because split indices are
  sorted — were all one class; "memorise 8 images" collapsed into "always predict class 0" and
  printed PASS in five steps. Separately, the checkpoint resume test failed for a reason that
  had nothing to do with checkpoints: dropout draws a fresh mask every forward pass, so
  interrupted and uninterrupted runs diverge from RNG alone. The tell was that two arms which
  should have differed printed byte-identical numbers.
- **Design the ablation with its control arm from the start**, rather than discovering the
  confound after specifying the experiment — and give every arm enough epochs to converge. The
  capacity-control arm was still improving when its budget ran out, which cost the comparison
  the isolation it was designed to provide.
- **Measure the noise floor before interpreting any comparison.** Two hours of repeated runs
  told me the architecture varies by 2.11 points against itself. Without that, a 2-point
  difference between arms would have looked like a result.
- **Run the power analysis before choosing the test split size**, not after.
- **Check backend op support during environment setup.** `AdaptiveAvgPool2d(3)` does not run on
  Apple's MPS backend when the input is not divisible by the output size, which blocked the
  exact arm in the pre-registered primary comparison and needed a measured workaround.
- **Look at the images, not the summary statistics about them.** A near-duplicate rate of 7.4%
  looked entirely plausible as a number; rendering the largest cluster showed that half of it
  was an artifact of perceptual hashing on near-blank pages.

---

## Roadmap

| Week | Work |
|---|---|
| 1 | **Done** — classification, from-scratch CNN, baselines, error analysis, and the four-arm pooling ablation with a measured seed-variance floor |
| 2 | Object detection on documents — logos, signatures, stamps, tables. A detection head written from scratch first, then compared against YOLO |
| 3 | OCR integration — extract text from detected regions. Motivated by Scientific plateauing at 0.648 F1 while other classes reach 0.85+, not by any claim that vision cannot help |
| 4 | A text model over the OCR output, fused with the image signal |
| 5 | The whole pipeline behind an API with a simple frontend |

The code is structured so each of these plugs in rather than requiring a rewrite: the backbone
is exposed separately from the head via `forward_features`, and the data and training layers do
not know what the model is.
