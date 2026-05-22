# Model Card — `video-efficientnet-sbi-v0`

> **Prototype — NOT production-accurate.** This detector ships behind the
> evaluation harness with `is_production = False`. **No accuracy figure is
> claimed** for it until `scripts/evaluate.py` produces a measured,
> reproducible number on a real labelled test set (M8 §10 / ACM 1.3 / 2.5).
>
> **RESEARCH-ONLY WEIGHTS.** The pretrained weights and their training data are
> licensed for **non-commercial academic / research use only**. DeepVerify Pro
> is designated a non-commercial research prototype (owner decision
> 2026-05-23). This detector **must not be used in a commercial deployment**
> while it carries Self-Blended-Images / FaceForensics++-derived weights.

| Field | Value |
|---|---|
| Detector name | `video-efficientnet-sbi-v0` |
| Feature | F2 — Live Video Face Authenticity Verification |
| ACM codes | 1.2, 1.3, 1.6, 2.5 |
| `is_production` | `False` (until the eval harness clears an owner-agreed bar) |
| Architecture | EfficientNet-B4 (`efficientnet_pytorch`), 2-class head |
| Weights | Self-Blended Images checkpoint (Shiohara & Yamasaki, CVPR 2022) |
| Input | 380×380 RGB face crop, scaled to `[0, 1]` |
| Output | `synthetic_probability` = softmax probability of the "fake" class |

## Intended use

Real-time per-frame deepfake screening for the F2 pipeline (frame →
68-/face-crop → `Detector` contract → audit log → colour indicator), as a
trained-model replacement for `video-landmark-heuristic-baseline-v0`. It is
**not** cleared for production screening: `is_production` stays `False` and the
colour indicator remains a *probabilistic* signal (ACM 1.3), never a guarantee.

## Method — Self-Blended Images (SBI)

SBI trains an EfficientNet-B4 classifier to spot the subtle blending
boundaries that face-manipulation leaves behind. Its training trick: it needs
**only real faces** — it manufactures its own forgeries by blending a slightly
perturbed copy of a face back onto itself ("self-blending"). This yields
strong cross-manipulation generalisation. Inference is **per-frame** — no clip
buffering, no temporal model (M8 §4c defers temporal detectors).

## Inference recipe (this adapter)

1. Detect the largest face with dlib's HOG frontal-face detector.
2. Expand the face box by `FACE_MARGIN` = 1/8 of its size per side (the SBI
   test-phase margin: `w/4` halved).
3. Crop, resize to 380×380 (linear), convert BGR→RGB, scale to `[0, 1]`.
4. EfficientNet-B4 forward → softmax → probability of the "fake" class.

A **clean adapter** — only the network architecture and the documented
inference recipe are reused. The SBI training framework, and any
DeepfakeBench `DETECTOR`/`BACKBONE`/`LOSSFUNC` registries, are **not vendored**
(M8 §8). The `Detector` ABC is unchanged.

## Training data

FaceForensics++ real videos + Self-Blended-Images augmentation. **No training
was done in this repository** — the published SBI checkpoint is adopted as-is.
DeepVerify Pro stores no training corpus.

## Licence (verified 2026-05-23 — M8 §7 hard gate)

- **SBI code & weights** — non-commercial academic / research use only;
  commercial use, sublicensing and redistribution explicitly prohibited
  (Kaede Shiohara; see the SBI repository LICENSE).
- **FaceForensics++** (training data, Technical University of Munich) —
  non-commercial research / education only; the terms bind any for-profit
  employer of the user.
- DeepVerify Pro is designated a non-commercial research prototype; under that
  designation the weights are licence-compatible. Full credit and the licence
  record live in `ATTRIBUTIONS.md`. Attribution alone is **not** permission —
  the research-only restriction is a hard constraint on how this detector may
  be used.

## Weights provenance (ACM 1.6 / 3.7)

Weights load from a local file (`Settings.sbi_weights_path`) only. The
architecture is built with `from_name` (no pretrained-weight download), so
there is **no runtime network access**; inference runs fully in-process; no
audio, video, frame or biometric data leaves the machine. Weights are
gitignored under `models/` (never committed) and installed only via the opt-in
`scripts/fetch_sbi_weights.py`, which structurally verifies the checkpoint.

## Known limits & residual risks (ACM 2.5 — none dropped)

- **Automation complacency.** A green/low indicator is a probabilistic score,
  not a guarantee. The F4 out-of-band financial trigger remains independent
  defence-in-depth and fires regardless of this score.
- **Adversarial escalation.** SBI keys on blending boundaries — strong against
  face-swaps, **weaker against fully-synthetic faces** with no blend seam.
- **False-positive cost.** Flagging a genuine participant has real cost; the
  eval harness reports the confusion matrix at the indicator thresholds so this
  is visible, not hidden.
- **Cross-domain drop.** Published SBI accuracy is on its own benchmarks, not
  this deployment. **No benchmark figure is quoted as ours** — `scripts/evaluate.py`
  measures *our* number on *our* test set.
- **Face-crop mismatch.** SBI was trained on RetinaFace crops; this adapter
  uses dlib's HOG detector for crop selection (reusing the existing project
  dependency rather than vendoring RetinaFace). The differing crop framing is
  a known domain shift that may reduce accuracy — to be quantified by the
  harness. dlib's HOG detector also assumes near-frontal faces and may raise
  `NoFaceDetected` on profile / strong-yaw frames.
- **Single-frame only.** Temporal cues (irregular blinking, micro-movements)
  are not used; temporal detectors are explicitly deferred (M8 §4c).

## What clears `is_production`

`scripts/evaluate.py` run over a real labelled test set, producing ROC-AUC,
EER and the confusion matrix at the indicator thresholds. `is_production` is
flipped to `True` only by a reviewed human decision once a measured number
clears an owner-agreed bar (M8 §10). Until then this card is the ceiling of any
honest claim about `video-efficientnet-sbi-v0`: **adopted pretrained weights,
unmeasured in this deployment, research prototype only.**
