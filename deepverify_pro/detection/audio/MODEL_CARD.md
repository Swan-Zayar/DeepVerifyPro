# Model Card — `audio-mfcc-heuristic-baseline-v0`

> **Prototype baseline — not production-accurate.**
> This is a documented heuristic, not a trained classifier. It exists to
> exercise the F1 pipeline end-to-end while PyTorch and a real LCNN remain
> deferred (`CODING_STANDARDS.md` §3 approved deviations).

| Field | Value |
|---|---|
| Detector name | `audio-mfcc-heuristic-baseline-v0` |
| Feature | F1 — Real-Time Audio Deepfake Detection |
| ACM codes | 1.2, 1.3, 1.6 |
| `is_production` | `False` (mandatory for baselines — §4.2) |
| Output | `synthetic_probability ∈ [0.30, 0.70]` (intentionally narrow band) |

## Intended use

Wire-up and integration testing of the F1 pipeline (MFCC extraction →
Detector contract → audit log → colour indicator). It is **not** suitable
for production deepfake screening and must not be deployed against real
calls without replacement by a trained model.

## What it does (transparent method)

1. Extract MFCCs at the product.md §3.3 hop of **25 ms** via `librosa`
   (default `n_mfcc=20`).
2. Reduce to two scalar statistics per audio segment:
   - **`mean_temporal_std`** — average per-coefficient standard deviation
     over time. Real speech is variable; flat or synthesised audio tends
     to vary less.
   - **`mean_delta_magnitude`** — average absolute first-difference of the
     MFCC matrix. A proxy for trajectory richness.
3. Map each statistic to `[0, 1]` against a *placeholder* anchor
   (`HEALTHY_TEMPORAL_STD = 30.0`, `HEALTHY_DELTA_MAG = 8.0`); lower values
   nudge towards "synthetic-leaning".
4. Average the two components and squash into the band `[0.30, 0.70]`.
   This is deliberate — the baseline must never report strong confidence
   in either direction (`ACM 1.3 / 2.5`).

## Training data

**None.** No training corpus. No fitted parameters. The two anchor
constants are *placeholders* and have not been calibrated against any
deepfake or clean-speech dataset. They are documented here precisely so
no one mistakes this baseline for a real classifier.

## Known limits

- The output band is centred on amber by construction. Treat green/red
  reads from this baseline as weak signals at best.
- The heuristic was not validated against deepfake audio (ElevenLabs,
  Resemble AI, Tacotron, etc.) — it cannot meaningfully discriminate
  against those tools.
- Flat / near-silent audio will trip the "low variance" component and
  drift towards red. This is an expected false-positive mode and is
  exactly the calibration question a real model must answer.
- Single-channel float audio in `[-1, 1]` only. Multi-channel input is
  rejected upstream in `mfcc.py` so the caller makes the down-mixing
  choice explicitly.

## What replaces it

A real LCNN ships when:
1. PyTorch is reintroduced (§3 deviation lifted), and
2. A training corpus + reproducible evaluation script land in the repo
   so any quoted accuracy is committed and reproducible (§4.2: zero
   fabricated metrics).

Until then this card is the ceiling of any honest performance claim
about `audio-mfcc-heuristic-baseline-v0`: **untested, uncalibrated,
prototype only.**
