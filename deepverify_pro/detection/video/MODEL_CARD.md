# Model Card — `video-landmark-heuristic-baseline-v0`

> **Prototype baseline — not production-accurate.**
> This is a documented geometric heuristic, not a trained classifier. It
> exists to exercise the F2 pipeline end-to-end while PyTorch and a real
> CNN remain deferred (`CODING_STANDARDS.md` §3 approved deviations).

| Field | Value |
|---|---|
| Detector name | `video-landmark-heuristic-baseline-v0` |
| Feature | F2 — Live Video Face Authenticity Verification |
| ACM codes | 1.3, 1.6 |
| `is_production` | `False` (mandatory for baselines — §4.2) |
| Output | `synthetic_probability ∈ [0.30, 0.70]` (intentionally narrow band) |
| Landmark backend | `dlib` 20.x + `shape_predictor_68_face_landmarks.dat` (canonical §3) |

## Intended use

Wire-up and integration testing of the F2 pipeline (frame → 68-landmark
extraction → `Detector` contract → audit log → colour indicator). It is
**not** suitable for production deepfake screening and must not be
deployed against real calls without replacement by a trained model.

## What it does (transparent method)

1. Detect the largest face in the frame using dlib's HOG-based frontal
   face detector (ships in the wheel).
2. Predict 68 facial landmarks via the canonical Davis King predictor,
   fetched on-demand by `scripts/fetch_landmarks.py` and stored under
   `models/` (gitignored — never committed; weights file is ~99 MB).
3. Reduce to two single-frame geometric statistics:
   - **`symmetry_deviation`** — mean Euclidean distance between each
     mirror-pair landmark and the reflection of its counterpart across
     the facial midline (landmarks 27 → 8), normalised by face width.
     Real faces show natural left-right asymmetry (~3–5% of face width);
     suspiciously over-symmetric output nudges towards "synthetic".
   - **`inter_ocular_ratio`** — distance between eye centres divided by
     face width. Frontal real faces sit near ~0.40; large deviations
     nudge towards "synthetic".
4. Map each statistic to `[0, 1]` against *placeholder* anchors
   (`HEALTHY_SYM_DEVIATION = 0.04`, `HEALTHY_IOR = 0.40`,
   `IOR_TOLERANCE = 0.10`).
5. Average the two components and squash into the band `[0.30, 0.70]`.
   This is deliberate — the baseline must never report strong confidence
   in either direction (`ACM 1.3 / 2.5`).

## Training data

**None.** No training corpus. No fitted parameters. The three anchor
constants are *placeholders* and have not been calibrated against any
deepfake or real-face dataset. They are documented here precisely so no
one mistakes this baseline for a real classifier.

## Predictor weights provenance (ACM 1.6 / 3.7)

The 68-landmark `.dat` file is fetched by a separate opt-in script,
never auto-downloaded at runtime. The runtime detector only reads the
file from disk — no network egress, no media leaves the process. SHA-256
of the canonical decompressed file is pinned in
`scripts/fetch_landmarks.py`; mismatches abort the fetch.

## Known limits

- The output band is centred on amber by construction. Treat green/red
  reads from this baseline as weak signals at best.
- The heuristic was not validated against deepfake video — it cannot
  meaningfully discriminate against modern generative face systems.
- **Frontal-pose assumption.** dlib's HOG detector is tuned for
  near-frontal faces; profile / strong-yaw frames may fail face detection
  outright, raising `NoFaceDetected`. The inter-ocular-ratio heuristic
  also implicitly assumes frontal geometry.
- **Multiple faces.** If more than one face is present, the largest
  bounding box is selected (typical speaker-view assumption). A
  production detector would score every face.
- **Single-frame only.** The baseline ignores temporal signals that
  product.md §3.3 calls out (irregular blinking, micro-movements,
  trajectory inconsistencies). Temporal features land with a real model.
- Face detection cost dominates per-call latency; the heuristic itself
  is O(68).

## What replaces it

A real CNN ships when:
1. PyTorch is reintroduced (§3 deviation lifted), and
2. A training corpus + reproducible evaluation script land in the repo
   so any quoted accuracy is committed and reproducible (§4.2: zero
   fabricated metrics).

Until then this card is the ceiling of any honest performance claim
about `video-landmark-heuristic-baseline-v0`: **untested, uncalibrated,
prototype only.**
