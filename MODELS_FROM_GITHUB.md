# Models adopted from GitHub — M8 summary

> Summary of every model / model-bearing project taken from GitHub during
> milestone **M8** (replace the F1/F2 detection baseline with a pretrained
> model). Companion to `ATTRIBUTIONS.md` and the per-model `MODEL_CARD*.md`
> files; the binding governance record is `CODING_STANDARDS.md` §3 (M8 entry).
>
> **Nothing was vendored.** No model weights are committed to this repository.
> What was adopted is *architecture* + *documented inference recipe*; weights
> are fetched separately by the owner under their own licence.

---

## 1. Self-Blended Images (SBI) — F2 deepfake detector

| | |
|---|---|
| Repository | <https://github.com/mapooon/SelfBlendedImages> |
| Paper | Kaede Shiohara & Toshihiko Yamasaki, *"Detecting Deepfakes with Self-Blended Images"*, CVPR 2022 (Oral) |
| Model | EfficientNet-B4 classifier (2-class), trained with the SBI method |
| Used in | `deepverify_pro/detection/video/efficientnet_sbi.py` — detector `video-efficientnet-sbi-v0` |
| Model card | `deepverify_pro/detection/video/MODEL_CARD_efficientnet_sbi.md` |

**What was taken:** the network architecture (EfficientNet-B4, 2-class head)
and the documented inference recipe only — face crop with a 1/8-per-side
margin, resize to 380×380, RGB, scale to `[0,1]`, softmax, fake-class index 1.
The SBI **training framework is NOT vendored** (no `AbstractDetector`, no
`DETECTOR`/`BACKBONE`/`LOSSFUNC` registries); the integration is a clean
adapter behind the unchanged `Detector` ABC (M8 §8).

**Weights:** the SBI EfficientNet-B4 checkpoint (`FFraw.tar` / `FFc23.tar`),
trained on FaceForensics++ real videos with Self-Blended-Images augmentation.
Distributed via Google Drive from the SBI repo. **Not committed** — installed
by the owner via the opt-in `scripts/fetch_sbi_weights.py`.

**Licence — RESEARCH-ONLY (verified 2026-05-23, M8 §7 hard gate):**
non-commercial academic / research use only; commercial use, sublicensing and
redistribution are explicitly prohibited. The weights additionally inherit the
**FaceForensics++ Terms of Use** (TU Munich) — non-commercial research /
education only, binding on any for-profit employer. DeepVerify Pro is
designated a non-commercial research prototype, under which these weights are
licence-compatible. **They must not be used in a commercial deployment.**

---

## 2. EfficientNet-PyTorch — backbone implementation

| | |
|---|---|
| Repository | <https://github.com/lukemelas/EfficientNet-PyTorch> |
| Author | Luke Melas-Kyriazi |
| Package | `efficientnet-pytorch` (PyPI) |
| Used in | `deepverify_pro/detection/video/efficientnet_sbi.py` (`EfficientNet.from_name`) |

**What was taken:** the EfficientNet-B4 architecture implementation, used to
reconstruct the network so the SBI checkpoint's state-dict loads into it. The
SBI checkpoint's keys match this library — **not `timm`** — which is why
`efficientnet-pytorch` replaced the `timm` named in the M8 proposal (recorded
in `CODING_STANDARDS.md` §3, M8 entry, decision B). Built with `from_name`
(no pretrained-weight download — ACM 1.6, no runtime network).

**Licence:** MIT — permissive; no usage restriction.

---

## 3. PyTorch — ML runtime (framework, not a model)

| | |
|---|---|
| Repository | <https://github.com/pytorch/pytorch> |
| Used in | `efficientnet_sbi.py` — tensor ops, `torch.load`, inference |
| Licence | BSD-3-Clause — permissive; no usage restriction |

Listed for completeness: the runtime that executes the model above. No model
weights originate from here.

---

## Honesty notes (ACM 1.3 / 2.5)

- **No accuracy is claimed.** `video-efficientnet-sbi-v0` ships with
  `is_production = False`. Real-world accuracy is unmeasured in this deployment
  until `scripts/evaluate.py` produces a number on a real labelled test set
  (M8 §10). No SBI benchmark figure is quoted as ours.
- **A face-crop domain shift exists.** SBI trained on RetinaFace crops; this
  adapter uses dlib's HOG detector (the existing project dependency) for crop
  selection. The framing difference may reduce accuracy — documented in the
  model card and to be quantified by the harness.
- **The research-only licence is a hard constraint**, not a footnote. It is
  recorded here, in `ATTRIBUTIONS.md`, in the model card, and in
  `CODING_STANDARDS.md` §3.
