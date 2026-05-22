# Milestone Proposal — M8: Replace the F1/F2 Detection Baseline with a Pretrained Model

> **Status: PROPOSAL — awaiting project-owner decision. Nothing in this document
> is approved or implemented. No code changes until §13 is signed off.**
>
> Author: prepared for owner review · Date: 2026-05-21
> Features: **F2** (Live Video Face Authenticity), **F1** (Real-Time Audio Deepfake Detection)
> ACM codes: 1.2, 1.3, 1.6, 2.5

---

## 1. The ask (one sentence)

Replace the chance-level F1/F2 heuristic baselines with **pretrained, openly-licensed
detection models wrapped as `Detector` subclasses** — which requires the owner to
**lift exactly one standing approval: the CODING_STANDARDS §3 "PyTorch deferred" deviation.**

Everything else in this proposal stays inside existing scope and existing approvals.

---

## 2. Problem statement

Manual testing of the current detector reports **~50% on both real and fake video** —
indistinguishable from a coin flip.

This is **not a defect**. It is the documented, expected behaviour of the baseline:

- Both detectors are *documented heuristics*, not trained classifiers.
- Output is deliberately **clamped to `[0.30, 0.70]`** (amber band) so the prototype
  never overstates confidence.
- Calibration anchors (`HEALTHY_TEMPORAL_STD`, `HEALTHY_SYM_DEVIATION`, …) are marked
  *"placeholder — NOT empirically validated"* in code and in both `MODEL_CARD.md` files.
- The audio model card states it outright: *"cannot meaningfully discriminate."*

The baselines exist to exercise the F1/F2 pipeline end-to-end. Both model cards name
this milestone explicitly under *"What replaces it."* M8 is that planned replacement.

**Note on honesty (ACM 1.3 / 1.2):** a detector stuck at chance is not harmless. It
feeds an authoritative-looking colour indicator with noise — an automation-complacency
hazard. Fixing it is not a "nice to have"; it directly serves ACM 1.2 and 1.3.

---

## 3. Scope check (what this is, and is not)

| Question | Answer |
|---|---|
| New product feature? | **No.** Detection accuracy is the core of F1/F2 already in `product.md`. |
| Changes the `Detector` ABC (CODING_STANDARDS §5)? | **No.** New `Detector` subclasses only. |
| Changes agents / tools / orchestrator? | **No.** The orchestrator depends only on the ABC. |
| Touches the privacy boundary (ACM 1.6)? | **No.** See §5 data-path check. |
| Outside a standing approval? | **Yes — one item:** the §3 "PyTorch deferred" deviation. |

The §3 deviation lift is the **entire reason this is a proposal and not just work.**

---

## 4. Proposed approach

**Adopt a pretrained model — do not train from scratch, do not tune heuristics.**

| Option considered | Verdict |
|---|---|
| Tune the existing heuristics | Stays at chance; presenting it as better would breach ACM 1.3. ✗ |
| Train a model from scratch | Needs a licensed corpus (FaceForensics++ ≈ tens of GB), GPU time, weeks. Overkill for a prototype. ✗ |
| **Adopt a pretrained, openly-licensed detector** | Training already done by the authors; honest detection in days. **✓ proposed** |

### 4a. F2 — video (primary)

- **Model:** EfficientNet-B4 trained with **Self-Blended Images (SBI)** — Shiohara &
  Yamasaki, *Detecting Deepfakes with Self-Blended Images*, CVPR 2022.
- **Why:** per-frame inference (real-time, no clip buffering), strong cross-manipulation
  generalisation, and the authors released trained weights.
- Backbone: EfficientNet — Tan & Le, *EfficientNet*, ICML 2019.

### 4b. F1 — audio (phase 2 of this milestone — not dropped)

The audio baseline is equally at chance. It is **not** in the primary phase only
because the reported failure was video, but it must not be hidden (ACM 2.5).
Recommended: a pretrained anti-spoofing model — e.g. **AASIST** (Jung et al., ICASSP
2022) or **RawNet2** — adopted via the *same* pattern once F2 lands.

### 4c. Explicitly deferred

Temporal video detectors (TALL, FTCN) are **not** proposed. They need clip buffering
(added latency) and a frame-buffer layer. Revisit only if the eval harness shows the
per-frame model misses temporal-only fakes.

---

## 5. ACM Mapping (mandatory acceptance gate — ethics skill §3)

1. **Codes served.**
   - **1.2 (avoid harm)** — a working detector replaces a chance-level signal that
     currently gives false reassurance to a deceived employee.
   - **1.3 (honest & trustworthy)** — the colour indicator becomes a meaningful
     probabilistic assessment instead of noise.
   - **2.5 (thorough evaluation)** — the model ships only behind the eval harness (§9)
     with a measured, reproducible number; named residual risks carried forward (§11).
   - **1.6 (privacy)** — preserved; see data-path check below.
2. **Mechanism.** A real trained CNN replaces the documented heuristic, wrapped behind
   the unchanged `Detector` ABC. Per-frame inference. Accuracy is *measured* by the
   harness, never asserted.
3. **Residual risk (2.5).** Cross-domain accuracy drop is real and expected; SBI is
   weaker on fully-synthetic (no-blend) faces. Mitigation is partial — see §11.
   `is_production` stays `False` until the harness clears an agreed bar.
4. **Data-path check (1.6).** Weights are loaded **from local disk only** — same
   pattern as the existing dlib predictor: an opt-in fetch script, SHA-256 pinned,
   no runtime network. Inference runs fully in-process. **No audio, video, frame,
   or biometric data leaves the machine.** Constraint holds.

**No ACM Mapping → not accepted. This mapping is the gate; it holds.**

---

## 6. Dependencies requested (the §3 decision)

| Dependency | Purpose | Approval status |
|---|---|---|
| `torch` (PyTorch) | ML runtime for the trained model | **Requires lifting the §3 "PyTorch deferred" deviation** |
| `timm` | EfficientNet-B4 backbone + weights loading | New pinned-stack entry, owner approval |

If approved, CODING_STANDARDS §3 is updated with a new dated, owner-attributed entry
recording the lift — the same way the existing deviations are recorded.

---

## 7. Weights, corpus, and licensing

- **Pretrained weights** come from the SBI release (or a DeepfakeBench / Hugging Face
  checkpoint). **Their licence MUST be verified before adoption** — attribution alone
  is not permission. This is a hard gate inside the milestone.
- **No training corpus required** — that is the entire point of adopting pretrained weights.
- **A small labelled test set IS required** for the eval harness (§9): a handful of
  known-real and known-fake clips, sourced under licence or provided by the owner.
- All model and framework authors credited in `ATTRIBUTIONS.md`.

---

## 8. Integration design (how it fits the locked architecture)

- **New file:** `deepverify_pro/detection/video/efficientnet_sbi.py` — a `Detector`
  subclass implementing `score(Frame) -> DetectionResult`.
- It wraps a `timm` EfficientNet-B4 `nn.Module`. **The DeepfakeBench framework is NOT
  vendored** — no `AbstractDetector`, no `DETECTOR`/`BACKBONE`/`LOSSFUNC` registries.
  Only the network architecture is reused; the integration is a clean adapter.
- `MODEL_CARD.md` written alongside it (training data, limits, the mandatory
  *"not production-accurate"* line until the harness says otherwise).
- `is_production = False` until §9 clears an agreed threshold.
- Tests under `tests/` mirroring the existing detector test pattern.
- **Untouched:** the `Detector` ABC, `agents/`, `tools/`, the orchestrator, F3/F4/F5.
  F4 (financial trigger) already fires independent of detector score — unaffected.

---

## 9. The evaluation harness (precondition — buildable now, no sign-off)

`scripts/evaluate.py`: runs a `Detector` over a labelled test set and reports
**ROC-AUC, EER, and the confusion matrix at the indicator thresholds.**

- Serves ACM 2.5; adds no dependency; does not expand scope → **no owner sign-off needed.**
- Run #1 target: the *current baseline* — turning "around 50%" into a committed,
  reproducible number (honours §4.2: zero fabricated metrics).
- Thereafter it is the gate that decides whether any model earns `is_production = True`.

Recommendation: build this regardless of the §13 decision — it is required either way.

---

## 10. Sequencing

1. **(No sign-off)** Build the evaluation harness; measure and record the baseline.
2. **(Owner decision — §13)** Approve this proposal → record the §3 deviation lift.
3. Verify the pretrained-weights licence.
4. Implement the F2 `Detector` subclass + model card + tests.
5. Run the harness → honest measured number. Flip `is_production` **only** if it
   clears the agreed bar; otherwise it stays `False` and stays labelled a baseline.
6. **Phase 2:** F1 audio model (AASIST / RawNet2), same pattern.

---

## 11. Residual risks carried forward (ACM 2.5 — none dropped)

- **Automation complacency** — a green indicator must never be read as a guarantee;
  it stays a probabilistic score. F4 remains independent defence-in-depth.
- **Adversarial escalation** — SBI keys on blending boundaries; strong against
  face-swaps, **weaker against fully-synthetic faces** with no blend seam. Documented
  in the model card; a temporal second opinion may be revisited later.
- **False-positive cost** — flagging a genuine participant has real cost; the harness
  reports the confusion matrix at threshold so this is visible, not hidden.
- **Cross-domain drop** — published accuracy is on the model's benchmarks, not our
  deployment. The harness measures *our* number; no benchmark figure is ever quoted
  as ours.

---

## 12. Definition of Done (per CODING_STANDARDS §8 + ethics skill §6)

- [ ] Traces to F1/F2; module headers present; no scope expansion.
- [ ] §3 deviation lift recorded with date + owner attribution.
- [ ] ACM 1.6 verified — weights from local disk, no media/biometric egress.
- [ ] ACM 1.3/2.5 — model card present; `is_production` honest; **zero fabricated
      metrics** — every number from the committed harness.
- [ ] DeepfakeBench framework not vendored; `Detector` ABC unchanged.
- [ ] `ruff`, `black`, `mypy`, `pytest` green; coverage target on `detection/` met.
- [ ] Weights licence verified and recorded in `ATTRIBUTIONS.md`.
- [ ] Residual risks (§11) retained in the model card.

---

## 13. Owner decision requested

Please approve / reject / modify each:

| # | Decision | Y / N / Modify |
|---|---|---|
| A | Lift the CODING_STANDARDS §3 "PyTorch deferred" deviation | |
| B | Approve `torch` + `timm` as pinned-stack additions | |
| C | Approve adopting pretrained EfficientNet-B4 / SBI weights (pending licence verification) | |
| D | Approve F1 audio (AASIST / RawNet2) as phase 2 of this milestone | |
| E | Authorise building the evaluation harness now, ahead of A–D | |

**Until A–C are approved, no PyTorch code is written.** Item E can proceed
independently and is recommended regardless.
