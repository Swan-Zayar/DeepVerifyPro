# DeepVerify Pro — Coding Standards

> Authoritative engineering contract for building DeepVerify Pro as a **working prototype that implements exactly what `.claude/product_description/product.md` describes — no more, no less.** These standards are binding on all code in this repository. Apply this skill **together with** the `deepverify-pro-ethics` skill (ACM Code of Ethics 1.2, 1.3, 1.6, 2.5, 3.1, 3.7) — that skill is the ethics gate; this skill is how the code implements it.

---

## 0. Scope Lock (read first — this is a hard rule)

1. **Every module, function, and PR must trace to a product.md feature (F1–F5) and its ACM codes.** State the mapping in the module docstring (see §6).
2. **Nothing outside product.md ships without explicit discussion.** If an implementation seems to need a capability not in product.md, **stop and raise it with the project owner before writing it.** Do not "improve" the product unilaterally.
3. **No fabricated capability or metric, anywhere** (code, docs, UI, logs). A prototype baseline is labelled as such (ACM 1.3 / 2.5).
4. If a change serves no ACM code in scope, it does not belong in the product — say so and stop.

---

## 1. Product → Feature → ACM map (the only features in scope)

| ID | Feature (product.md §3.3) | ACM | Prototype responsibility |
|----|---------------------------|-----|--------------------------|
| **F1** | Real-Time Audio Deepfake Detection | 1.2, 1.3 | MFCC features every ~25 ms → `Detector` → probability + colour state |
| **F2** | Live Video Face Authenticity Verification | 1.3, 1.6 | 68 facial landmarks → `Detector` → per-frame score |
| **F3** | Cryptographic Content Provenance Signing | 2.5, 3.7 | C2PA sign at origin; verify incoming; flag missing/invalid signature |
| **F4** | Out-of-Band Financial Authorisation Trigger | 1.2, 2.5 | Detect financial language → independent-channel confirmation |
| **F5** | Audit Trail & Incident Reporting | 3.1, 3.7 | Timestamped, tamper-evident append-only log of all events |

The product is **two user-facing parts** (product.md framing):

- **Part A — Sign:** digitally sign media (C2PA) at point of origin. (F3)
- **Part B — Detect:** real-time deepfake detection over a captured stream + provenance verification + financial trigger + audit logging + colour indicator. (F1, F2, F3, F4, F5)

---

## 2. Architecture (locked)

**One ADK orchestrator agent coordinating deterministic tools.** Detection and cryptography are **never** delegated to an LLM — they are deterministic signal-processing / ML / crypto functions exposed as ADK tools.

```
Capture (MeetingAdapter) ──► ADK Orchestrator Agent
                                  │  calls deterministic tools:
                                  ├─ audio_detect      (F1)
                                  ├─ video_detect      (F2)
                                  ├─ provenance_verify (F3)
                                  ├─ sign_media        (F3)
                                  ├─ financial_trigger (F4)
                                  └─ audit_log         (F5)
                                  ▼
                         Colour-coded confidence state + audit log
```

Three pipelines run in parallel per product.md §3.4: **audio**, **video**, **provenance**. The provenance pipeline runs **independently** of the detection engines.

### Repository layout (canonical)

```
deepverify_pro/
  agents/        ADK orchestrator agent + agent config            (architecture)
  tools/         ADK tools — thin deterministic wrappers           (F1–F5)
  detection/
    base.py      Detector ABC — the pluggable interface
    audio/       MFCC extraction + audio Detector impls (baseline) (F1)
    video/       68-landmark extraction + video Detector impls     (F2)
  provenance/    C2PA sign + verify via c2pa-python                (F3)
  adapters/      MeetingAdapter ABC; LocalAdapter; ZoomAdapter(stub)(integration)
  authorization/ out-of-band financial trigger                     (F4)
  audit/         tamper-evident append-only log (hash chain)       (F5)
  api/           FastAPI app (sign + detect endpoints)             (surface)
  cli/           Typer CLI (sign + detect commands)                (surface)
  config/        pydantic-settings configuration
  indicator/     colour-state model (green/amber/red), no UI hype  (F1)
tests/           pytest suites mirroring the package tree
keys/            signing keys — GITIGNORED, never committed
```

---

## 3. Technology stack (pinned choices — do not substitute without discussion)

| Concern | Choice | Rationale |
|---|---|---|
| Language | **Python 3.11+** | ADK, ML, c2pa-python are all Python |
| Agent framework | **`google-adk`** | Orchestrator only (§2) |
| Audio features | **`librosa`** (MFCC) | product.md §3.3/3.4 specifies MFCC |
| Video landmarks | **`mediapipe`** (68-point) | product.md specifies 68 landmarks |
| ML runtime | **PyTorch** behind `Detector` | swappable; baseline ≠ production |
| Provenance | **`c2pa-python`** (official CAI) | product.md specifies C2PA |
| Capture | **`opencv-python`**, **`sounddevice`** | local webcam/mic/screen via adapter |
| API / CLI | **FastAPI** + **Typer** | the two thin surfaces |
| Config | **`pydantic-settings`** | typed, env-driven |
| Lint / format / types | **ruff + black + mypy** | enforced in CI and pre-commit |
| Tests | **pytest** (+ `pytest-cov`) | see §7 |

### §3 — Approved prototype deviations (owner-approved 2026-05-18)

These substitutions were discussed and approved per the §3 rule ("do not substitute without discussion"):

- **PyTorch deferred.** The prototype baseline uses real feature extraction (librosa MFCC, 68-point landmarks) + a lightweight classical classifier (numpy / scikit-learn), labelled "not production-accurate" (§4.2). PyTorch is reintroduced only when a real trained model lands.
- **68-point landmarks via `dlib`** (not mediapipe-468), to stay faithful to product.md §3.3's explicit "68 facial landmark points". Documented fallback: mediapipe-468 with a 68-subset mapping **only if `dlib` will not install**, disclosed in the model card as an approximation (§4.2).
- **FastAPI deferred.** This round is CLI-only (Typer). FastAPI remains the pinned choice for the next round.
- **C2PA via the official `c2patool` binary** (not the `c2pa-python` binding). `c2pa-python` 0.32.6 cannot produce a verifiable claim signature for an offline (no-TSA) self-signed signer — verified methodically: `from_info` no-TSA errors `Signature: empty string`; `from_callback` yields `claimSignature.mismatch` for ECDSA-raw, ECDSA-DER, and Ed25519 alike (while cert trust + data-hash bindings pass), proving a wrapper-level TBS defect, not an encoding mistake. `c2patool` is the CAI reference implementation; it signs/verifies offline self-signed correctly. Invoked locally only — no network, no TSA (honors §4.1). Still C2PA, still F3 — no product-scope change.

---

## 4. Ethics rules baked into engineering (binding — ACM gate)

### 4.1 Privacy hard-rule (ACM 1.6)
- **No audio, video, frame, MFCC, landmark, or biometric data may leave the local process to any third party.** No external/cloud API calls carrying media or derived biometrics.
- Network egress is **deny-by-default**. Only `localhost` is permitted in the prototype. A future `ZoomAdapter` is the *only* sanctioned external boundary and only under Zoom raw-data entitlement — added separately, with discussion.
- Signing private keys live in `keys/` (gitignored) or an OS keystore. **Never** commit keys or log key material.

### 4.2 Honesty (ACM 1.3, 2.5)
- A detection result is a **probability / confidence score**, never an absolute guarantee. The colour state (green/amber/red) is a *probabilistic indicator* — code comments, docstrings, API fields, and any UI text must say so. Never name a field/flag `is_genuine` / `verified_human`; use `synthetic_probability`, `confidence`, `indicator_state`.
- Every model implementation ships a **model card** (`detection/**/MODEL_CARD.md`): training data (or "heuristic/none"), known limits, and an explicit line: *"Prototype baseline — not production-accurate."*
- **Zero fabricated metrics.** No hard-coded accuracy/precision numbers in code, docs, README, badges, or demo output unless produced by a committed, reproducible evaluation script.

### 4.3 Defence-in-depth (ACM 1.2)
- The F4 financial trigger **fires regardless of any detection score** and must be independently unit-testable with detectors stubbed/disabled. It must never be gated on detector output.

### 4.4 Audit & infrastructure care (ACM 3.1, 3.7)
- Every detection event, score, provenance check, flag, and financial trigger is appended to the F5 audit log as a **hash-chained, append-only** record (each entry stores the previous entry's hash). Tampering must be detectable by a verification function.
- Audit records must not contain raw media or biometric vectors — store scores, hashes, timestamps, and event types only (re-honours 1.6).

### 4.5 Risk carry-forward (ACM 2.5)
- Prototype docs must retain the three named residual risks (automation complacency, adversarial escalation, false-positive cost). Nothing is presented as risk-free.

---

## 5. The pluggable `Detector` contract (F1/F2)

`detection/base.py` defines one ABC. Audio and video detectors implement it. The orchestrator depends only on this interface so models are swappable without touching agents/tools.

```python
class Detector(ABC):
    name: str            # e.g. "audio-lcnn-baseline-v0"
    is_production: bool  # MUST be False for prototype baselines

    @abstractmethod
    def score(self, frame: Frame) -> DetectionResult:
        """Return synthetic_probability in [0,1] + indicator_state. Pure; no network; no disk writes except via audit_log tool."""
```

- Baseline implementations must perform **real feature extraction** (real MFCC / real 68 landmarks). The classifier may be lightweight or a documented heuristic, but it must be honest about being a baseline (§4.2).
- Adding a model = new `Detector` subclass + model card + tests. It must **not** change the interface or expand product scope.

---

## 6. Module docstring header (mandatory on every module)

```python
"""<one-line purpose>.

Feature: F<n> (<product.md feature name>)
ACM: <codes, e.g. 1.2, 1.3>
Scope: in-product.md  | discuss-required
"""
```

A reviewer must be able to reject any file whose header cannot honestly name a product.md feature.

---

## 7. Code conventions

- **Typing:** full type hints; `mypy` clean (strict on `detection/`, `provenance/`, `authorization/`, `audit/`).
- **Style:** `ruff` + `black`; no `print` — use the stdlib `logging` module (audit events go through the F5 tool, not `logging`).
- **Determinism:** tools in `tools/` are thin and deterministic; no hidden global state; seed any randomness and log the seed.
- **Errors:** fail explicitly with typed exceptions; never silently swallow a detection or signing failure (a swallowed failure is a security failure here).
- **Secrets:** no keys, tokens, or paths-to-keys in code or VCS; load via `config/` + env.
- **Tests:** `pytest`; unit tests required for every tool and `Detector`; F4 must have a test proving it fires with all detectors disabled; the audit log must have a tamper-detection test. Target ≥80% coverage on `detection/`, `provenance/`, `authorization/`, `audit/`.
- **Commits/PRs:** each PR states its F-id(s), ACM codes, and confirms the Definition of Done (§8).

---

## 8. Definition of Done (per feature / PR — all must hold)

- [ ] Traces to a product.md feature (F1–F5); module headers present (§6).
- [ ] No capability outside product.md (or explicitly approved in discussion).
- [ ] ACM 1.6: no media/biometric egress; no committed keys; verified.
- [ ] ACM 1.3/2.5: probabilistic naming; model card present; no fabricated metrics.
- [ ] ACM 1.2: F4 path (if touched) fires independent of detector scores, with a test.
- [ ] ACM 3.1/3.7: events written to the hash-chained audit log; tamper test passes.
- [ ] `ruff`, `black`, `mypy`, `pytest` all green; coverage targets met.
- [ ] Consistent with the architecture in §2 (ADK orchestrator + deterministic tools).

---

## 9. Out of scope for the prototype (do not build without discussion)

- Live Zoom / Microsoft Teams deployment (requires Zoom raw-data entitlement + Marketplace/security review). `ZoomAdapter` ships as a documented **stub** behind `MeetingAdapter`; real integration is a separate, discussed milestone.
- Google Meet (not in product.md; web-sandboxed with no raw media access).
- Production-grade trained models, multi-agent topologies, or any feature not enumerated in §1.
