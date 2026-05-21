# DeepVerify Pro

Real-time deepfake detection + C2PA content provenance — a working prototype.
Two surfaces: **Sign** (C2PA signing at point of origin) and **Detect**
(audio + video + provenance + financial-trigger over a captured stream, with a
colour-coded confidence indicator and a tamper-evident audit log).

## Before writing, extending, or reviewing ANY code

Invoke both project skills first — they are binding, not optional:

- **`deepverify-pro-coding-standards`** — engineering contract: the scope lock
  (only F1–F5), the locked ADK-orchestrator + deterministic-tools architecture,
  the pinned Python stack, code conventions, and the Definition of Done.
- **`deepverify-pro-ethics`** — the ACM Code of Ethics gate (1.2 avoid harm,
  1.3 honest, 1.6 privacy, 2.5 risk evaluation, 3.1 public good,
  3.7 societal infrastructure).

Authoritative feature spec: `.claude/product_description/product.md`.

## Features — only these ship (scope lock)

- **F1** Real-time audio deepfake detection    (ACM 1.2, 1.3)
- **F2** Live video face authenticity          (ACM 1.3, 1.6)
- **F3** C2PA cryptographic content provenance (ACM 2.5, 3.7)
- **F4** Out-of-band financial auth trigger    (ACM 1.2, 2.5)
- **F5** Tamper-evident audit trail            (ACM 3.1, 3.7)

Anything outside F1–F5 needs owner discussion. No fabricated metrics or
capabilities — a prototype baseline is always labelled as such.

## Layout

`deepverify_pro/` — `agents`, `tools`, `detection`, `provenance`, `adapters`,
`authorization`, `audit`, `cli`, `config`, `indicator`. `tests/` mirrors the
package tree. (`api/` — FastAPI — is an approved deferral; CLI-only this round.)

## Commands (virtualenv at `.venv/`)

```
source .venv/bin/activate
python -m pytest                 # tests
python -m ruff check .           # lint
python -m black --check .        # format
python -m mypy deepverify_pro    # types
```

All four must be green before any commit.

## Conventions

- Every module starts with the docstring header — feature F-id, ACM codes,
  scope — see coding-standards §6.
- Commits name their F-id(s) and end with the `Co-Authored-By` trailer.
- `keys/` and `models/` are gitignored — never commit key material or model
  weights.
