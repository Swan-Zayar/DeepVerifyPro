# DeepVerify Pro — Deviations Log

Owner-approved deviations from `CODING_STANDARDS §3` and the ACM-codes ethics
contract (`.claude/skills/deepverify-pro-ethics/SKILL.md`). Every entry must
declare: scope, the rule being deviated from, codes traded, mitigations
shipped, and the rollback condition. No silent deviations.

---

## M10 — F3+F4 composition: trust list + provenance-driven OOB trigger (2026-05-25)

### Scope

The F4 out-of-band financial authorisation trigger is composed with the F3
provenance verifier so that any financial document submitted to `/verify`
with `financial_context=true` fires an OOB challenge when its C2PA manifest
is **missing**, **cryptographically invalid**, **OR signed by an issuer not
on the deployment's trust list**. The verifier also gained a separate
`is_trusted_issuer` boolean and the deployment-configurable
`Settings.signing_trusted_issuers` allow-list.

### What this deviates from

- **CODING_STANDARDS §2** — the locked architecture diagram shows six
  deterministic tools. The composition adds a seventh
  (`provenance_financial_trigger`) so a single ADK tool encapsulates the
  F3-verdict → F4-dispatch path. The architecture is still "one ADK
  orchestrator coordinating deterministic tools," and detection /
  cryptography are still never delegated to an LLM. Still a composition of
  two product.md features (F3 + F4), not a new feature.
- **CODING_STANDARDS §4.3 / ACM 1.2** — the F4 trigger surface widens from
  `{transcript signals}` to `{transcript signals ∪ failed-provenance financial doc}`.
  The independence guarantee is preserved: the new tool consults *only*
  the `ProvenanceResult`, never a detector handle or detector score
  (statically pinned by `test_tool_signature_takes_no_detector_handle`).
- **Latent honesty bug in `provenance/verifier.py`** — previously,
  `has_valid_signature=True` was returned for cryptographically-valid
  manifests whose issuer was not vouched for by the deploying
  organisation. That conflated crypto with trust (the §5 anti-pattern
  this product exists to prevent). Now two booleans:
  `has_valid_signature` (crypto only) and `is_trusted_issuer` (allow-list).

### Codes traded (ethics skill §2)

- **ACM 1.2 (avoid harm)** — *served.* The OOB challenge now fires on the
  three failure modes of a financial document's provenance, not just
  transcript keywords. An attacker self-signing a forgery with their own
  cert is caught at the trust-list check, not waved through by
  `has_valid_signature=True`.
- **ACM 1.3 (honest and trustworthy)** — *served.* `has_valid_signature`
  and `is_trusted_issuer` are surfaced separately in the API and the
  audit log, so the client can never mistake "the bytes were signed by
  someone" for "the deployment vouches for this signer."
- **ACM 2.5 (risk evaluation)** — *served.* The fail-closed default
  (empty `signing_trusted_issuers` ⇒ nobody trusted) refuses to silently
  authorise any financial document until the deployment populates its
  allow-list.

### ACM Mapping (ethics skill §3)

1. **Codes served** — 1.2 (defence-in-depth widened), 1.3 (honest two-
   boolean verdict), 2.5 (fail-closed trust).
2. **Mechanism** — `provenance_financial_trigger(result, recipient,
   channel, audit)` consults only `ProvenanceResult`, dispatches via an
   `OutOfBandChannel`, and appends one F5 audit event on every call (pass
   or fail). The new `verify_for_financial(input_path, recipient)`
   orchestrator method runs `provenance_verify` then
   `provenance_financial_trigger` into the shared F5 hash chain.
3. **Residual risk** — the allow-list is plaintext config; an attacker
   with write access to the deployment's environment can add their own
   CN. Mitigated by the F5 audit log (the addition is observable in
   `signing_trusted_issuers` changes if those are deployment-managed) but
   not eliminated. Trust-list management is out of scope for the
   prototype.
4. **Data-path check** — no new data paths; no media ever appears in the
   audit payload (the F5 tool writes only the verdict, issuer CN, reason
   code, and channel receipt id — ACM 1.6).

### Mitigations shipped with this deviation

- **Fail-closed default.** `Settings.signing_trusted_issuers` defaults to
  `()`. Until the deployment populates it with the leaf-cert common names
  of its own signing infrastructure, **every** issuer is untrusted and
  the composition refuses to authorise any financial document.
- **Honest two-boolean API.** `ProvenanceOut` now carries both
  `has_valid_signature` and `is_trusted_issuer`. The reason string also
  appends `(issuer not in deployment trust list)` when the signature is
  valid but untrusted.
- **Defence-in-depth pinned.**
  `test_tool_signature_takes_no_detector_handle` statically asserts that
  `provenance_financial_trigger` accepts no `detector` / `score` /
  `synthetic_probability` / `indicator_state` / `audio` / `video`
  parameter. The test breaks first if the tool ever grows a detector
  dependency.
- **F5 audit hygiene.** A `provenance.financial_trigger` event is
  appended on every call — pass or fail — so a silently-skipped
  composition is indistinguishable from a missed dispatch.

### Out of scope (this deviation does NOT authorise)

- Routing the trust-list check through an online PKI / OCSP / TSA — the
  product runs on-prem with no internet at runtime; the allow-list IS
  the trust anchor.
- Auto-populating `signing_trusted_issuers` from c2patool's bundled CA
  list — the bundle is for the public web, not the deploying
  organisation's signing infrastructure.
- Gating the trigger on any F1/F2 detection score — that would collapse
  the §4.3 / ACM 1.2 independence guarantee.
- Logging the recipient's identity, contact details, or the document
  body in the F5 chain — payload is metadata only.

### Rollback condition (revoked automatically when any of these hold)

- An online trust path replaces the offline allow-list — re-evaluate
  ACM 1.6 (the product was on-prem with no internet at runtime
  precisely because online PKI lookups would expose media metadata).
- The seventh tool is removed in favour of inlining its logic into
  `verify_for_financial` — re-state the §2 deviation accordingly and
  re-run the static defence-in-depth tests.
- The `has_valid_signature` field's semantics are changed back to mean
  "valid AND trusted" — this would re-introduce the §5 anti-pattern and
  must not happen.

### Materialised in

- `deepverify_pro/provenance/verifier.py` — `is_trusted_issuer`,
  `_is_trusted` (fail-closed), reason string honesty.
- `deepverify_pro/config/settings.py` — `signing_trusted_issuers`
  allow-list (default `()`).
- `deepverify_pro/authorization/trigger.py` — `TriggerReasonCode`
  Literal, `build_provenance_challenge`.
- `deepverify_pro/tools/provenance_financial_trigger.py` — the seventh
  ADK tool (composition).
- `deepverify_pro/agents/orchestrator.py` — `verify_for_financial`,
  `trusted_issuers` constructor param, 6→7 tool tuple.
- `deepverify_pro/api/app.py` — `/verify` `financial_context` +
  `recipient` form fields, `VerifyFinancialResponse`,
  `build_default_orchestrator` wires `signing_trusted_issuers`.
- `deepverify_pro/api/schemas.py` — `is_trusted_issuer` on
  `ProvenanceOut`, `ProvenanceFinancialOut`, `VerifyFinancialResponse`.
- Tests: `tests/test_provenance.py` (trust-list), `tests/test_financial_trigger.py`
  (reason codes), `tests/test_provenance_financial_trigger.py`
  (composition), `tests/test_orchestrator.py` (verify_for_financial,
  seven-tool surface), `tests/test_api.py` (`/verify` financial-context
  branch, seven-tool surface).

---

## M9 — Tunnel-exposed demo backend (2026-05-23)

### Scope

A demo build of the F1–F5 prototype is exposed to external viewers
("friends") so they can interact with the live UI without each running the
backend locally. The frontend is served by Vite through the existing
reserved ngrok domain. The backend (`python -m deepverify_pro.api`) runs
on the owner's laptop bound to `127.0.0.1:8000` and is exposed via a
second ngrok tunnel.

### What this deviates from

- **ACM 1.6 hard rule** (ethics skill §2): *"All three pipelines run
  on-premise; no audio, video, communication, or biometric data leaves
  the organisation's own servers."* Inference still runs on the owner's
  laptop, but friend-supplied frames now transit ngrok's edge before
  reaching it.
- **`deepverify_pro/api/app.py`** module-level claim that the API binds
  `127.0.0.1` by default. The bind itself is unchanged; the ngrok tunnel
  makes that bind reachable from the public internet.
- **`DeepfakeDetectionDemo.tsx`** previously claimed the detect endpoint
  *"MUST stay a localhost origin."* That claim is now scoped to the
  default build; the demo build acknowledges the deviation via an
  on-screen disclosure banner.

### Codes traded (ethics skill §2)

- **ACM 1.6 (respect privacy)** — *partially traded.* Inference is still
  on-prem; the data path now traverses a third party (ngrok TLS edge).
- **ACM 3.1 (public good)** — *served* by this deviation: external
  evaluation of the prototype is possible without each viewer installing
  the toolchain.
- **ACM 3.7 (societal infrastructure)** — *served* in the same way:
  earlier surfacing of design choices to a wider audience.

### ACM Mapping (ethics skill §3)

1. **Codes served** — 3.1 + 3.7 (broader evaluation surface).
2. **Mechanism** — short-lived ngrok tunnel; owner controls uptime; demo
   URL shared individually rather than published.
3. **Residual risk** — friend-supplied media traverses ngrok's edge,
   which terminates TLS. Risk cannot be eliminated while the tunnel
   exists; mitigated by the disclosure banner and demo-only scope.
4. **Data-path check** — backend continues to delete temp files in its
   `finally` blocks (`deepverify_pro/api/app.py`). No media body is
   logged. The F5 audit chain still runs on-prem; only request envelopes
   (path + status) transit ngrok.

### Mitigations shipped with this deviation

- **Disclosure banner** under the live-detection section of
  `src/app/components/DeepfakeDetectionDemo.tsx`. Renders only when
  `VITE_DVP_API_URL` resolves to a non-loopback origin (`IS_LOCAL_BACKEND`
  guard). Reads:
  > **Demo deployment.** Frames are sent to a remote prototype backend
  > over a third-party tunnel before processing — they leave your
  > machine. Do not share real or sensitive media. Production builds run
  > on-premise (ACM 1.6).
- **Conditional copy** on the screen-share start panel: the original
  *"frames sent only to your local backend"* sentence is swapped for an
  honest tunnel-mode line when the backend is non-loopback.
- **No new logging.** uvicorn access logs continue to record request
  envelopes (path + status), never bodies. No media payload is persisted
  beyond the per-request `finally` cleanup.
- **Demo signing cert only.** F3 signing material remains the prototype
  `keys/test_signing.crt` — assets visibly carry that issuer name; no
  real CA is claimed.
- **Secrets gitignored.** `ngrok.yml` (holds authtoken) and `.env.local`
  (holds the rotating backend URL) are gitignored. The committable
  template is `ngrok.example.yml`.

### Out of scope (this deviation does NOT authorise)

- Logging media bodies, transcripts, or recipient names.
- Persisting friend-uploaded frames/audio beyond the per-request
  lifecycle already in `deepverify_pro/api/app.py`.
- Removing or hiding the disclosure banner.
- Publishing the demo URL (social, marketing, broadcast). Sharing remains
  individual / small-group.
- Using real personal media — owner's own face only, or synthetic test
  content.
- Promoting the prototype signing cert as a real CA.

### Rollback condition (revoked automatically when any of these hold)

- The demo period ends — tear down the backend tunnel.
- A real (non-prototype) signing CA is introduced (`test_signing.*`
  retired) — ACM 1.6 must be reinstated before any media surface is
  reachable from a non-loopback origin.
- The frontend is migrated to a published hosting target (Vercel,
  Cloud Run static, Firebase Hosting, etc.) without a paired
  re-evaluation of the data path — a new deviation entry is required.
- A friend reports media in the demo backend's logs or an unexpected
  retention path — tunnel comes down immediately; new entry required to
  re-open.

### Materialised in

- `src/app/components/DeepfakeDetectionDemo.tsx` — disclosure banner +
  `IS_LOCAL_BACKEND` guard + tunnel-skip header.
- `ngrok.example.yml` — dual-tunnel config template.
- `.gitignore` — `ngrok.yml`, `.env.local`.
