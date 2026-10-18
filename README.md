# SecurePulse

SecurePulse is a local Streamlit demonstration of cardiometabolic risk estimation
and unusual-measurement detection. It uses generated synthetic records only. It is
not a medical device, has not been clinically validated, and must not be used to
diagnose or treat anyone.

## Quick start

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
python train_model.py
```

Then start the backend and frontend in separate terminals as described in
[Local frontend and backend](#local-frontend-and-backend). Open the local URL
printed by Streamlit. By default, the app binds to loopback
(`127.0.0.1`) and disables Streamlit usage-statistics collection. Form submissions
are sent over loopback to the local prediction API and processed in memory; readings
and scores are not written to disk. After a successful prediction, a local demo audit
chain stores only a UTC timestamp and a fixed event type in
`data/securepulse_demo_audit.sqlite3`. Do not enter real
patient information. If you intentionally change the bind address for remote access,
configure deployment authentication, HTTPS, and network access controls separately
before exposing the app. The application also stops if Streamlit is configured with
a non-loopback bind address. This does not detect an external reverse proxy, tunnel,
or port-forward that exposes a loopback-bound server; keep those disabled unless
deployment access controls are in place.

## Local frontend and backend

SecurePulse has a separate FastAPI prediction engine. Run it and the Streamlit
frontend in two terminals after generating the synthetic data and models:

```powershell
# Terminal 1: start the backend API (loopback only)
python -m modules.api

# Terminal 2: start the browser UI
streamlit run app.py
```

The UI checks `http://127.0.0.1:8000/health` and sends synthetic readings to
`/predict`. The API validates the request, loads the models, returns the risk and
anomaly outputs, and records only the privacy-minimized audit event. If the API is
offline, the UI reports the connection failure and does not fall back to running
the models itself. Set `SECUREPULSE_API_URL` only when intentionally changing the
API address; use loopback for this local demo.

This is a local service connection, not a cloud deployment or a persistent
database. Submitted readings remain in process memory; the audit SQLite file is
local and is not a distributed ledger. Never send real patient data to this demo.
The container image binds the UI to its network interface only when explicitly
configured with `SECUREPULSE_ALLOW_REMOTE=1`; use that only behind an authenticated
deployment boundary. The image itself does not provide authentication.

## Optional private Google Cloud Run deployment

This repository does not automatically deploy itself. After installing the Google
Cloud CLI, signing in with `gcloud auth login`, selecting a billed project, and
ensuring the account has Cloud Run and Cloud Build permissions, deploy the
synthetic-only container with Cloud Run authentication required:

```powershell
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy secure-pulse `
  --source . `
  --region asia-south1 `
  --no-allow-unauthenticated `
  --set-env-vars SECUREPULSE_ALLOW_REMOTE=1 `
  --max 1
```

Replace `asia-south1` with an approved Cloud Run region for your project. Grant
`roles/run.invoker` only to intended identities and configure an approved identity
access layer for browser access. Do not add `--allow-unauthenticated`. A successful
deploy command is required before a Cloud Run URL exists; pushing to GitHub alone
does not deploy the service. The container runs the API and UI together; its audit
SQLite file uses ephemeral container storage and is not a persistent backend.
Never submit real patient data.

The results include plain-language guidance. Risk bands are software cutoffs applied
to a model estimate for synthetic labels; they are not medical guideline thresholds or
real-world disease probabilities. The anomaly flag only compares profiles with
synthetic training patterns. Neither output diagnoses a condition or explains medical
causes.

Streamlit static-file serving is explicitly disabled. The ignore rules help prevent
local logs, databases, uploads, and generated model artifacts from being added to
version control accidentally; `.gitignore` does not remove files already tracked.

## Project layout

- `app.py` — local Streamlit interface with Synthetic Patients, Live, and History tabs.
- `modules/api.py` — local FastAPI prediction engine used by the Streamlit frontend.
- `generate_data.py` — deterministic synthetic dataset generator.
- `train_model.py` — trains and stores the risk and anomaly models.
- `modules/data_processor.py` — feature schema and input validation.
- `modules/risk_prediction.py` — classifier training and prediction.
- `modules/anomaly_detection.py` — Isolation Forest training and scoring.
- `modules/privacy.py` — salted pseudonyms and direct-identifier removal.
- `modules/synthetic_audit.py` — local hash-chain prototype for demo events only.
- `tests/` — unit tests for validation, modeling, privacy, audit-chain, and app flows.
- `notebooks/securepulse_development.ipynb` — exploratory development notebook.
- `data/` and `models/` — generated CSV/model artifacts.

## Open-source license

SecurePulse is distributed under the MIT License; see [LICENSE](./LICENSE).
The included patient data is generated synthetic data, not real patient records.
Trained `.joblib` model artifacts are excluded from version control; generate
them locally with the quick-start commands above.

## Synthetic audit-chain prototype

After each successful demo prediction, SecurePulse appends a UTC timestamp and the
fixed event type `synthetic_prediction_completed` to a local SQLite hash chain. The
chain verifies links between entries and stops appending if it detects corruption.
It does not store readings, scores, patient identifiers, or arbitrary caller data.
The local file is ignored by Git.

The dashboard flow is: **new synthetic reading → validation → risk estimate →
anomaly detection → privacy-minimized audit event → dashboard refresh**. It displays
the latest result and session-only history of prediction summaries. Streamlit keeps
active form-widget values in session memory while the browser session is open; raw
readings are not copied into prediction history or the audit chain. History is not
persisted after the session ends.
The Live form has an optional fictional demo-name field. Its value is cleared on
submission and is not used by the models or copied into results, session history, or
the audit chain. Never enter a real person's name.

This is **not a blockchain**: it has no distributed peers, consensus, independent
validator, trusted timestamping, or external tamper anchor. A local administrator
who can rewrite the database can also rebuild the chain. It is a learning/demo
integrity check only, not an immutable audit system, security control, or compliance
feature. It is synthetic-demo-only; do not use it to audit real patient activity.

## Regenerate data and models

```powershell
python generate_data.py --samples 10000 --seed 42
python train_model.py
```

The generator rejects datasets smaller than 100 rows. Generated measurements and
labels are artificial and make no claim to represent a real population. Model
metrics are only a software demonstration, not a measure of clinical performance.
Generated model files and local data (other than the checked-in synthetic CSV) are
excluded by `.gitignore` to reduce accidental disclosure. Keep real records,
credentials, and models trained on private records out of this demo.

## Tests

```powershell
python -m pytest
```

## Privacy and limitations

No patient-record database or external model service is used. The app does not show
individual dataset rows or save submitted readings or prediction results. Its
synthetic audit prototype does persist event timestamps locally as described above.
Pseudonyms use HMAC-SHA256 and a
caller-provided secret of at least 16 characters; keep that salt out of source
control and logs. Pseudonyms are not anonymization and do not make real health
data safe to share. Before any real
health-data deployment, obtain appropriate clinical, security, privacy, and legal
review and implement the necessary access controls, auditability, encryption, and
regulatory safeguards. This local demo has no user authentication, access-control
system, or production-grade encryption/storage configuration and must not be used
for real patient data. No software can be guaranteed 100% secure.

## Future real-data deployment readiness (not implemented)

The following is a planning checklist for a possible EU-hosted deployment on
Google Cloud Platform (GCP) using Microsoft Entra ID. None of these integrations
or controls are implemented by this repository. Keep this project local and
synthetic-only until the responsible organization has approved the intended use
and completed the required privacy, legal, clinical, and security reviews.

1. **Establish governance first.** Identify the controller and processors, purpose,
   lawful basis, data categories, users, retention, and applicable EU and national
   requirements. Have qualified privacy/legal staff determine whether a Data
   Protection Impact Assessment (DPIA), data-processing agreements, or other
   safeguards are required. Hosting in an EU region does not itself establish
   compliance.
2. **Choose and review the GCP design.** Select an approved EU region and document
   data flows, backups, support access, disaster recovery, and any cross-border
   processing. Use separate development, test, and production projects. Keep
   development and tests on synthetic data.
3. **Design identity and authorization.** Use organization-managed Microsoft
   Entra ID accounts with MFA and a reviewed federation/OIDC design. Put the app
   behind an approved HTTPS identity-aware ingress; do not expose Streamlit
   directly. Enforce authorization on the server for every protected operation,
   use least-privilege roles, and regularly review and revoke access. UI visibility
   is not authorization.
4. **Decide whether data must be retained.** If persistence is necessary, select an
   organization-approved storage service and document encryption, service
   identities, key management, backups, retention, deletion, and restore testing.
   Keep direct identifiers separate from analytical data where appropriate.
   Pseudonymized information remains sensitive and must be access-controlled.
5. **Protect secrets, logs, and artifacts.** Use managed secret storage and
   narrowly scoped service identities. Redact patient values and credentials from
   application, proxy, analytics, and error logs. Restrict and retain audit logs
   according to approved policy. Only load trusted, integrity-checked model
   artifacts; never accept user-uploaded serialized models.
6. **Validate the clinical and ML use independently.** The current models use
   synthetic data and labels and are not clinically validated. Do not use their
   outputs for real-world diagnosis or treatment. Any future clinical use requires
   appropriate clinical validation, subgroup and bias evaluation, human oversight,
   model versioning, monitoring, and a rollback process.
7. **Gate launch on testing and sign-off.** Independently test authentication,
   authorization, data minimization, logging, deletion, backup recovery, incident
   response, dependency and image patching, and vulnerability handling. Obtain
   documented approval from the organization's privacy, security, legal, and
   clinical owners before any real patient data is processed.

This checklist is not legal advice, a compliance assessment, or a security
certification. No system can be guaranteed 100% secure.

## Real-time patient data and permissioned ledger (proposal only)

This section records a possible future architecture requested for SecurePulse.
It is **not implemented**: the application does not connect to an EHR, receive
real-time patient data, store patient records, authenticate users through Entra ID,
or connect to a blockchain. Continue to use synthetic records only. Do not enter,
upload, or commit real patient data.

### Proposed data flow

1. An authorized user signs in through the organization's Microsoft Entra ID
   tenant with MFA. Server-side authorization checks role and purpose for every
   protected action.
2. An approved EHR integration sends authorized FHIR resources through an
   authenticated, encrypted interface. Validate the source, schema, consent,
   provenance, and authorization before accepting data.
3. Store the minimum necessary clinical data in an organization-approved,
   encrypted EU-hosted data store. Keep identifiers and clinical records protected
   by least-privilege access, managed keys, retention/deletion controls, backup
   protection, and audited access.
4. If a permissioned Hyperledger network is approved, write only narrowly scoped,
   tamper-evident audit evidence. Keep patient records, readings, diagnoses,
   identifiers, tokens, and record contents off-chain. Any event reference or
   cryptographic commitment must be reviewed for linkability and re-identification
   risk; hashes and metadata are not automatically anonymous.
5. Read authorized clinical data from the protected off-chain store for processing.
   The currently bundled synthetic-data models are not validated for clinical
   decisions and must not be used to diagnose, treat, or triage real patients.
6. Return results only to authorized users, with their provenance, model version,
   limitations, and audit events handled under an approved retention policy.

### Required design decisions and launch gates

- Name the data controller, processors, EHR owners, ledger operators, and
  permissioned-network governance members.
- Specify jurisdictions, approved hosting regions, FHIR profiles and versions,
  consent/access rules, data categories, retention periods, and deletion behavior.
- Decide how access revocation, correction, and deletion are represented when
  ledger entries are immutable. Obtain privacy/legal review before writing any
  person-linked reference or commitment to a ledger.
- Complete threat modeling, privacy impact review, security assessment, clinical
  validation, incident-response planning, and independent acceptance testing.
- Obtain documented sign-off from the responsible privacy, legal, clinical, and
  security owners before processing real patient data.

This proposal is not a deployment instruction, compliance determination, or
security certification. Do not connect this local demo to real patient data or a
production ledger until the design is approved and implemented by qualified
personnel.
