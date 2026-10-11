# DevPilot AI

An AI software-engineering agent: given a task, it inspects a sandboxed
codebase, proposes a change, waits for human approval, writes the
change, runs the test suite, and — if tests fail — diagnoses the
failure, proposes a fix, asks for approval again, and retries. Built
with Django REST Framework + LangGraph + Gemini, with a small React
frontend.

## Project structure

```
DevPilot-AI/
├── backend/
│   ├── manage.py
│   ├── requirements.txt        # reconstructed & verified — pip install -r this
│   ├── .env.example            # copy to .env and add your real key
│   ├── .env                    # NOT committed — your real GEMINI_API_KEY
│   ├── db.sqlite3              # created by `manage.py migrate`
│   │
│   ├── config/                 # Django project settings
│   │   ├── settings.py
│   │   ├── urls.py             # all API routes
│   │   ├── asgi.py / wsgi.py
│   │
│   ├── core/                   # the Django app
│   │   ├── agent.py            # the LangGraph state machine (the agent itself)
│   │   ├── agent_service.py    # thin service layer views.py calls into
│   │   ├── tools.py            # sandboxed file/test tools the agent may call
│   │   ├── ai_service.py       # simple one-shot Gemini chat helper
│   │   ├── views.py            # REST endpoints
│   │   ├── models.py           # (no models needed yet)
│   │   ├── admin.py / apps.py
│   │   ├── migrations/
│   │   ├── tests.py            # automated test suite (see below)
│   │   └── run_agent.py        # manual CLI harness for the full flow
│   │
│   └── workspace/               # the SANDBOX the agent reads/writes/tests
│       ├── auth.py              # sample file with a login() function
│       ├── test_auth.py         # sample test for it
│       └── safe_test.py         # a harmless extra file
│
└── frontend/                    # React + Vite chat UI
    ├── src/App.jsx               # currently wired to /api/ai/chat/ only
    ├── package.json
    └── ...
```

`backend/venv/` and `frontend/node_modules/` were removed — they're
regenerated from `requirements.txt` / `package.json` (see Setup).

## What was cleaned up

- Removed `backend/venv/` (182 MB) and `frontend/node_modules/` (63 MB)
  — regenerable, shouldn't be shipped in the zip.
- Removed `__pycache__/`, `.pytest_cache/`, `db.sqlite3`.
- Removed 11 one-off manual debug scripts that had accumulated in
  `backend/` and `backend/core/` (`test_ai_debug.py`, `hitl_test.py`,
  `gemini_test.py`, `test_debug_pipeline.py`, `test_proposal_hitl.py`,
  etc.) — these were exploratory scripts used while building the graph,
  not a real test suite, and duplicated logic that now lives in
  `core/tests.py`.
- Removed a stray malformed `workspace/workspace/auth.py` left over
  from an earlier bug (see below).
- Removed the duplicate `debug_fix_graph` from `agent.py` — it was a
  second copy of the same write→test→debug→fix loop that already
  exists inside the main `agent` graph, kept only for isolated manual
  testing. The one unified graph now handles everything.

## Backend integrations completed

The graph in `core/agent.py` already modeled the full flow; it just
wasn't reachable except by running scripts by hand. It's now exposed
over HTTP:

| Endpoint | Purpose |
|---|---|
| `POST /api/agent/run/` | Start a new agent run: `{"task": "..."}` → runs the main flow until it finishes or needs human approval |
| `POST /api/agent/resume/` | Resume a paused run: `{"thread_id": "...", "decision": "approve" \| "reject"}` — handles **both** HITL checkpoints (approving a normal change, and approving an AI-generated fix) since they share one `approval` node |
| `GET /api/status/` | Health check |
| `POST /api/ai/chat/` | Simple one-shot Gemini chat (what the current frontend uses) |

Response shape from both agent endpoints:
```json
// paused, needs a human decision
{ "status": "waiting_for_approval", "approval_request": {...}, "thread_id": "..." }

// finished
{ "status": "completed", "final_result": "...", "test_result": "...", "debug_result": "...", "messages": [...], "thread_id": "..." }
```

Mapped to your checklist:

1. **Test main agent flow** — done & verified (see Testing below): task → propose → approve → write → test.
2. **Connect debug loop** — already wired inside the single graph (`test` → `debug` → `ai_debug` → `prepare_debug_fix`); confirmed working end-to-end.
3. **HITL for AI fix** — `POST /api/agent/resume/` handles this checkpoint too. **A real bug was found and fixed here** (see below) — the AI fix was being silently auto-approved without asking a human again.
4. **Connect test/retry loop** — verified working; also fixed a bytecode-cache issue that could make retries see stale test results (see below).
5. **Full end-to-end testing** — verified all three paths: approve→pass, approve→fail→AI-fix→approve→pass, and reject→clean-stop. Real Gemini calls still need to be tested manually with your key (LLM output can't be verified in this offline environment) — use `core/run_agent.py` or the new endpoints for that.
6. **Cleanup & organization** — done (this README + structure above).

### Two real bugs found and fixed while wiring this up

1. **HITL for the AI fix was being skipped.** `approval_node` had a
   guard for "already answered" that reused the *old* decision — so
   once you approved the first change, the AI's later fix proposal was
   auto-approved without ever asking again. Fixed by resetting
   `approval` back to `""` in `write_node` once a decision is consumed,
   so each new proposal gets its own interrupt. Covered by
   `test_write_node_resets_approval_for_next_hitl_checkpoint` in
   `core/tests.py`.
2. **Stale test results in the retry loop.** `run_tests` reused Python's
   `__pycache__` bytecode cache between successive runs; a rewritten
   file with the same size could be served from a stale `.pyc`, making
   the retry loop see the wrong pass/fail result. Fixed in
   `core/tools.py` by clearing `__pycache__` and disabling bytecode
   writing before every test run.

## Setup (local development)

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your keys (see below)
python manage.py migrate
python manage.py runserver
```

`backend/.env` needs at least:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` / `GEMINI_API_KEY` / `HF_TOKEN` | LLM providers — set one or more |
| `LANGGRAPH_POSTGRES_URI` | PostgreSQL for resumable run checkpoints (tables are created automatically) |
| `GITHUB_TOKEN` | Fine-grained token: Contents + Pull requests (read/write) |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG` | Standard Django settings |

### Frontend
```bash
cd frontend
npm install
npm run dev                        # http://localhost:5173
```

## How a run works

```
explore repo -> propose change -> [you approve the diff] -> agent branch
  -> write file -> run tests -> (fail: debug -> propose fix -> approve -> retry)
  -> commit -> [you approve the push] -> push branch -> open pull request
```

- Two human approvals: one for every file change, one before anything
  leaves your machine (push + PR).
- Commits and pushes to `main`/`master` are refused.
- Run history (`GET /api/agent/runs/`) lets you reopen a paused run and
  continue its approval after a page reload.

### LLM provider fallback

Providers are tried in `LLM_PROVIDER_ORDER` (default
`openrouter,gemini,huggingface`). A provider without a key is skipped.
Rate limits / outages are retried once (`LLM_RETRIES`), then the next
provider is used. If all fail, the error lists every provider's failure.
Models can be overridden with `OPENROUTER_MODEL`, `GEMINI_MODEL`,
`HF_REPO_ID`.

### Code search

`search_code` ranks files (BM25-style keyword scoring, camelCase/snake_case
aware, boosts files that *define* the thing searched for) and returns line
numbers + snippets. `read_file` accepts `start_line`/`end_line` for large
files. This is keyword retrieval, not embeddings.

## API

All endpoints except register/login need `Authorization: Bearer <access>`.

| Method | Path | |
|---|---|---|
| POST | `/api/auth/register/`, `/api/auth/login/`, `/api/auth/refresh/` | JWT auth |
| POST | `/api/agent/run/` | `{repo_url, task}` -> result or `waiting_for_approval` |
| POST | `/api/agent/resume/` | `{thread_id, decision: "approve"\|"reject"}` |
| GET | `/api/agent/runs/` | your recent runs |
| GET | `/api/agent/runs/<thread_id>/` | one run (+ pending approval if paused) |
| GET | `/api/github/status/`, `/api/github/connect/` | GitHub App connection state / install URL |
| POST | `/api/github/disconnect/` | forget the user's GitHub connections |
| GET | `/api/github/callback/` | where GitHub returns after installation (browser redirect) |

## GitHub access: personal token or GitHub App

- **Single token (default):** `GITHUB_TOKEN` in `.env`; you grant that token
  access to each repository.
- **GitHub App:** every user clicks **Connect GitHub** once and chooses which
  repositories to share; DevPilot mints short-lived, per-repository tokens and
  also works with private repositories. Setup (one-time registration):
  **[docs/GITHUB_APP_SETUP.md](docs/GITHUB_APP_SETUP.md)**.

## Testing

```bash
cd backend
python manage.py test core          # Django tests (needs the Postgres from LANGGRAPH_POSTGRES_URI)
python -m pytest core/test_git_delivery.py core/test_llm_fallback.py core/test_code_search.py
cd ../frontend && npm run lint && npm run build
```

`FullGraphDeliveryTests` drives the real compiled graph without an LLM
(approve -> test -> commit -> approve push -> push -> PR) against a local
bare git remote, so the delivery pipeline is covered offline. Anything
that needs a real model is left for manual end-to-end testing.

## Deployment (Docker)

```bash
cp .env.example .env               # fill in secrets
docker compose up -d --build
# UI:  http://localhost:8080      API: http://localhost:8000
```

- `db` (PostgreSQL 16) holds both Django tables and LangGraph checkpoints.
- `backend` runs migrations, collects static files, then gunicorn
  (long timeout: agent runs are synchronous).
- `frontend` is the built React app behind nginx. `VITE_API_URL` is baked
  in at build time and must be reachable from the **browser**.
- For a real domain, set `VITE_API_URL`, `DJANGO_ALLOWED_HOSTS`,
  `CORS_ALLOWED_ORIGINS` and `CSRF_TRUSTED_ORIGINS` in `.env`, and put a
  TLS-terminating reverse proxy (Caddy, nginx, a cloud load balancer) in front.

### Housekeeping

Cloned repositories accumulate in `workspace/runs/`. Remove finished runs
older than 7 days (paused runs are kept):

```bash
python manage.py cleanup_workspaces --days 7 --dry-run   # preview
python manage.py cleanup_workspaces --days 7
# Docker: docker compose exec backend python manage.py cleanup_workspaces --days 7
```

## ⚠️ Security notes

- **Rotate any API key that was ever committed or shared**, and keep real
  keys only in `.env` (git-ignored).
- The agent clones a repository and runs its tests (`pytest` / `npm test`).
  **That is arbitrary code execution.** Mitigations in place: the test
  process does not receive DevPilot's secrets (keys, tokens, DB URIs),
  the Docker backend runs as a non-root user, and nothing is pushed
  without your approval. It is **not** a full sandbox — only point
  DevPilot at repositories you trust, and run it on an isolated machine.
  Per-run containers with no network access would be the next hardening step.
- Use a fine-grained GitHub token limited to the repositories you test on.

## Known limitations / next steps

- Runs are synchronous: the HTTP request stays open while the agent
  works. Streaming progress or a background worker would improve UX.
- Retrieval is keyword-based; embeddings could be added for large repos.
- No fork workflow: pushing requires write access to the target repository.


## Security and background execution updates

The security-hardening branch adds bounded API input validation, safer API error responses,
GitHub URL validation, and fail-closed Git staging so only paths recorded by the agent are
staged for commits. LLM provider retries are bounded and provider logs record provider,
attempt, error type, and latency without printing raw exception messages.

### Background runs in Docker Compose

The Compose stack now includes Redis, a Celery worker, and Celery Beat for stale-run recovery. Compose enables
`DEVPILOT_ASYNC_RUNS=True`; the API returns a run ID immediately and the React UI polls
the persisted run status. The worker shares the workspace volume with the backend. A recovery task marks queued/running jobs older than 60 minutes as interrupted. Repository test output is capped, and the test process receives a reduced environment with a private HOME and host credential sockets removed. Local
development remains synchronous by default; set `DEVPILOT_ASYNC_RUNS=True` only when a
Celery worker and broker are running.

Start the stack with:

```bash
docker compose up -d --build
docker compose logs -f backend worker
```

Run the offline test suites in the backend environment:

```bash
python manage.py test core
pytest core/test_git_delivery.py core/test_llm_fallback.py core/test_background_tasks.py
```

**Security limitation:** These changes do not yet provide a hardened sandbox for arbitrary
repository code. The current test runner still executes project commands inside the backend
container, albeit with a reduced environment and an unprivileged container user. Do not run
untrusted repositories in a public multi-user deployment until a separate execution sandbox
with strict resource, filesystem, privilege, and network controls has been implemented and
tested.


## Repository test execution safety

The Docker Compose backend and worker select `DEVPILOT_TEST_SANDBOX=bubblewrap`. In that mode, test/build commands run with network and other namespaces unshared, a read-only container filesystem, only the current run's workspace writable, a cleared child environment, and CPU/address-space/file-size/process/file-descriptor limits. If Bubblewrap or `prlimit` is unavailable, execution fails closed; it never silently falls back to an unsandboxed command. The sandbox is defense in depth, not a guarantee against every kernel/container escape, and disk-usage quotas are not currently enforced. Do not expose arbitrary-code execution publicly without further security review and adversarial testing.

For trusted local development outside Docker Compose, set `DEVPILOT_TEST_SANDBOX=unsafe-local` explicitly. This executes commands without OS sandbox isolation and must not be used for untrusted repositories.

## LLM usage and search benchmark

Successful LLM calls log provider, model, latency, available input/output token counts, and an estimated USD cost. Configure `LLM_COST_<PROVIDER>_INPUT_PER_1M` and `LLM_COST_<PROVIDER>_OUTPUT_PER_1M` using the provider's current pricing. Zero is the default, so cost remains an estimate and is not a persisted billing ledger; calls without token metadata are logged as unknown.

Run a baseline for the current BM25-style keyword search from the repository root:

```bash
python backend/scripts/benchmark_code_search.py ./path/to/repository --repeats 5
```

Save baseline results before changing retrieval. The script reports latency and top paths; relevance still needs a manually labeled query set before comparing lexical search with hybrid/semantic retrieval.
