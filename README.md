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

## Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then put your real GEMINI_API_KEY in .env
python manage.py migrate
python manage.py runserver
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Testing

Automated (no Gemini key needed — deterministic graph logic, sandboxed
tools, and endpoint validation only):
```bash
cd backend
python manage.py test core
```

Manual, full end-to-end with a real model (needs `GEMINI_API_KEY` in `.env`):
```bash
cd backend
python -m core.run_agent
```
Or drive it over HTTP once `runserver` is up:
```bash
curl -X POST http://127.0.0.1:8000/api/agent/run/ \
  -H "Content-Type: application/json" \
  -d '{"task": "Fix the login bug in auth.py"}'

curl -X POST http://127.0.0.1:8000/api/agent/resume/ \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<from previous response>", "decision": "approve"}'
```

## ⚠️ Security note

The uploaded `.env` contained a live Gemini API key. Treat it as
exposed and **rotate it** in Google AI Studio, then put the new key
in `backend/.env` (never `.env.example`, and never commit `.env` —
it's already in `.gitignore`).

## Known limitations / good next steps

- The LangGraph checkpointer is `InMemorySaver` — runs are lost on
  server restart and won't work across multiple worker processes.
  Fine for local dev; swap for `SqliteSaver`/`PostgresSaver` before
  deploying for real.
- The frontend (`App.jsx`) only calls `/api/ai/chat/` — it hasn't been
  wired to the new `/api/agent/run/` and `/api/agent/resume/`
  endpoints yet (out of scope here since you asked specifically for
  the backend).


## Current project status

The core portfolio MVP is implemented:

- React frontend with authentication and agent controls
- Django REST API with JWT authentication
- LangGraph orchestration
- Gemini as the primary LLM with Hugging Face fallback for transient provider failures
- Tool calling for workspace inspection and controlled code changes
- Human-in-the-loop approval before file writes
- Real pytest execution after changes
- Automatic failure detection and AI-assisted debug/fix flow
- A second human approval checkpoint for AI-generated fixes
- PostgreSQL LangGraph checkpoint persistence
- Per-user agent thread ownership checks
- Workspace path traversal protection
- Automated Django tests for deterministic agent behavior and API validation

The main engineering story is that DevPilot is not only a code generator. It follows an inspect -> propose -> approve -> modify -> test -> debug -> approve -> retry workflow, with persistent state and controlled tools.

## Final local verification

Because API credentials and the local PostgreSQL service are intentionally not stored in Git, run the following once on the development machine:

```powershell
cd C:\DevPilot-AI\backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py test
python -c "import core.agent; print('agent imported successfully')"

cd ..\frontend
npm install
npm run build
```

Then start PostgreSQL and Django and test one real task from the UI.

`backend/.env.example` contains the required environment variable names. Never commit the real `.env` or API tokens.
