"""
DevPilot AI - LangGraph agent workflow (multi-step edition).

What changed compared with the single-edit version:

  * After every approved edit the graph returns to the agent, so a task that
    needs several files (button + CSS + tests + .gitignore) can be finished.
  * The agent ends its work by replying "DONE". Only then do tests run.
  * A text-only reply no longer loops silently: the agent is nudged, and the
    run ends cleanly if it keeps stalling.
  * A failing test/build goes back to the agent (any ecosystem, not only
    pytest) with the real output, up to MAX_DEBUG_ATTEMPTS times.
  * Several proposals in one turn are handled: the first goes to approval,
    the agent is told to re-propose the rest one at a time.
  * Commit stages only files the agent changed (see tools.git_commit note).
  * Reports distinguish "tests passed" from "only the build passed".
"""

import os
import atexit
import ast
import re

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "config.settings"
)

import django

django.setup()

from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt


from core.tools import (
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    propose_new_file,
    git_diff,
    git_status,
    create_agent_branch,
    write_file,
    write_new_file,
    run_tests,
    git_commit,
    git_push,
    create_pull_request,
    set_github_token,
    reset_github_token,
    set_workspace,
    reset_workspace,
    safe_workspace_path,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Agent turns allowed per run (resets after a failed test round).
MAX_AGENT_STEPS = 40

# How many times a failing test/build is sent back to the agent.
MAX_DEBUG_ATTEMPTS = 3

# How many times a text-only reply is nudged before the run ends.
MAX_NUDGES = 2

# LangGraph counts every node execution. A multi-file run needs far more
# than the default of 25, so pass this in the config of agent.invoke/stream
# (the compiled graph below also tries to carry it as a default).
RECURSION_LIMIT = 200

PROPOSAL_TOOLS = {
    "propose_file_change",
    "propose_new_file",
}


# ============================================================
# 2. AGENT STATE
# ============================================================

class AgentState(TypedDict):

    # Run identity
    thread_id: str

    # Original user request
    task: str

    # Conversation/tool history
    messages: Annotated[
        list[AnyMessage],
        add_messages
    ]

    # Human approval
    approval: str

    # Current proposed file change
    proposed_path: str
    proposed_content: str

    # "existing_file" or "new_file"
    proposal_type: str

    # Git branch operation
    branch_result: str

    # Test/debug information
    test_result: str
    debug_result: str
    final_result: str

    # Kept for API compatibility with the views/UI
    failed_file: str

    # Delivery (commit / push / pull request)
    commit_result: str
    push_approval: str
    push_result: str
    pr_url: str
    pr_result: str

    # GitHub App installation used for this run (0 = single GITHUB_TOKEN mode)
    github_installation_id: int

    # Workspace used by this run
    workspace_path: str

    # Safety limits
    agent_steps: int
    debug_attempts: int

    # Multi-step support
    applied_files: list[str]
    nudges: int


# ============================================================
# 3-5. LLM PROVIDERS, ERROR DETECTION AND FALLBACK
#
# See core/llm_providers.py. Providers (OpenRouter, Gemini,
# Hugging Face) are built from environment variables and tried in
# LLM_PROVIDER_ORDER; rate limits / outages fall through to the
# next provider automatically.
# ============================================================

from core import github_app
from core.llm_providers import (
    is_transient_provider_error,
    invoke_with_fallback,
)


def invoke_llm(messages, tools=None):
    """Invoke the configured LLM providers with automatic fallback."""
    return invoke_with_fallback(messages, tools)


# ============================================================
# 6. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are DevPilot AI, an AI software engineering agent.

Your job is to understand the user's software-development task,
inspect the provided repository, and complete the task by proposing
code changes ONE FILE AT A TIME. Every change needs human approval.
After you finish, the system runs the tests, commits, asks the human
for push approval, pushes, and opens a pull request.

============================================================
AVAILABLE TOOLS
============================================================

1. list_dir
Understand the project structure (framework, language, tests,
configuration). Do not list the same directory twice.

2. search_code
Find relevant files, functions, classes, imports, routes, tests.
Prefer one specific query (an identifier or filename). Do not repeat
a search once the information is established.

3. read_file
Read the exact contents of a file. Always read an existing file before
proposing a modification, and read each file only once.

4. propose_file_change
Modify an EXISTING file.
- First read the target file.
- old_text must exactly match existing text.
- new_text contains only the replacement section.
- Do not provide the entire file as old_text.
- Do not provide a unified diff.
- The tool only creates a proposal; a human must approve it.

5. propose_new_file
Create a NEW file.
- First verify the file does not exist.
- Follow the repository's language/framework conventions.
- Provide the path and the complete content.
- The tool only creates a proposal; a human must approve it.

6. git_diff
Inspect current repository changes.

7. git_status
Inspect the working-tree status.

Do not modify the repository with git commands. You never commit,
push, or create branches yourself.

============================================================
MULTI-STEP TASKS
============================================================

1. Before the first edit, write a short numbered plan listing every
   file you will change or create.
2. Call exactly ONE propose_file_change or propose_new_file per turn.
   Never two in the same turn.
3. After each approved change, the system applies it and tells you.
   Continue with the next item of your plan.
4. When every item is complete, reply with the word DONE as the first
   word, followed by a one-line summary. Never reply DONE while a
   requested part is missing.
5. If the task asks for tests and the project has no test setup, add
   the test framework (devDependencies, a "test" script, config) and
   the test file to your plan.
6. If the repository has no .gitignore, create one first
   (node_modules, dist, .env, *.pem).
7. You do not run tests. The system runs them after you reply DONE and
   will show you the real failure output if they fail.
8. If the task is impossible with the available tools, explain why in
   plain text and start your reply with DONE.

============================================================
TASK FIDELITY
============================================================

The user's request is the source of truth. Preserve the requested
programming language, framework, file type, filename, feature,
behavior, and scope. NEVER silently change the user's request.

Example:
User: "Add a hello() function to a Python file."
Repository: React/Vite, no Python files.
Correct: verify no suitable Python file exists; do not switch to
JavaScript; do not modify App.jsx; if creating the requested Python
file satisfies the request, use propose_new_file; otherwise report
the limitation and reply DONE.
Incorrect: adding hello() to App.jsx or any unrelated file.

If the user explicitly requests Python, JavaScript, Django or React,
do not replace it with another language or framework. If the user does
not specify, use the language/framework already used by the relevant
part of the repository.

============================================================
REPOSITORY INVESTIGATION
============================================================

Before making a change:
1. Understand the repository structure.
2. Identify the relevant language/framework.
3. Find and read the relevant files.
4. Determine the safe modification.

Once you have enough information, STOP INVESTIGATING and start
proposing changes. Do not spend your steps on repeated reads/searches.

============================================================
WHEN THE REQUESTED FILE DOES NOT EXIST
============================================================

If the user asks to modify a file that does not exist, first determine
whether another existing file is clearly intended. If not, and creating
a new file satisfies the request, use propose_new_file. Do not modify
an unrelated existing file.

============================================================
BEFORE PROPOSING A CHANGE
============================================================

Verify:
1. The target file is correct.
2. An existing target file was read.
3. The language and framework match the request.
4. The change directly serves the task.
5. Unrelated functionality is preserved.
6. The change differs from the current file.

If any check fails, do not propose the change.

============================================================
MINIMAL CHANGE PRINCIPLE
============================================================

Make the smallest reasonable change in each file. A task may need
several files; that is fine. Do not rewrite files unnecessarily, refactor
unrelated code, rename unrelated variables, or change architecture
without permission. Do not change dependencies unless the task needs
them (for example, adding a test framework when tests are requested).

============================================================
TESTS AND FAILURES
============================================================

After you reply DONE the system runs the project's own test command
(or the build if there is no test script). If it fails, you receive the
real output. Then:
1. Read the failing file(s) named in the output.
2. Fix the actual cause with the smallest change.
3. Do not invent files, functions, or APIs.
4. Do not edit a test just to make it pass.
5. Do not replace a placeholder assertion with a meaningless one.
6. Reply DONE when the failure is fixed.

============================================================
HUMAN-IN-THE-LOOP
============================================================

You never modify files directly. Every modification goes through:
proposal -> human approval -> branch creation (first time only) ->
write -> back to you for the next step.

Never reuse an old approval for a new proposal.

============================================================
STOP CONDITIONS
============================================================

Stop (reply DONE) when:
1. Every part of the task is complete.
2. The task cannot safely be completed, and you have explained why.
3. A human rejected a change (the run ends automatically).

Never loop indefinitely.
"""


# ============================================================
# 7. HELPERS
# ============================================================

def message_text(content) -> str:
    """Normalize LangChain message content into plain text."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                parts.append(item.get("text", ""))

            elif isinstance(item, str):
                parts.append(item)

        return "\n".join(parts)

    return str(content)


def _trailing_tool_messages(state: AgentState):
    """ToolMessages produced by the latest round of tool calls."""

    collected = []

    for message in reversed(state["messages"]):

        if isinstance(message, ToolMessage):
            collected.append(message)
        else:
            break

    return list(reversed(collected))


def _parse_proposal(content: str):
    """Return a proposal dict from a propose_* tool result, or None."""

    if not content:
        return None

    try:
        result = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(result, dict):
        return None

    status = result.get("status")

    if status not in {
        "pending_approval",
        "pending_new_file_approval",
    }:
        return None

    path = str(result.get("path", "")).strip()
    proposed = result.get("proposed_content", "")

    if not path or not isinstance(proposed, str) or not proposed:
        return None

    return {
        "proposed_path": path,
        "proposed_content": proposed,
        "proposal_type": (
            "new_file"
            if status == "pending_new_file_approval"
            else "existing_file"
        ),
    }


def _write_succeeded(result) -> bool:

    if isinstance(result, dict):
        return result.get("status") == "success"

    return "'status': 'success'" in str(result)


def _verified_by_tests(test_result: str) -> bool:
    """
    True only when a real test command passed.

    A passing `npm run build` is NOT a test run, so it must not be
    reported as "tests passed".
    """

    if "'status': 'passed'" not in test_result:
        return False

    match = re.search(r"'command': \[([^\]]*)\]", test_result)

    if not match:
        return False

    tokens = re.findall(r"'([^']*)'", match.group(1))

    return any("test" in token.lower() for token in tokens)


# ============================================================
# 8. MAIN AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    workspace_token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        step_count = state.get("agent_steps", 0) + 1

        task = state.get("task", "")

        messages = [
            SystemMessage(
                content=(
                    SYSTEM_PROMPT
                    + "\n\n"
                    + "CURRENT USER TASK:\n"
                    + task
                )
            ),
            *state["messages"],
        ]

        response = invoke_llm(
            messages,
            tools=[
                search_code,
                read_file,
                list_dir,
                propose_file_change,
                propose_new_file,
                git_diff,
                git_status,
            ],
        )

        return {
            "messages": [response],
            "agent_steps": step_count,
        }

    finally:

        reset_workspace(workspace_token)


# ============================================================
# 9. TOOL NODE
# ============================================================

tool_executor = ToolNode([
    search_code,
    read_file,
    list_dir,
    propose_file_change,
    propose_new_file,
    git_diff,
    git_status,
])


def tool_node(state: AgentState):

    workspace_token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        return tool_executor.invoke(state)

    finally:

        reset_workspace(workspace_token)


# ============================================================
# 10. TOOL ROUTING
# ============================================================

def route_after_tools(state: AgentState):
    """
    If any tool in the latest round produced a proposal, extract it.
    Otherwise return to the agent.
    """

    for message in _trailing_tool_messages(state):

        if getattr(message, "name", "") in PROPOSAL_TOOLS:
            return "extract_proposal"

    return "agent"


# ============================================================
# 11. EXTRACT FILE PROPOSAL
# ============================================================

def extract_proposal_node(state: AgentState):
    """
    Take the first valid proposal from the latest tool round.

    If the model proposed several files in one turn, only the first is
    sent for approval and the agent is told to re-propose the rest.
    """

    proposals = []

    for message in _trailing_tool_messages(state):

        if getattr(message, "name", "") not in PROPOSAL_TOOLS:
            continue

        parsed = _parse_proposal(message_text(message.content))

        if parsed:
            proposals.append(parsed)

    if not proposals:

        return {
            "proposed_path": "",
            "proposed_content": "",
            "proposal_type": "",
        }

    update = dict(proposals[0])

    if len(proposals) > 1:

        skipped = ", ".join(p["proposed_path"] for p in proposals[1:])

        update["messages"] = [
            HumanMessage(
                content=(
                    f"Only the first proposal ({proposals[0]['proposed_path']}) "
                    "was submitted for approval. These proposals were NOT "
                    f"submitted: {skipped}. Propose them again, one per turn, "
                    "after the current change is applied."
                )
            )
        ]

    return update


def route_after_extract_proposal(state: AgentState):

    if (
        state.get("proposed_path")
        and isinstance(state.get("proposed_content"), str)
        and state.get("proposed_content")
    ):
        return "approval"

    return "agent"


# ============================================================
# 12. AGENT ROUTING
# ============================================================

def should_continue(state: AgentState):
    """
    tools  -> the agent called a tool
    finish -> the agent said DONE (or stalled) after applying changes
    stop   -> nothing was changed; end the run with the agent's message
    nudge  -> the agent replied with text only; ask it to continue
    """

    last_message = state["messages"][-1]

    applied = state.get("applied_files", [])

    # Tool calls have priority.
    if getattr(last_message, "tool_calls", None):
        return "tools"

    text = message_text(last_message.content).strip()

    # Explicit completion signal.
    if text.upper().startswith("DONE"):
        return "finish" if applied else "stop"

    # Stalling or out of budget.
    if (
        state.get("nudges", 0) >= MAX_NUDGES
        or state.get("agent_steps", 0) >= MAX_AGENT_STEPS
    ):
        return "finish" if applied else "stop"

    return "nudge"


def nudge_node(state: AgentState):

    return {
        "nudges": state.get("nudges", 0) + 1,
        "messages": [
            HumanMessage(
                content=(
                    "Continue. Call a tool for the next part of the task, "
                    "or reply DONE if every part is complete. If the task "
                    "cannot be done, explain why and reply DONE."
                )
            )
        ],
    }


def stop_node(state: AgentState):

    last_message = state["messages"][-1]

    text = message_text(last_message.content).strip()

    return {
        "final_result": (
            text
            or "The agent stopped without proposing a change."
        )
    }


# ============================================================
# 13. HUMAN APPROVAL NODE
# ============================================================

def approval_node(state: AgentState):

    proposed_path = state.get("proposed_path", "")

    proposed_content = state.get("proposed_content", "")

    if (
        not proposed_path
        or not isinstance(proposed_content, str)
    ):
        return {"approval": "reject"}

    # If the graph is resumed after interrupt(), LangGraph supplies the
    # new decision. The write node resets approval after consuming it,
    # so a later proposal cannot reuse the previous decision.
    if state.get("approval") in ("approve", "reject"):
        return {"approval": state["approval"]}

    # Original content lets the UI render a real diff (read-only).
    original_content = ""

    _token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        _target = safe_workspace_path(proposed_path)

        if _target is not None and _target.is_file():

            original_content = _target.read_text(
                encoding="utf-8",
                errors="ignore",
            )

    finally:

        reset_workspace(_token)

    decision = interrupt({

        "type": "code_change",

        "original_content": original_content,

        "question": "Do you approve this code change?",

        "message": (
            "DevPilot AI is requesting human approval "
            "before modifying files."
        ),

        "path": proposed_path,

        "content": proposed_content,
    })

    return {"approval": decision}


# ============================================================
# 14. REJECTION NODE
# ============================================================

def rejected_node(state: AgentState):

    applied = state.get("applied_files", [])

    if applied:

        message = (
            "Human rejected the proposed change. The run was stopped. "
            "Changes already applied in the workspace (not committed): "
            + ", ".join(applied)
        )

    else:

        message = (
            "Human rejected the proposed change. "
            "No files were modified."
        )

    return {
        "final_result": message,
        "test_result": "",
        "debug_result": "",
    }


# ============================================================
# 15. APPROVAL ROUTING
# ============================================================

def route_after_approval(state: AgentState):

    if state.get("approval") == "approve":
        return "create_branch"

    return "rejected"


# ============================================================
# 16. CREATE GIT BRANCH
# ============================================================

def create_branch_node(state: AgentState):
    """
    Create a dedicated Git branch for this DevPilot run.

    Runs only once per run: later approved edits reuse the branch.
    Controlled by LangGraph; the LLM cannot call it.
    """

    if state.get("branch_result", "").startswith("success:"):
        return {}

    thread_id = state.get("thread_id", "")

    workspace_path = state.get("workspace_path", "")

    if not thread_id:

        return {
            "branch_result": "error: thread ID is missing.",
            "final_result": "Could not create the branch: thread ID is missing.",
        }

    if not workspace_path:

        return {
            "branch_result": "error: workspace path is missing.",
            "final_result": "Could not create the branch: workspace path is missing.",
        }

    branch_name = f"devpilot/{thread_id}"

    workspace_token = set_workspace(workspace_path)

    try:

        result = create_agent_branch.invoke({
            "branch_name": branch_name
        })

        if (
            not isinstance(result, dict)
            or result.get("status") != "success"
        ):

            return {
                "branch_result": f"error: {result}",
                "final_result": f"Could not create the branch: {result}",
            }

        return {"branch_result": f"success: {branch_name}"}

    except Exception as exc:

        return {
            "branch_result": f"error: {exc}",
            "final_result": f"Could not create the branch: {exc}",
        }

    finally:

        reset_workspace(workspace_token)


# ============================================================
# 17. BRANCH ROUTING
# ============================================================

def route_after_branch(state: AgentState):

    if state.get("branch_result", "").startswith("success:"):
        return "write"

    return END


# ============================================================
# 18. WRITE NODE (returns to the agent for the next step)
# ============================================================

def write_node(state: AgentState):

    path = state.get("proposed_path", "")

    content = state.get("proposed_content", "")

    workspace_token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        if state.get("proposal_type", "existing_file") == "new_file":

            result = write_new_file.invoke({
                "path": path,
                "content": content,
            })

        else:

            result = write_file.invoke({
                "path": path,
                "content": content,
            })

    except Exception as exc:

        result = {"status": "error", "message": str(exc)}

    finally:

        reset_workspace(workspace_token)

    ok = _write_succeeded(result)

    applied = list(state.get("applied_files", []))

    if ok and path not in applied:
        applied.append(path)

    if ok:

        note = (
            f"Applied the change to {path}. Files changed so far: "
            f"{', '.join(applied)}. If any part of the task is still "
            "missing, propose the next change (one file at a time). "
            "If everything is complete, reply DONE."
        )

    else:

        note = (
            f"Writing {path} FAILED: {result}. Fix the problem and "
            "propose the change again, or reply DONE and explain why "
            "it cannot be done."
        )

    previous = state.get("final_result", "")

    return {

        "final_result": f"{previous}\n{result}".strip(),

        "applied_files": applied,

        "nudges": 0,

        # Consume the current approval decision and proposal.
        "approval": "",
        "proposed_path": "",
        "proposed_content": "",
        "proposal_type": "",

        "messages": [HumanMessage(content=note)],
    }


# ============================================================
# 19. TEST NODE
# ============================================================

def test_node(state: AgentState):

    workspace_token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        result = run_tests.invoke({})

        return {"test_result": str(result)}

    finally:

        reset_workspace(workspace_token)


# ============================================================
# 20. TEST RESULT ROUTING
# ============================================================

def route_after_test(state: AgentState):

    test_result = state.get("test_result", "")

    # "no_tests" means there was nothing to run (e.g. a new repo). That
    # must not block delivery; the push approval warns that nothing was
    # verified.
    if (
        "'status': 'passed'" in test_result
        or "'status': 'no_tests'" in test_result
    ):
        return "commit"

    return "test_failed"


def test_failed_node(state: AgentState):
    """
    Send the real failure output back to the agent (any ecosystem).
    """

    attempts = state.get("debug_attempts", 0) + 1

    if attempts > MAX_DEBUG_ATTEMPTS:

        previous = state.get("final_result", "")

        return {
            "debug_attempts": attempts,
            "debug_result": (
                "Maximum debug attempts reached. DevPilot stopped the "
                "automatic fix loop."
            ),
            "final_result": (
                previous
                + "\nTests/build still failing after the maximum number "
                "of fix attempts. Nothing was pushed."
            ).strip(),
        }

    output = str(state.get("test_result", ""))[-6000:]

    return {

        "debug_attempts": attempts,

        "debug_result": (
            f"Attempt {attempts}/{MAX_DEBUG_ATTEMPTS}: the test/build "
            "step failed. The agent is working on a fix."
        ),

        "failed_file": "",

        # Fresh budget for the fix round.
        "agent_steps": 0,
        "nudges": 0,

        "messages": [
            HumanMessage(
                content=(
                    "The test/build step FAILED. Real output:\n\n"
                    + output
                    + "\n\nRead the failing files named in the output, "
                    "then fix the cause with the smallest change, one "
                    "propose_* call at a time. Do not edit a test just to "
                    "make it pass. Reply DONE when the failure is fixed."
                )
            )
        ],
    }


def route_after_test_failed(state: AgentState):

    if state.get("debug_attempts", 0) <= MAX_DEBUG_ATTEMPTS:
        return "agent"

    return END


# ============================================================
# 21. DELIVERY: COMMIT -> PUSH APPROVAL -> PUSH -> PULL REQUEST
#
# Commit is local and safe (agent branch only). Push and PR leave
# the machine, so they require a SECOND human approval.
# ============================================================

def _commit_message(state: AgentState) -> str:

    task = " ".join(state.get("task", "").split())[:60] or "automated change"

    return f"DevPilot: {task}"


def _verification_sentence(test_result: str) -> str:

    if _verified_by_tests(test_result):
        return "Automated tests passed before this branch was pushed. "

    if "'status': 'passed'" in test_result:
        return (
            "**Only the build passed; no automated tests were run, so the "
            "behavior is not verified.** "
        )

    return (
        "**No automated tests were found, so this change is not "
        "verified.** "
    )


def commit_node(state: AgentState):

    token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    try:

        # "files" limits staging to what the agent changed. If your
        # git_commit tool does not accept it yet, update it (see notes);
        # unknown keys are ignored by the tool's input schema.
        result = git_commit.invoke({
            "message": _commit_message(state),
            "files": state.get("applied_files", []),
        })

        return {"commit_result": str(result)}

    finally:

        reset_workspace(token)


def route_after_commit(state: AgentState):

    if "'status': 'success'" in state.get("commit_result", ""):
        return "push_approval"

    return END


def push_approval_node(state: AgentState):
    # NOTE: interrupt() re-runs this node on resume, so it must stay free
    # of side effects.

    test_result = state.get("test_result", "")

    tested = _verified_by_tests(test_result)

    built = "'status': 'passed'" in test_result

    if tested:
        verification = "passed"
        message = (
            "Tests passed and changes are committed locally. "
            "Approve to push the branch and open a pull request."
        )
    elif built:
        verification = "build_only"
        message = (
            "The build passed but no automated tests were run, so the "
            "behavior is NOT verified. Changes are committed locally. "
            "Approve to push the branch and open a pull request, then "
            "review it carefully on GitHub."
        )
    else:
        verification = "no_tests"
        message = (
            "No automated tests were found, so this change is NOT "
            "verified. It is committed locally. Approve to push the "
            "branch and open a pull request, then review it carefully "
            "on GitHub."
        )

    decision = interrupt({
        "type": "push_approval",
        "question": "Push this branch and open a pull request?",
        "tests": verification,
        "message": message,
        "commit": state.get("commit_result", ""),
        "branch": state.get("branch_result", ""),
        "files": state.get("applied_files", []),
    })

    return {"push_approval": decision}


def route_after_push_approval(state: AgentState):

    if state.get("push_approval") == "approve":
        return "push"

    return END


def _github_token_for(state: AgentState):
    """
    Fresh GitHub App token for this run, or None to use GITHUB_TOKEN.

    Minted right before use (never stored in the checkpoint) because the
    run may have waited hours for approval and installation tokens last
    about an hour.
    """

    installation_id = state.get("github_installation_id") or 0

    if not installation_id:
        return None

    repo_name = Path(state.get("workspace_path", "")).name or None

    return github_app.installation_token(installation_id, repo_name)


def _error_result(message: str) -> str:

    return str({"status": "error", "message": message})


def push_node(state: AgentState):

    try:
        github_token = _github_token_for(state)
    except Exception as exc:
        return {"push_result": _error_result(str(exc))}

    token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    gh_ctx = (
        set_github_token(github_token)
        if github_token
        else None
    )

    try:

        result = git_push.invoke({})

        return {"push_result": str(result)}

    finally:

        if gh_ctx is not None:
            reset_github_token(gh_ctx)

        reset_workspace(token)


def route_after_push(state: AgentState):

    if "'status': 'success'" in state.get("push_result", ""):
        return "pull_request"

    return END


def pull_request_node(state: AgentState):

    try:
        github_token = _github_token_for(state)
    except Exception as exc:

        message = _error_result(str(exc))

        return {
            "pr_result": message,
            "final_result": (
                f"{state.get('final_result', '')}\nPR: {message}"
            ),
        }

    token = set_workspace(
        state.get("workspace_path", "workspace")
    )

    gh_ctx = (
        set_github_token(github_token)
        if github_token
        else None
    )

    try:

        task = state.get("task", "")

        files = state.get("applied_files", [])

        files_line = (
            "**Files changed:** " + ", ".join(files) + "\n\n"
            if files
            else ""
        )

        result = create_pull_request.invoke({
            "title": _commit_message(state),
            "body": (
                "Automated change proposed by DevPilot AI.\n\n"
                f"**Task:** {task}\n\n"
                + files_line
                + _verification_sentence(state.get("test_result", ""))
                + "Please review before merging."
            ),
        })

        pr_url = (
            result.get("url", "")
            if isinstance(result, dict)
            else ""
        )

        return {
            "pr_url": pr_url,
            "pr_result": str(result),
            "final_result": (
                f"{state.get('final_result', '')}\nPR: {result}"
            ),
        }

    finally:

        if gh_ctx is not None:
            reset_github_token(gh_ctx)

        reset_workspace(token)


# ============================================================
# 22. GRAPH BUILDER AND NODES
# ============================================================

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("nudge", nudge_node)
graph_builder.add_node("stop", stop_node)
graph_builder.add_node("extract_proposal", extract_proposal_node)
graph_builder.add_node("approval", approval_node)
graph_builder.add_node("rejected", rejected_node)
graph_builder.add_node("create_branch", create_branch_node)
graph_builder.add_node("write", write_node)
graph_builder.add_node("test", test_node)
graph_builder.add_node("test_failed", test_failed_node)
graph_builder.add_node("commit", commit_node)
graph_builder.add_node("push_approval", push_approval_node)
graph_builder.add_node("push", push_node)
graph_builder.add_node("pull_request", pull_request_node)


# ============================================================
# 23. EDGES
# ============================================================

graph_builder.add_edge(START, "agent")

# Agent -> tools / nudge / test (finished) / stop
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "nudge": "nudge",
        "finish": "test",
        "stop": "stop",
    },
)

graph_builder.add_edge("nudge", "agent")
graph_builder.add_edge("stop", END)

# Tools -> agent / extract proposal
graph_builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "agent": "agent",
        "extract_proposal": "extract_proposal",
    },
)

# Extract proposal -> approval / agent
graph_builder.add_conditional_edges(
    "extract_proposal",
    route_after_extract_proposal,
    {
        "approval": "approval",
        "agent": "agent",
    },
)

# Approval -> create branch / rejected
graph_builder.add_conditional_edges(
    "approval",
    route_after_approval,
    {
        "create_branch": "create_branch",
        "rejected": "rejected",
    },
)

graph_builder.add_edge("rejected", END)

# Create branch -> write / end
graph_builder.add_conditional_edges(
    "create_branch",
    route_after_branch,
    {
        "write": "write",
        END: END,
    },
)

# Write -> back to the agent for the next step
graph_builder.add_edge("write", "agent")

# Test -> commit / failure loop
graph_builder.add_conditional_edges(
    "test",
    route_after_test,
    {
        "commit": "commit",
        "test_failed": "test_failed",
    },
)

graph_builder.add_conditional_edges(
    "test_failed",
    route_after_test_failed,
    {
        "agent": "agent",
        END: END,
    },
)

# Delivery
graph_builder.add_conditional_edges(
    "commit",
    route_after_commit,
    {
        "push_approval": "push_approval",
        END: END,
    },
)

graph_builder.add_conditional_edges(
    "push_approval",
    route_after_push_approval,
    {
        "push": "push",
        END: END,
    },
)

graph_builder.add_conditional_edges(
    "push",
    route_after_push,
    {
        "pull_request": "pull_request",
        END: END,
    },
)

graph_builder.add_edge("pull_request", END)


# ============================================================
# 24. CHECKPOINTER
# ============================================================

def create_checkpointer():
    """
    Create a persistent PostgreSQL-backed LangGraph checkpoint store.
    """

    uri = os.environ["LANGGRAPH_POSTGRES_URI"]

    pool = ConnectionPool(
        conninfo=uri,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    )

    checkpointer = PostgresSaver(pool)

    # Idempotent: creates the checkpoint tables on a fresh database.
    checkpointer.setup()

    return (pool, checkpointer)


checkpoint_pool, checkpointer = create_checkpointer()

atexit.register(checkpoint_pool.close)


# ============================================================
# 25. COMPILE GRAPH
# ============================================================

_compiled = graph_builder.compile(
    checkpointer=checkpointer
)

# Carry a higher default recursion limit when this LangGraph version
# supports it. If the returned object lacks get_state, fall back to the
# plain compiled graph and pass RECURSION_LIMIT in the config instead.
try:

    _configured = _compiled.with_config(
        {"recursion_limit": RECURSION_LIMIT}
    )

    agent = (
        _configured
        if hasattr(_configured, "get_state")
        else _compiled
    )

except Exception:

    agent = _compiled


# ============================================================
# 26. GRAPH FLOW
# ============================================================

"""
DevPilot AI multi-step workflow:

      START
        |
        v
    +-------+   tool call    +-------+
    | AGENT | -------------> | TOOLS |
    +-------+ <------------- +-------+
     ^  |  ^      (read/search)    |
     |  |  |                       | propose_* result
     |  |  | text only             v
     |  |  +---- NUDGE        EXTRACT PROPOSAL
     |  |                          |
     |  | "DONE" (changes applied) v
     |  |                      APPROVAL --reject--> REJECTED --> END
     |  |                          |
     |  |                       approve
     |  |                          v
     |  |                   CREATE BRANCH (first time only)
     |  |                          |
     |  |                          v
     |  |                        WRITE
     |  |                          |
     |  +--------------------------+   (back to the agent: next step)
     |
     |  "DONE" -> TEST
     |              |
     |      fail    |    pass / no tests
     +---- TEST_FAILED         |
      (real output,            v
       max 3 rounds)        COMMIT
                               |
                               v
                         PUSH APPROVAL --reject--> END
                               |
                            approve
                               v
                             PUSH
                               |
                               v
                         PULL REQUEST --> END

Safety:
- Every file change needs human approval.
- Branch creation is controlled by LangGraph, never by the LLM.
- main/master can never be committed to or pushed.
- Push + PR need a second human approval.
- Limits: MAX_AGENT_STEPS, MAX_NUDGES, MAX_DEBUG_ATTEMPTS, RECURSION_LIMIT.
"""


# ============================================================
# 27. DIRECT EXECUTION TEST
# ============================================================

if __name__ == "__main__":

    print("DevPilot AI agent loaded successfully.")