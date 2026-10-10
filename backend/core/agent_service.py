import re
import subprocess
import uuid

from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agent import agent


from django.conf import settings

from core import github_app
from core.tools import github_auth_args

WORKSPACE_ROOT = Path(settings.DEVPILOT_WORKSPACE_ROOT).resolve()


class ThreadOwnershipError(Exception):
    pass


def new_thread_id() -> str:
    return str(uuid.uuid4())


def _workspace_for_thread(thread_id: str) -> Path:
    return WORKSPACE_ROOT / "runs" / str(thread_id)


def _validate_github_url(repo_url: str) -> tuple[str, str]:
    parsed = urlparse(repo_url.strip())

    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("repo_url must be a GitHub HTTPS repository URL")

    match = re.fullmatch(
        r"/([^/]+)/([^/]+?)(?:\.git)?/?",
        parsed.path,
    )

    if not match:
        raise ValueError(
            "repo_url must look like "
            "https://github.com/owner/repository"
        )

    owner, repo_name = match.groups()

    return owner, repo_name


def _prepare_repository(repo_url: str, thread_id: str, token: str | None = None) -> str:
    owner, repo_name = _validate_github_url(repo_url)

    run_workspace = _workspace_for_thread(thread_id)
    repo_workspace = run_workspace / repo_name

    run_workspace.mkdir(parents=True, exist_ok=True)

    if repo_workspace.exists():
        if not (repo_workspace / ".git").exists():
            raise ValueError(
                "Workspace already exists but is not a Git repository"
            )

        return str(repo_workspace)

    # With a GitHub App token the clone also works for private repositories.
    # The credential is passed to this one git process only (see
    # github_auth_args), so it is never written to .git/config.
    auth = github_auth_args(token) if token else []

    try:
        subprocess.run(
            [
                "git",
                *auth,
                "clone",
                "--depth",
                "1",
                repo_url,
                str(repo_workspace),
            ],
            cwd=run_workspace,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        if token:
            detail = detail.replace(token, "***")
        raise ValueError(f"Unable to clone GitHub repository: {detail}") from exc

    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            "GitHub repository clone timed out after 120 seconds"
        ) from exc

    return str(repo_workspace)


def create_agent_run(user, thread_id: str, task: str, repo_url: str = ""):
    from core.models import AgentRun

    return AgentRun.objects.create(
        user=user,
        thread_id=uuid.UUID(str(thread_id)),
        task=task,
        repo_url=repo_url,
    )


def record_run_outcome(thread_id: str, payload: dict | None = None, error: str = ""):
    """Persist the latest status of a run so it shows up in run history."""
    from core.models import AgentRun

    try:
        run = AgentRun.objects.get(thread_id=uuid.UUID(str(thread_id)))
    except (AgentRun.DoesNotExist, ValueError):
        return

    if error:
        run.status = "error"
        run.summary = error[:2000]
    elif payload:
        run.status = payload.get("status", run.status)
        if payload.get("pr_url"):
            run.pr_url = payload["pr_url"]
        # A failure reason is more useful in history than the last tool output.
        text = payload.get("error") or payload.get("final_result") or ""
        if text:
            run.summary = text[:2000]

    run.save()


def get_pending_approval(thread_id: str):
    """
    Return the approval request a paused run is waiting on, or None.
    Lets the UI re-open an approval after a page reload.
    """
    try:
        snapshot = agent.get_state({"configurable": {"thread_id": str(thread_id)}})

        for task in snapshot.tasks:
            for interrupt_ in getattr(task, "interrupts", ()):
                return interrupt_.value

    except Exception:
        return None

    return None


def validate_owned_thread(user, thread_id: str) -> str:
    from core.models import AgentRun

    try:
        parsed = uuid.UUID(str(thread_id))

    except (ValueError, AttributeError, TypeError):
        raise ValueError("thread_id must be a valid UUID")

    try:
        run = AgentRun.objects.get(thread_id=parsed)

    except AgentRun.DoesNotExist:
        raise ThreadOwnershipError("Agent run not found")

    if run.user_id != user.id:
        raise ThreadOwnershipError(
            "You do not have access to this agent run"
        )

    return str(run.thread_id)


def _initial_state(
    task: str,
    workspace_path: str,
    thread_id: str,
    installation_id: int = 0,
) -> dict:

    return {
        "thread_id": thread_id,
        "task": task,
        "messages": [HumanMessage(content=task)],
        "approval": "",
        "proposed_path": "",
        "proposed_content": "",
        "proposal_type": "",
        "branch_result": "",
        "test_result": "",
        "debug_result": "",
        "final_result": "",
        "failed_file": "",
        "commit_result": "",
        "push_approval": "",
        "push_result": "",
        "pr_url": "",
        "pr_result": "",
        # Only the (non-secret) installation id is stored in the checkpoint.
        # Tokens are minted when needed because they expire after ~1 hour.
        "github_installation_id": installation_id,
        "workspace_path": workspace_path,
        "agent_steps": 0,
        "debug_attempts": 0,
    }

def run_agent_task(
    task: str,
    thread_id: str,
    repo_url: str,
    installation_id: int = 0,
) -> dict:
    clone_token = None

    if installation_id:
        _, repo_name = _validate_github_url(repo_url)
        clone_token = github_app.installation_token(installation_id, repo_name)

    workspace_path = _prepare_repository(
        repo_url,
        thread_id,
        clone_token,
    )

    return agent.invoke(
        _initial_state(
            task,
            workspace_path,
            thread_id,
            installation_id,
        ),
        {
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )


def resume_agent_task(
    thread_id: str,
    decision: str,
) -> dict:
    workspace_path = _workspace_for_thread(thread_id)

    if not workspace_path.exists():
        raise ValueError(
            "Repository workspace no longer exists; "
            "the agent run cannot be resumed"
        )

    repo_dirs = [
        p
        for p in workspace_path.iterdir()
        if p.is_dir()
    ]

    if len(repo_dirs) != 1:
        raise ValueError(
            "Repository workspace is invalid or missing"
        )

    return agent.invoke(
        Command(resume=decision),
        {
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )


def _serialize_messages(messages) -> list:
    serialized = []

    for message in messages:
        content = message.content

        if isinstance(content, list):
            parts = []

            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    parts.append(
                        item.get("text", "")
                    )

                elif isinstance(item, str):
                    parts.append(item)

            content = "\n".join(parts)

        serialized.append(
            {
                "role": getattr(
                    message,
                    "type",
                    message.__class__.__name__,
                ),
                "content": content,
            }
        )

    return serialized


def _failure_message(text) -> str:
    """Extract the 'message' from a stringified tool result dict."""
    import ast

    try:
        data = ast.literal_eval(str(text))
        if isinstance(data, dict):
            return str(data.get("message", "")) or str(text)
    except (ValueError, SyntaxError):
        pass
    return str(text)


def serialize_result(result: dict) -> dict:
    interrupts = result.get("__interrupt__")

    if interrupts:
        return {
            "status": "waiting_for_approval",
            "approval_request": interrupts[0].value,
        }

    final_result = result.get(
        "final_result",
        "",
    )

    branch_result = result.get("branch_result", "")

    # A failed branch step used to end the run silently ("Could not complete"
    # with no reason). Report it as an error with the actual message.
    if isinstance(branch_result, str) and branch_result.startswith("error:"):
        return {
            "status": "error",
            "error": "Could not create the agent branch. " + branch_result[6:].strip(),
            "final_result": "",
            "test_result": "",
            "debug_result": "",
            "commit_result": "",
            "push_result": "",
            "pr_url": "",
            "branch_result": branch_result,
            "messages": _serialize_messages(result.get("messages", [])),
        }

    push_result = str(result.get("push_result", ""))
    pr_result = str(result.get("pr_result", ""))
    delivery_error = ""

    if "'status': 'error'" in push_result:
        # Committed locally, but the branch never reached GitHub.
        status = "push_failed"
        delivery_error = "Push failed: " + _failure_message(push_result)

    elif "'status': 'error'" in pr_result:
        status = "pr_failed"
        delivery_error = (
            "The branch was pushed, but the pull request could not be created: "
            + _failure_message(pr_result)
        )

    elif (
        isinstance(final_result, str)
        and final_result.startswith(
            "Human rejected the proposed change."
        )
    ):
        status = "rejected"

    elif (
        result.get("test_result")
        and "'status': 'passed'" in str(
            result.get("test_result")
        )
    ):
        status = "completed"

    elif (
        "'status': 'failed'" in str(result.get("test_result", ""))
        or "'status': 'error'" in str(result.get("test_result", ""))
    ):
        # The run ended without a passing test, so nothing was committed
        # or pushed. Do not report that as "completed".
        status = "tests_failed"

    elif final_result:
        status = "completed"

    else:
        status = "cannot_complete"

    return {
        "status": status,
        "final_result": final_result,
        "test_result": result.get(
            "test_result",
            "",
        ),
        "debug_result": result.get(
            "debug_result",
            "",
        ),
        "commit_result": result.get("commit_result", ""),
        "push_result": result.get("push_result", ""),
        "pr_url": result.get("pr_url", ""),
        "branch_result": branch_result,
        "pr_result": pr_result,
        "error": delivery_error,
        "messages": _serialize_messages(
            result.get(
                "messages",
                [],
            )
        ),
    }