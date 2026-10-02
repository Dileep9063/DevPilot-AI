import re
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agent import agent


WORKSPACE_ROOT = Path("workspace").resolve()


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

    match = re.fullmatch(r"/([^/]+)/([^/]+?)(?:\.git)?/?", parsed.path)
    if not match:
        raise ValueError("repo_url must look like https://github.com/owner/repository")

    owner, repo_name = match.groups()
    return owner, repo_name


def _prepare_repository(repo_url: str, thread_id: str) -> str:
    owner, repo_name = _validate_github_url(repo_url)
    run_workspace = _workspace_for_thread(thread_id)
    repo_workspace = run_workspace / repo_name

    run_workspace.mkdir(parents=True, exist_ok=True)

    if repo_workspace.exists():
        if not (repo_workspace / ".git").exists():
            raise ValueError("Workspace already exists but is not a Git repository")
        return str(repo_workspace)

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_workspace)],
            cwd=run_workspace,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Unable to clone GitHub repository: {exc.stderr.strip() or exc.stdout.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("GitHub repository clone timed out after 120 seconds") from exc

    return str(repo_workspace)


def create_agent_run(user, thread_id: str, task: str):
    from core.models import AgentRun
    return AgentRun.objects.create(user=user, thread_id=uuid.UUID(str(thread_id)), task=task)


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
        raise ThreadOwnershipError("You do not have access to this agent run")
    return str(run.thread_id)


def _initial_state(task: str, workspace_path: str) -> dict:
    return {
        "messages": [HumanMessage(content=task)],
        "approval": "",
        "proposed_path": "",
        "proposed_content": "",
        "test_result": "",
        "debug_result": "",
        "final_result": "",
        "failed_file": "",
        "workspace_path": workspace_path,
    }


def run_agent_task(task: str, thread_id: str, repo_url: str) -> dict:
    workspace_path = _prepare_repository(repo_url, thread_id)
    return agent.invoke(
        _initial_state(task, workspace_path),
        {"configurable": {"thread_id": thread_id}},
    )


def resume_agent_task(thread_id: str, decision: str) -> dict:
    workspace_path = _workspace_for_thread(thread_id)
    if not workspace_path.exists():
        raise ValueError("Repository workspace no longer exists; the agent run cannot be resumed")

    repo_dirs = [p for p in workspace_path.iterdir() if p.is_dir()]
    if len(repo_dirs) != 1:
        raise ValueError("Repository workspace is invalid or missing")

    return agent.invoke(
        Command(resume=decision),
        {"configurable": {"thread_id": thread_id}},
    )


def _serialize_messages(messages) -> list:
    serialized = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts)
        serialized.append({"role": getattr(message, "type", message.__class__.__name__), "content": content})
    return serialized


def serialize_result(result: dict) -> dict:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return {"status": "waiting_for_approval", "approval_request": interrupts[0].value}
    return {
        "status": "completed",
        "final_result": result.get("final_result", ""),
        "test_result": result.get("test_result", ""),
        "debug_result": result.get("debug_result", ""),
        "messages": _serialize_messages(result.get("messages", [])),
    }
