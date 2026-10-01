import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agent import agent
from core.models import AgentRun


class ThreadOwnershipError(Exception):
    pass


def new_thread_id() -> str:
    return str(uuid.uuid4())


def create_agent_run(user, thread_id: str, task: str) -> AgentRun:
    return AgentRun.objects.create(user=user, thread_id=uuid.UUID(str(thread_id)), task=task)


def validate_owned_thread(user, thread_id: str) -> str:
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


def _initial_state(task: str) -> dict:
    return {
        "messages": [HumanMessage(content=task)],
        "approval": "",
        "proposed_path": "",
        "proposed_content": "",
        "test_result": "",
        "debug_result": "",
        "final_result": "",
        "failed_file": "",
    }


def run_agent_task(task: str, thread_id: str) -> dict:
    return agent.invoke(_initial_state(task), {"configurable": {"thread_id": thread_id}})


def resume_agent_task(thread_id: str, decision: str) -> dict:
    return agent.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})


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
