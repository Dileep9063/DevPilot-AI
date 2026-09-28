"""
Thin service layer between the Django views and the LangGraph agent
defined in core/agent.py.

This module is responsible for:
- Starting a new agent run for a task (POST /api/agent/run/)
- Resuming a run that is paused on a human-in-the-loop approval
  (POST /api/agent/resume/) — this covers BOTH approval points in the
  graph: approving a normal proposed change, and approving an AI-
  generated fix after a failed test.
- Converting LangGraph/LangChain objects into plain JSON-safe dicts.

Each conversation with the agent is identified by a `thread_id`. The
graph's checkpointer (InMemorySaver, see core/agent.py) uses this to
remember where a run is paused so it can be resumed later.
"""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from core.agent import agent


def new_thread_id() -> str:
    return str(uuid.uuid4())


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
    """Start (or restart) an agent run for `task` on `thread_id`."""

    config = {"configurable": {"thread_id": thread_id}}

    return agent.invoke(_initial_state(task), config)


def resume_agent_task(thread_id: str, decision: str) -> dict:
    """
    Resume a run paused at a human-approval interrupt.

    `decision` must be "approve" or "reject". This same function
    handles both HITL checkpoints in the graph (approving a normal
    proposed change, and approving an AI-generated debug fix) because
    they share the same `approval` node.
    """

    config = {"configurable": {"thread_id": thread_id}}

    return agent.invoke(Command(resume=decision), config)


def _serialize_messages(messages) -> list:
    serialized = []

    for message in messages:
        content = message.content

        # Some LangChain messages return content as a list of blocks
        # (e.g. Gemini text blocks) instead of a plain string.
        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)

            content = "\n".join(text_parts)

        serialized.append({
            "role": getattr(message, "type", message.__class__.__name__),
            "content": content,
        })

    return serialized


def serialize_result(result: dict) -> dict:
    """
    Convert a raw LangGraph result dict into a JSON-safe response.

    If the graph is paused waiting on human approval, `__interrupt__`
    will be present in the result — this is surfaced as
    status="waiting_for_approval" along with the details the frontend
    needs to render an approve/reject prompt.
    """

    interrupts = result.get("__interrupt__")

    if interrupts:
        return {
            "status": "waiting_for_approval",
            "approval_request": interrupts[0].value,
        }

    return {
        "status": "completed",
        "final_result": result.get("final_result", ""),
        "test_result": result.get("test_result", ""),
        "debug_result": result.get("debug_result", ""),
        "messages": _serialize_messages(result.get("messages", [])),
    }
