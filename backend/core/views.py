from rest_framework.decorators import api_view
from rest_framework.response import Response

from .ai_service import ask_gemini
from . import agent_service


@api_view(["GET"])
def status(request):
    return Response({
        "status": "success",
        "message": "DevPilot AI backend is running"
    })


@api_view(["POST"])
def ai_chat(request):
    message = request.data.get("message")

    if not message:
        return Response(
            {"error": "Message is required"},
            status=400
        )

    try:
        answer = ask_gemini(message)

        return Response({
            "response": answer
        })

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=500
        )


@api_view(["POST"])
def agent_run(request):
    """
    Start a new DevPilot AI agent run.

    Body: { "task": "<what you want the agent to do>" }

    Returns either:
      - status="completed" with the final result, or
      - status="waiting_for_approval" with an `approval_request` the
        caller must show to a human, then send back to
        /api/agent/resume/ along with the returned `thread_id`.
    """

    task = request.data.get("task")

    if not task:
        return Response(
            {"error": "task is required"},
            status=400
        )

    thread_id = request.data.get("thread_id") or agent_service.new_thread_id()

    try:
        result = agent_service.run_agent_task(task, thread_id)

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=500
        )

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id

    return Response(payload)


@api_view(["POST"])
def agent_resume(request):
    """
    Resume an agent run that is paused on a human-in-the-loop approval.

    Body: { "thread_id": "<id returned by /api/agent/run/>",
            "decision": "approve" | "reject" }

    Used for BOTH approval checkpoints in the graph: approving a
    normal proposed file change, and approving an AI-generated fix
    after a failed test.
    """

    thread_id = request.data.get("thread_id")
    decision = request.data.get("decision")

    if not thread_id:
        return Response(
            {"error": "thread_id is required"},
            status=400
        )

    if decision not in ("approve", "reject"):
        return Response(
            {"error": "decision must be 'approve' or 'reject'"},
            status=400
        )

    try:
        result = agent_service.resume_agent_task(thread_id, decision)

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=500
        )

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id

    return Response(payload)
