from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ai_service import ask_gemini
from . import agent_service
from .models import AgentRun


@api_view(["GET"])
@permission_classes([])
def status(request):
    return Response({"status": "success", "message": "DevPilot AI backend is running"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ai_chat(request):
    message = request.data.get("message")
    if not message:
        return Response({"error": "Message is required"}, status=400)
    try:
        return Response({"response": ask_gemini(message)})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_run(request):
    task = request.data.get("task")
    repo_url = request.data.get("repo_url")
    if not task:
        return Response({"error": "task is required"}, status=400)
    if not repo_url:
        return Response({"error": "repo_url is required"}, status=400)

    supplied_thread_id = request.data.get("thread_id")
    thread_id = None
    try:
        if supplied_thread_id:
            thread_id = agent_service.validate_owned_thread(request.user, supplied_thread_id)
        else:
            thread_id = agent_service.new_thread_id()
            agent_service.create_agent_run(request.user, thread_id, task, repo_url)

        result = agent_service.run_agent_task(task, thread_id, repo_url)
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=403)
    except ValueError as e:
        agent_service.record_run_outcome(thread_id, error=str(e))
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        agent_service.record_run_outcome(thread_id, error=str(e))
        return Response({"error": str(e)}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
    agent_service.record_run_outcome(thread_id, payload)
    return Response(payload)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_resume(request):
    thread_id = request.data.get("thread_id")
    decision = request.data.get("decision")

    if not thread_id:
        return Response({"error": "thread_id is required"}, status=400)
    if decision not in ("approve", "reject"):
        return Response({"error": "decision must be 'approve' or 'reject'"}, status=400)

    try:
        agent_service.validate_owned_thread(request.user, thread_id)
        result = agent_service.resume_agent_task(thread_id, decision)
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=403)
    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        agent_service.record_run_outcome(thread_id, error=str(e))
        return Response({"error": str(e)}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
    agent_service.record_run_outcome(thread_id, payload)
    return Response(payload)


def _run_summary(run):
    return {
        "thread_id": str(run.thread_id),
        "task": run.task,
        "repo_url": run.repo_url,
        "status": run.status,
        "pr_url": run.pr_url,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def agent_runs(request):
    """The current user's most recent runs (newest first)."""
    runs = AgentRun.objects.filter(user=request.user)[:50]
    return Response({"runs": [_run_summary(r) for r in runs]})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def agent_run_detail(request, thread_id):
    """One run, plus its pending approval request if it is paused."""
    try:
        agent_service.validate_owned_thread(request.user, thread_id)
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=404)
    except ValueError as e:
        return Response({"error": str(e)}, status=400)

    run = AgentRun.objects.get(thread_id=thread_id)
    data = _run_summary(run)
    data["summary"] = run.summary

    if run.status == "waiting_for_approval":
        data["approval_request"] = agent_service.get_pending_approval(thread_id)

    return Response(data)
