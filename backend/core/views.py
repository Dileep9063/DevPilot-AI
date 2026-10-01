from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ai_service import ask_gemini
from . import agent_service


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
    if not task:
        return Response({"error": "task is required"}, status=400)

    supplied_thread_id = request.data.get("thread_id")
    try:
        if supplied_thread_id:
            thread_id = agent_service.validate_owned_thread(request.user, supplied_thread_id)
        else:
            thread_id = agent_service.new_thread_id()
            agent_service.create_agent_run(request.user, thread_id, task)

        result = agent_service.run_agent_task(task, thread_id)
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=403)
    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
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
        return Response({"error": str(e)}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
    return Response(payload)
