from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ai_service import ask_gemini
from . import agent_service, github_app
from .models import AgentRun, GitHubInstallation


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

        installation_id = 0

        # App mode: the repository must belong to an installation this user
        # has connected. Without an App configured, GITHUB_TOKEN is used.
        if github_app.is_configured():
            owner, repo_name = agent_service._validate_github_url(repo_url)
            installation_id = github_app.resolve_repo_access(
                request.user, owner, repo_name
            )

        result = agent_service.run_agent_task(
            task, thread_id, repo_url, installation_id
        )
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=403)
    except github_app.GitHubAccessError as e:
        agent_service.record_run_outcome(thread_id, error=str(e))
        return Response({"error": str(e), "code": "github_access"}, status=403)
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


# ------------------------------------------------------------
# GitHub App connection
# ------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def github_status(request):
    """Is App mode on, and which GitHub accounts has this user connected?"""
    installs = GitHubInstallation.objects.filter(user=request.user)
    return Response({
        "app_configured": github_app.is_configured(),
        "connected": installs.exists(),
        "installations": [
            {
                "installation_id": i.installation_id,
                "account": i.account_login,
                "repository_selection": i.repository_selection,
            }
            for i in installs
        ],
    })


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def github_connect(request):
    """URL that sends the user to GitHub to install / configure the App."""
    if not github_app.is_configured():
        return Response({"error": "GitHub App is not configured on the server"}, status=400)

    state = github_app.make_state(request.user.id)
    return Response({"url": github_app.connect_url(state)})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def github_disconnect(request):
    """Forget the user's connections. (Uninstalling is done on GitHub.)"""
    deleted, _ = GitHubInstallation.objects.filter(user=request.user).delete()
    return Response({"disconnected": deleted})


def _frontend_redirect(**params):
    from urllib.parse import urlencode

    from django.conf import settings as dj_settings
    from django.http import HttpResponseRedirect

    return HttpResponseRedirect(f"{dj_settings.FRONTEND_URL}/?{urlencode(params)}")


def github_callback(request):
    """
    GitHub redirects the browser here after the App is installed.

    There is no JWT on a browser redirect, so the signed `state` identifies the
    DevPilot user. The installation is saved only after GitHub confirms, using
    the user's own authorization code, that this user can access it.
    """
    from django.contrib.auth import get_user_model

    if request.GET.get("error"):
        return _frontend_redirect(github="error", reason=request.GET.get("error_description", "GitHub authorization was cancelled"))

    user_id = github_app.read_state(request.GET.get("state", ""))
    code = request.GET.get("code")
    installation_id = request.GET.get("installation_id")

    if user_id is None:
        return _frontend_redirect(github="error", reason="The connection request expired. Please try again.")

    if not code or not installation_id or not installation_id.isdigit():
        return _frontend_redirect(github="error", reason="GitHub did not return the expected details.")

    try:
        user = get_user_model().objects.get(pk=user_id)
        user_token = github_app.exchange_code(code)
        accessible = github_app.user_installations(user_token)
    except github_app.GitHubAccessError as e:
        return _frontend_redirect(github="error", reason=str(e))
    except Exception:
        return _frontend_redirect(github="error", reason="Could not verify your GitHub account. Please try again.")

    match = next((i for i in accessible if str(i.get("id")) == installation_id), None)

    if match is None:
        return _frontend_redirect(github="error", reason="That installation does not belong to your GitHub account.")

    GitHubInstallation.objects.update_or_create(
        user=user,
        installation_id=int(installation_id),
        defaults={
            "account_login": (match.get("account") or {}).get("login", ""),
            "repository_selection": match.get("repository_selection", ""),
        },
    )

    return _frontend_redirect(github="connected")
