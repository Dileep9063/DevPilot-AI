import logging

from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .ai_service import ask_gemini
from . import agent_service, github_app
from .models import AgentRun, GitHubInstallation

logger = logging.getLogger(__name__)
MAX_TASK_LENGTH = 4000
MAX_REPO_URL_LENGTH = 300


@api_view(["GET"])
@permission_classes([])
def status(request):
    return Response({"status": "success", "message": "DevPilot AI backend is running"})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ai_chat(request):
    message = request.data.get("message")
    if not isinstance(message, str) or not message.strip():
        return Response({"error": "message must be a non-empty string"}, status=400)
    if len(message) > MAX_TASK_LENGTH:
        return Response({"error": f"message must be at most {MAX_TASK_LENGTH} characters"}, status=400)
    try:
        return Response({"response": ask_gemini(message.strip())})
    except Exception:
        logger.exception("AI chat provider failed")
        return Response({"error": "The AI provider request failed. Please try again later."}, status=502)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def agent_run(request):
    task = request.data.get("task")
    repo_url = request.data.get("repo_url")
    if not isinstance(task, str) or not task.strip():
        return Response({"error": "task must be a non-empty string"}, status=400)
    if len(task) > MAX_TASK_LENGTH:
        return Response({"error": f"task must be at most {MAX_TASK_LENGTH} characters"}, status=400)
    if not isinstance(repo_url, str) or not repo_url.strip():
        return Response({"error": "repo_url must be a non-empty string"}, status=400)
    repo_url = repo_url.strip()
    if len(repo_url) > MAX_REPO_URL_LENGTH:
        return Response({"error": f"repo_url must be at most {MAX_REPO_URL_LENGTH} characters"}, status=400)
    try:
        # Validate before creating a database row or starting any Git operation.
        agent_service._validate_github_url(repo_url)
    except ValueError as exc:
        return Response({"error": str(exc)}, status=400)

    # If a thread_id is supplied, preserve the ownership/security contract
    # before rejecting thread reuse. A caller must not learn that another
    # user's run exists merely because this endpoint disallows thread reuse.
    supplied_thread_id = request.data.get("thread_id")
    if supplied_thread_id:
        try:
            agent_service.validate_owned_thread(request.user, supplied_thread_id)
        except agent_service.ThreadOwnershipError as exc:
            return Response({"error": str(exc)}, status=403)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=400)

        # Starting a run and resuming a checkpoint are separate operations.
        # Reusing an owned thread could overwrite its LangGraph checkpoint.
        return Response(
            {"error": "Do not pass thread_id to /api/agent/run/. Use /api/agent/resume/ to resume an existing run."},
            status=400,
        )

    thread_id = None
    try:
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

        from django.conf import settings

        if settings.DEVPILOT_ASYNC_RUNS:
            from .tasks import run_agent_job

            AgentRun.objects.filter(thread_id=thread_id).update(status="queued", updated_at=timezone.now())
            try:
                run_agent_job.delay(thread_id, task, repo_url, installation_id)
            except Exception:
                logger.exception("Could not enqueue agent run (thread_id=%s)", thread_id)
                agent_service.record_run_outcome(thread_id, error="The agent job could not be queued.")
                return Response({"error": "The agent job could not be queued. Please retry."}, status=503)
            return Response({"status": "queued", "thread_id": thread_id}, status=202)

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
    except Exception:
        logger.exception("Agent run failed (thread_id=%s)", thread_id or "unassigned")
        agent_service.record_run_outcome(thread_id, error="Agent run failed. Check server logs for details.")
        return Response({"error": "The agent run failed. Please check the run status and server logs."}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
    agent_service.record_run_outcome(thread_id, payload)
    if payload.get("status") == "completed" and payload.get("pr_url"):
        agent_service.cleanup_completed_workspace(thread_id)
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
        from django.conf import settings
        run_record = AgentRun.objects.get(thread_id=thread_id)
        installation_id = 0
        if github_app.is_configured():
            owner, repo_name = agent_service._validate_github_url(run_record.repo_url)
            installation_id = github_app.resolve_repo_access(request.user, owner, repo_name)

        if settings.DEVPILOT_ASYNC_RUNS:
            from .tasks import resume_agent_job

            claimed = AgentRun.objects.filter(
                thread_id=thread_id, status="waiting_for_approval"
            ).update(status="queued", updated_at=timezone.now())
            if not claimed:
                return Response(
                    {"error": "This run is not waiting for approval or has already been resumed."},
                    status=409,
                )
            try:
                resume_agent_job.delay(thread_id, decision, installation_id)
            except Exception:
                logger.exception("Could not enqueue agent resume (thread_id=%s)", thread_id)
                agent_service.record_run_outcome(thread_id, error="The resume job could not be queued.")
                return Response({"error": "The resume job could not be queued. Please retry."}, status=503)
            return Response({"status": "queued", "thread_id": thread_id}, status=202)

        result = agent_service.resume_agent_task(thread_id, decision, installation_id)
    except agent_service.ThreadOwnershipError as e:
        return Response({"error": str(e)}, status=403)
    except ValueError as e:
        return Response({"error": str(e)}, status=400)
    except Exception:
        logger.exception("Agent resume failed (thread_id=%s)", thread_id)
        agent_service.record_run_outcome(thread_id, error="Agent resume failed. Check server logs for details.")
        return Response({"error": "The agent could not resume this run. Please check the run status and server logs."}, status=500)

    payload = agent_service.serialize_result(result)
    payload["thread_id"] = thread_id
    agent_service.record_run_outcome(thread_id, payload)
    if payload.get("status") == "completed" and payload.get("pr_url"):
        agent_service.cleanup_completed_workspace(thread_id)
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
def github_repositories(request):
    """Repositories selected for the user's connected DevPilot App installations."""
    if not github_app.is_configured():
        return Response({"app_configured": False, "repositories": []})
    repositories = {}
    try:
        for installation in GitHubInstallation.objects.filter(user=request.user):
            for repo in github_app.installation_repositories(installation.installation_id):
                full_name = repo.get("full_name")
                if full_name:
                    repositories[full_name.lower()] = {
                        "full_name": full_name,
                        "html_url": repo.get("html_url", ""),
                        "clone_url": repo.get("clone_url", ""),
                        "default_branch": repo.get("default_branch", ""),
                        "private": bool(repo.get("private", False)),
                    }
    except Exception:
        logger.exception("Could not list GitHub repositories for user_id=%s", request.user.id)
        return Response({"error": "Could not load GitHub repositories. Check the App installation permissions and try again."}, status=502)
    return Response({
        "app_configured": True,
        "repositories": sorted(repositories.values(), key=lambda item: item["full_name"].lower()),
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
