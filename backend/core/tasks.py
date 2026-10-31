"""Celery tasks for long-running DevPilot agent workflows."""
import logging
import threading
from contextlib import contextmanager

from celery import shared_task
from django.db import close_old_connections
from django.utils import timezone
from core.models import AgentRun

logger = logging.getLogger(__name__)


def _claim_queued_run(thread_id):
    # Atomic claim prevents duplicate delivery from running a job twice.
    return AgentRun.objects.filter(
        thread_id=thread_id, status="queued"
    ).update(status="running", updated_at=timezone.now()) == 1


@contextmanager
def _run_heartbeat(thread_id, interval_seconds=60):
    """Refresh active jobs so recovery only marks genuinely abandoned work."""
    stop = threading.Event()

    def beat():
        while not stop.wait(interval_seconds):
            close_old_connections()
            try:
                AgentRun.objects.filter(
                    thread_id=thread_id, status="running"
                ).update(updated_at=timezone.now())
            except Exception:
                logger.exception("Could not refresh run heartbeat: thread_id=%s", thread_id)
            finally:
                close_old_connections()

    thread = threading.Thread(
        target=beat, name=f"devpilot-heartbeat-{thread_id}", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


@shared_task(ignore_result=True, acks_late=False)
def run_agent_job(thread_id, task, repo_url, installation_id=0):
    close_old_connections()
    try:
        if not _claim_queued_run(thread_id):
            logger.warning("Skipping duplicate or stale agent job: thread_id=%s", thread_id)
            return
        from core import agent_service
        with _run_heartbeat(thread_id):
            result = agent_service.run_agent_task(task, thread_id, repo_url, int(installation_id or 0))
        payload = agent_service.serialize_result(result)
        agent_service.record_run_outcome(thread_id, payload)
        if payload.get("status") == "completed" and payload.get("pr_url"):
            agent_service.cleanup_completed_workspace(thread_id)
    except Exception:
        logger.exception("Background agent run failed: thread_id=%s", thread_id)
        from core import agent_service
        agent_service.record_run_outcome(thread_id, error="The agent run failed. Check server logs for details.")
    finally:
        close_old_connections()


@shared_task(ignore_result=True, acks_late=False)
def resume_agent_job(thread_id, decision, installation_id=0):
    close_old_connections()
    try:
        if not _claim_queued_run(thread_id):
            logger.warning("Skipping duplicate or stale resume job: thread_id=%s", thread_id)
            return
        from core import agent_service
        with _run_heartbeat(thread_id):
            result = agent_service.resume_agent_task(thread_id, decision, int(installation_id or 0))
        payload = agent_service.serialize_result(result)
        agent_service.record_run_outcome(thread_id, payload)
        if payload.get("status") == "completed" and payload.get("pr_url"):
            agent_service.cleanup_completed_workspace(thread_id)
    except Exception:
        logger.exception("Background agent resume failed: thread_id=%s", thread_id)
        from core import agent_service
        agent_service.record_run_outcome(thread_id, error="The agent could not resume this run. Check server logs for details.")
    finally:
        close_old_connections()


@shared_task(ignore_result=True, acks_late=False)
def recover_stale_agent_runs():
    """Mark queued/running jobs abandoned by a crashed worker as interrupted."""
    from datetime import timedelta
    from django.utils import timezone

    close_old_connections()
    try:
        cutoff = timezone.now() - timedelta(minutes=60)
        stale_runs = AgentRun.objects.filter(
            status__in=["queued", "running"],
            updated_at__lt=cutoff,
        )
        count = stale_runs.update(
            status="error",
            updated_at=timezone.now(),
            summary=(
                "This run was interrupted because its worker did not finish "
                "within the recovery window. Please start a new run."
            ),
        )
        if count:
            logger.warning("Marked %s stale DevPilot runs as interrupted", count)
        return count
    finally:
        close_old_connections()
