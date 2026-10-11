"""Celery tasks for long-running DevPilot agent workflows."""
import logging

from celery import shared_task
from django.db import close_old_connections
from core.models import AgentRun

logger = logging.getLogger(__name__)


def _claim_queued_run(thread_id):
    # Atomic claim prevents duplicate delivery from running a job twice.
    return AgentRun.objects.filter(
        thread_id=thread_id, status="queued"
    ).update(status="running") == 1


@shared_task(ignore_result=True, acks_late=False)
def run_agent_job(thread_id, task, repo_url, installation_id=0):
    close_old_connections()
    try:
        if not _claim_queued_run(thread_id):
            logger.warning("Skipping duplicate or stale agent job: thread_id=%s", thread_id)
            return
        from core import agent_service
        result = agent_service.run_agent_task(task, thread_id, repo_url, int(installation_id or 0))
        agent_service.record_run_outcome(thread_id, agent_service.serialize_result(result))
    except Exception:
        logger.exception("Background agent run failed: thread_id=%s", thread_id)
        from core import agent_service
        agent_service.record_run_outcome(thread_id, error="The agent run failed. Check server logs for details.")
    finally:
        close_old_connections()


@shared_task(ignore_result=True, acks_late=False)
def resume_agent_job(thread_id, decision):
    close_old_connections()
    try:
        if not _claim_queued_run(thread_id):
            logger.warning("Skipping duplicate or stale resume job: thread_id=%s", thread_id)
            return
        from core import agent_service
        result = agent_service.resume_agent_task(thread_id, decision)
        agent_service.record_run_outcome(thread_id, agent_service.serialize_result(result))
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
