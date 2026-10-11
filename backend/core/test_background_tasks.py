from datetime import timedelta
from unittest import mock
import uuid

from django.utils import timezone

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import AgentRun
from core.tasks import recover_stale_agent_runs, run_agent_job


class BackgroundTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="worker-test", password="test-password"
        )
        self.run = AgentRun.objects.create(
            user=self.user,
            thread_id=uuid.uuid4(),
            task="test task",
            repo_url="https://github.com/example/repo",
            status="queued",
        )

    @mock.patch("core.tasks.close_old_connections")
    def test_worker_claims_queued_run_and_records_result(self, _close_connections):
        with mock.patch(
            "core.agent_service.run_agent_task",
            return_value={"final_result": "ok"},
        ) as execute, mock.patch(
            "core.agent_service.serialize_result",
            return_value={"status": "completed", "final_result": "ok"},
        ), mock.patch("core.agent_service.record_run_outcome") as record:
            run_agent_job.run(
                str(self.run.thread_id),
                "test task",
                "https://github.com/example/repo",
                0,
            )

        execute.assert_called_once()
        record.assert_called_once_with(
            str(self.run.thread_id),
            {"status": "completed", "final_result": "ok"},
        )
        self.run.refresh_from_db()
        # The mocked persistence layer does not update status; claiming the job does.
        self.assertEqual(self.run.status, "running")

    @mock.patch("core.tasks.close_old_connections")
    def test_worker_skips_duplicate_delivery(self, _close_connections):
        self.run.status = "running"
        self.run.save(update_fields=["status"])

        with mock.patch("core.agent_service.run_agent_task") as execute:
            run_agent_job.run(
                str(self.run.thread_id),
                "test task",
                "https://github.com/example/repo",
                0,
            )

        execute.assert_not_called()

    @mock.patch("core.tasks.close_old_connections")
    def test_stale_run_recovery_marks_abandoned_runs_as_error(self, _close_connections):
        stale_time = timezone.now() - timedelta(hours=2)
        AgentRun.objects.filter(pk=self.run.pk).update(updated_at=stale_time)
        recent = AgentRun.objects.create(
            user=self.user,
            thread_id=uuid.uuid4(),
            task="recent task",
            repo_url="https://github.com/example/repo",
            status="queued",
        )

        recovered = recover_stale_agent_runs.run()

        self.assertEqual(recovered, 1)
        self.run.refresh_from_db()
        recent.refresh_from_db()
        self.assertEqual(self.run.status, "error")
        self.assertIn("interrupted", self.run.summary.lower())
        self.assertEqual(recent.status, "queued")
