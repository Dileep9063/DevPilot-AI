from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase

from core.models import AgentRun
from core.tasks import run_agent_job


class BackgroundTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="worker-test", password="test-password"
        )
        self.run = AgentRun.objects.create(
            user=self.user,
            task="test task",
            repo_url="https://github.com/example/repo",
            status="queued",
        )

    def test_worker_claims_queued_run_and_records_result(self):
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

    def test_worker_skips_duplicate_delivery(self):
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
