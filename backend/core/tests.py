"""
Automated tests for DevPilot AI.

These replace the one-off manual scripts that used to live scattered
across backend/ and backend/core/ (test_ai_debug.py, hitl_test.py,
test_debug_pipeline.py, etc). Those scripts were useful during
development but called Gemini for real, printed to stdout, and had to be
run and read by hand.

Here we test the same logic in a repeatable, offline way:
  - The deterministic graph nodes/routing (no LLM calls involved).
  - The sandboxed file tools.
  - The API endpoints' request validation.

Anything that requires a real Gemini call (the LLM-driven nodes, or a
full agent_run against a live model) is intentionally left for manual
end-to-end testing with a real GEMINI_API_KEY — see the project README
for that checklist.
"""

import uuid

from django.contrib.auth.models import User
from django.test import TestCase

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from rest_framework.test import APIClient

from .models import AgentRun

from core.agent import (
    extract_proposal_node,
    debug_node,
    route_after_test,
    route_after_debug_fix,
    write_node,
    route_after_commit,
    route_after_push_approval,
    route_after_push,
)

from core.tools import (
    WORKSPACE,
    write_file,
    read_file,
    propose_file_change,
)


class AgentGraphUnitTests(TestCase):
    """Deterministic node/routing logic — no Gemini calls involved."""

    def test_extract_proposal_parses_pending_approval(self):
        message = ToolMessage(
            content=str({
                "status": "pending_approval",
                "path": "auth.py",
                "proposed_content": "def login():\n    pass",
            }),
            tool_call_id="1",
        )

        result = extract_proposal_node({"messages": [message]})

        self.assertEqual(result["proposed_path"], "auth.py")
        self.assertIn(
            "def login",
            result["proposed_content"],
        )

    def test_extract_proposal_ignores_non_dict_content(self):
        message = ToolMessage(
            content="not a dict",
            thread_id=uuid.uuid4(),
            tool_call_id="1",
        )

        result = extract_proposal_node({"messages": [message]})

        # Non-dict tool output is not a proposal: nothing is queued for approval.
        self.assertEqual(result["proposed_path"], "")
        self.assertEqual(result["proposed_content"], "")

    def test_debug_node_detects_failure(self):
        state = {
            "test_result": "FAILED test_auth.py::test_login - assert False"
        }

        result = debug_node(state)

        self.assertEqual(
            result["failed_file"],
            "test_auth.py",
        )

        self.assertIn(
            "failure_detected",
            result["debug_result"],
        )

    def test_debug_node_no_failure(self):
        result = debug_node({
            "test_result": "1 passed in 0.05s"
        })

        self.assertEqual(
            result["failed_file"],
            "",
        )

        self.assertEqual(
            result["debug_result"],
            "No test failure detected.",
        )

    def test_route_after_test(self):
        self.assertEqual(
            route_after_test({
                "test_result": "{'status': 'passed'}"
            }),
            "commit",
        )

        self.assertEqual(
            route_after_test({
                "test_result": "{'status': 'failed'}"
            }),
            "debug",
        )

        # No tests found must not block delivery; a real error must.
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'no_tests'}"}),
            "commit",
        )
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'error'}"}),
            "debug",
        )

    def test_route_after_debug_fix_with_proposal(self):
        state = {
            "proposed_path": "auth.py",
            "proposed_content": "code",
        }

        self.assertEqual(
            route_after_debug_fix(state),
            "approval",
        )

    def test_route_after_debug_fix_without_proposal(self):
        state = {
            "proposed_path": "",
            "proposed_content": "",
        }

        self.assertEqual(
            route_after_debug_fix(state),
            END,
        )

    def test_write_node_resets_approval_for_next_hitl_checkpoint(self):
        """
        Regression test: once a human approves a change, that decision
        must not silently carry over and auto-approve a LATER approval
        checkpoint in the same run (e.g. approving an AI-generated fix
        after a failed test). write_node must clear "approval" so the
        next visit to approval_node asks a human again.
        """

        test_path = "unit_test_write_reset.py"

        try:
            state = {
                "proposed_path": test_path,
                "proposed_content": "print('ok')",
                "approval": "approve",
            }

            result = write_node(state)

            self.assertEqual(
                result["approval"],
                "",
            )

        finally:
            (WORKSPACE / test_path).unlink(
                missing_ok=True
            )


class ToolsTests(TestCase):
    """The sandboxed file tools the agent is allowed to call."""

    def test_write_then_read_file_roundtrip(self):
        test_path = "unit_test_roundtrip.py"

        try:
            write_result = write_file.invoke({
                "path": test_path,
                "content": "print('hi')",
            })

            self.assertEqual(
                write_result["status"],
                "success",
            )

            read_result = read_file.invoke({
                "path": test_path,
            })

            self.assertEqual(
                read_result["content"],
                "print('hi')",
            )

        finally:
            (WORKSPACE / test_path).unlink(
                missing_ok=True
            )

    def test_propose_file_change_blocks_path_traversal(self):
        result = propose_file_change.invoke({
            "path": "../outside_workspace.py",
            "old_text": "a",
            "new_text": "b",
        })

        self.assertEqual(
            result["status"],
            "error",
        )

    def test_propose_file_change_returns_pending_approval(self):
        old_text = (WORKSPACE / "auth.py").read_text(encoding="utf-8").splitlines()[0]

        result = propose_file_change.invoke({
            "path": "auth.py",
            "old_text": old_text,
            "new_text": old_text + "  # reviewed",
        })

        self.assertEqual(
            result["status"],
            "pending_approval",
        )

        self.assertEqual(
            result["path"],
            "auth.py",
        )


class AgentApiValidationTests(TestCase):
    """Request validation for the agent endpoints (no Gemini calls)."""

    def setUp(self):
        self.client = APIClient()

        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="Testuser@123",
        )

        self.client.force_authenticate(
            user=self.user
        )

    def test_status_endpoint(self):
        response = self.client.get(
            "/api/status/"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.data["status"],
            "success",
        )

    def test_agent_run_requires_task(self):
        response = self.client.post(
            "/api/agent/run/",
            {},
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_agent_resume_requires_thread_id(self):
        response = self.client.post(
            "/api/agent/resume/",
            {"decision": "approve"},
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_agent_resume_requires_valid_decision(self):
        response = self.client.post(
            "/api/agent/resume/",
            {
                "thread_id": "abc",
                "decision": "maybe",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            400,
        )
    def test_agent_run_requires_authentication(self):
        self.client.force_authenticate(user=None)

        response = self.client.post(
            "/api/agent/run/",
            {"task": "test"},
            format="json",
        )

        self.assertEqual(response.status_code, 401)


    def test_agent_resume_requires_authentication(self):
        self.client.force_authenticate(user=None)

        response = self.client.post(
            "/api/agent/resume/",
            {
                "thread_id": str(uuid.uuid4()),
                "decision": "approve",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)

class AgentThreadOwnershipTests(TestCase):
    """Authenticated users can only access their own agent threads."""

    def setUp(self):
        self.client = APIClient()

        self.user_a = User.objects.create_user(
            username="user_a",
            email="usera@example.com",
            password="Testuser@123",
        )

        self.user_b = User.objects.create_user(
            username="user_b",
            email="userb@example.com",
            password="Testuser@123",
        )

        self.agent_run = AgentRun.objects.create(
            user=self.user_a,
            thread_id=uuid.uuid4(),
            task="Test agent task",
        )

    def test_user_cannot_resume_another_users_thread(self):
        self.client.force_authenticate(
            user=self.user_b
        )

        response = self.client.post(
            "/api/agent/resume/",
            {
                "thread_id": str(self.agent_run.thread_id),
                "decision": "approve",
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            403,
        )

    def test_user_cannot_run_using_another_users_thread(self):
        self.client.force_authenticate(
            user=self.user_b
        )

        response = self.client.post(
            "/api/agent/run/",
            {
                "task": "Try to access another user's thread",
                "repo_url": "https://github.com/owner/repo",
                "thread_id": str(self.agent_run.thread_id),
            },
            format="json",
        )

        self.assertEqual(
            response.status_code,
            403,
        )
class JWTAuthenticationTests(TestCase):
    """Test JWT login and token refresh behavior."""

    def setUp(self):
        self.client = APIClient()

        self.user = User.objects.create_user(
            username="jwt_user",
            email="jwt@example.com",
            password="Testuser@123",
        )

    def test_login_returns_access_and_refresh_tokens(self):
        response = self.client.post(
            "/api/auth/login/",
            {
                "username": "jwt_user",
                "password": "Testuser@123",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)
        self.assertIn("refresh", response.data)

    def test_refresh_returns_new_access_token(self):
        login_response = self.client.post(
            "/api/auth/login/",
            {
                "username": "jwt_user",
                "password": "Testuser@123",
            },
            format="json",
        )

        refresh_token = login_response.data["refresh"]

        response = self.client.post(
            "/api/auth/refresh/",
            {
                "refresh": refresh_token,
            },
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)
    def test_login_rejects_invalid_password(self):
        response = self.client.post(
            "/api/auth/login/",
            {
                "username": "jwt_user",
                "password": "WrongPassword123",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 401)

class ProviderFallbackTests(TestCase):
    def test_transient_provider_error_detection(self):
        from core.agent import is_transient_provider_error

        self.assertTrue(is_transient_provider_error(Exception("503 UNAVAILABLE")))
        self.assertTrue(is_transient_provider_error(Exception("429 rate limit")))
        self.assertFalse(is_transient_provider_error(Exception("invalid request")))


class DeliveryRoutingTests(TestCase):
    """Commit -> push approval -> push -> pull request routing."""

    def test_commit_success_asks_for_push_approval(self):
        self.assertEqual(
            route_after_commit({"commit_result": "{'status': 'success'}"}),
            "push_approval",
        )

    def test_nothing_to_commit_or_error_ends_run(self):
        self.assertEqual(
            route_after_commit({"commit_result": "{'status': 'nothing_to_commit'}"}),
            END,
        )
        self.assertEqual(
            route_after_commit({"commit_result": "{'status': 'error'}"}),
            END,
        )

    def test_push_only_after_human_approval(self):
        self.assertEqual(route_after_push_approval({"push_approval": "approve"}), "push")
        self.assertEqual(route_after_push_approval({"push_approval": "reject"}), END)
        self.assertEqual(route_after_push_approval({}), END)

    def test_pull_request_only_after_successful_push(self):
        self.assertEqual(
            route_after_push({"push_result": "{'status': 'success'}"}),
            "pull_request",
        )
        self.assertEqual(route_after_push({"push_result": "{'status': 'error'}"}), END)


class RunHistoryApiTests(TestCase):
    """Run history list/detail endpoints and outcome recording."""

    def setUp(self):
        from core import agent_service

        self.agent_service = agent_service
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="hist_a", email="a@example.com", password="Testuser@123"
        )
        self.other = User.objects.create_user(
            username="hist_b", email="b@example.com", password="Testuser@123"
        )
        self.run = AgentRun.objects.create(
            user=self.user,
            thread_id=uuid.uuid4(),
            task="Add logout",
            repo_url="https://github.com/o/r",
        )

    def test_list_requires_authentication(self):
        self.assertEqual(self.client.get("/api/agent/runs/").status_code, 401)

    def test_list_returns_only_own_runs_newest_first(self):
        AgentRun.objects.create(user=self.other, thread_id=uuid.uuid4(), task="not mine")
        newer = AgentRun.objects.create(user=self.user, thread_id=uuid.uuid4(), task="newer")

        self.client.force_authenticate(user=self.user)
        runs = self.client.get("/api/agent/runs/").json()["runs"]

        self.assertEqual([r["task"] for r in runs], ["newer", "Add logout"])
        self.assertEqual(runs[0]["thread_id"], str(newer.thread_id))

    def test_detail_hides_other_users_runs(self):
        self.client.force_authenticate(user=self.other)
        res = self.client.get(f"/api/agent/runs/{self.run.thread_id}/")
        self.assertEqual(res.status_code, 404)

    def test_record_outcome_saves_status_pr_and_summary(self):
        self.agent_service.record_run_outcome(
            self.run.thread_id,
            {"status": "completed", "pr_url": "https://github.com/o/r/pull/1",
             "final_result": "done"},
        )
        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "completed")
        self.assertEqual(self.run.pr_url, "https://github.com/o/r/pull/1")
        self.assertEqual(self.run.summary, "done")

    def test_record_error_marks_run_as_error(self):
        self.agent_service.record_run_outcome(self.run.thread_id, error="boom")
        self.run.refresh_from_db()
        self.assertEqual((self.run.status, self.run.summary), ("error", "boom"))

    def test_detail_includes_pending_approval_when_paused(self):
        from unittest import mock

        self.run.status = "waiting_for_approval"
        self.run.save()
        self.client.force_authenticate(user=self.user)

        with mock.patch.object(
            self.agent_service, "get_pending_approval",
            return_value={"path": "a.py", "content": "x"},
        ):
            data = self.client.get(f"/api/agent/runs/{self.run.thread_id}/").json()

        self.assertEqual(data["approval_request"]["path"], "a.py")


class FullGraphDeliveryTests(TestCase):
    """
    Drives the REAL compiled graph (no LLM) through:
    approve change -> branch -> write -> test -> commit -> approve push
    -> push -> pull request, against a local bare "GitHub" remote.

    Needs git, pytest and the Postgres checkpointer (same as the app).
    """

    def setUp(self):
        import shutil
        import subprocess
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="devpilot_e2e_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.remote = f"{self.tmp}/remote.git"
        self.work = f"{self.tmp}/work"

        def sh(*args, cwd):
            subprocess.run(args, cwd=cwd, check=True, capture_output=True)

        self.sh = sh
        sh("git", "init", "--bare", "remote.git", cwd=self.tmp)
        sh("git", "clone", "remote.git", "work", cwd=self.tmp)
        with open(f"{self.work}/calc.py", "w") as f:
            f.write("def add(a, b):\n    return a + b\n")
        with open(f"{self.work}/test_calc.py", "w") as f:
            f.write("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
        sh("git", "add", "-A", cwd=self.work)
        sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init", cwd=self.work)
        sh("git", "branch", "-M", "main", cwd=self.work)
        sh("git", "push", "origin", "main", cwd=self.work)
        # Looks like GitHub for PR purposes; pushes go to the local bare repo.
        sh("git", "remote", "set-url", "origin", "https://github.com/test/repo.git", cwd=self.work)
        sh("git", "config", "remote.origin.pushurl", self.remote, cwd=self.work)

    def _start(self, path="calc.py", content=None, proposal_type="existing_file"):
        from core import agent_service
        from core.agent import agent

        if content is None:
            content = 'def add(a, b):\n    """Return a + b."""\n    return a + b\n'

        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        agent.update_state(
            cfg, agent_service._initial_state("Document add()", self.work, thread_id),
            as_node="__start__",
        )
        agent.update_state(cfg, {
            "proposed_path": path,
            "proposed_content": content,
            "proposal_type": proposal_type,
        }, as_node="extract_proposal")
        return agent, cfg, thread_id

    def _remote_branches(self):
        import subprocess

        out = subprocess.run(
            ["git", "--git-dir", self.remote, "branch"], capture_output=True, text=True
        ).stdout
        return out.split()

    def test_full_flow_to_pull_request(self):
        import os
        from unittest import mock
        from langgraph.types import Command
        from core import tools, agent_service

        agent, cfg, thread_id = self._start()

        first = agent.invoke(None, cfg)["__interrupt__"][0].value
        self.assertEqual(first["type"], "code_change")
        self.assertIn("return a + b", first["original_content"])
        self.assertEqual(agent_service.get_pending_approval(thread_id), first)

        second = agent.invoke(Command(resume="approve"), cfg)
        push_request = second["__interrupt__"][0].value
        self.assertEqual(push_request["type"], "push_approval")
        self.assertIn("'status': 'passed'", second["test_result"])
        self.assertEqual(self._remote_branches(), ["main"], "nothing pushed before approval")

        class Resp:
            def __init__(self, code, data):
                self.status_code, self._d, self.text = code, data, ""

            def json(self):
                return self._d

            def raise_for_status(self):
                pass

        fake = mock.MagicMock()
        fake.RequestException = Exception
        fake.get.return_value = Resp(200, {"default_branch": "main"})
        fake.post.return_value = Resp(201, {"html_url": "https://github.com/test/repo/pull/7", "number": 7})

        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "fake-token"}), \
                mock.patch.object(tools, "requests", fake):
            final = agent.invoke(Command(resume="approve"), cfg)

        payload = agent_service.serialize_result(final)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["pr_url"], "https://github.com/test/repo/pull/7")

        branch = f"devpilot/{thread_id}"
        self.assertIn(branch, self._remote_branches())
        self.assertEqual(fake.post.call_args.kwargs["json"]["head"], branch)
        self.assertEqual(fake.post.call_args.kwargs["json"]["base"], "main")

        with open(f"{self.work}/.git/config") as f:
            self.assertNotIn("fake-token", f.read(), "token must never be persisted")

    def test_rejecting_push_keeps_changes_local(self):
        from langgraph.types import Command

        agent, cfg, thread_id = self._start()
        agent.invoke(None, cfg)
        agent.invoke(Command(resume="approve"), cfg)
        final = agent.invoke(Command(resume="reject"), cfg)

        self.assertEqual(final.get("push_result", ""), "")
        self.assertEqual(final.get("pr_url", ""), "")
        self.assertEqual(self._remote_branches(), ["main"])

    def test_repo_without_tests_still_reaches_push_approval(self):
        """
        Regression: a repo with no tests (e.g. just created) used to end the
        run silently after writing the file locally, so nothing was pushed.
        """
        from langgraph.types import Command

        # Replace the fixture project with a repo that only has a README.
        self.sh("git", "rm", "-q", "calc.py", "test_calc.py", cwd=self.work)
        with open(f"{self.work}/README.md", "w") as f:
            f.write("# demo\n")
        self.sh("git", "add", "-A", cwd=self.work)
        self.sh("git", "-c", "user.name=t", "-c", "user.email=t@t",
                "commit", "-m", "readme only", cwd=self.work)

        agent, cfg, thread_id = self._start(
            path="hello.py", content='print("hello")\n', proposal_type="new_file"
        )

        first = agent.invoke(None, cfg)["__interrupt__"][0].value
        self.assertEqual(first["type"], "code_change")

        second = agent.invoke(Command(resume="approve"), cfg)
        push_request = second["__interrupt__"][0].value

        self.assertEqual(push_request["type"], "push_approval")
        self.assertEqual(push_request["tests"], "no_tests")
        self.assertIn("NOT verified", push_request["message"])
        self.assertIn("'status': 'no_tests'", second["test_result"])


class CleanupWorkspacesCommandTests(TestCase):
    """cleanup_workspaces removes old finished clones and keeps the rest."""

    def setUp(self):
        import os
        import pathlib
        import tempfile
        import shutil
        from unittest import mock

        from core import agent_service

        self.root = pathlib.Path(tempfile.mkdtemp(prefix="devpilot_ws_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "runs").mkdir()
        patcher = mock.patch.object(agent_service, "WORKSPACE_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.user = User.objects.create_user(username="cleanup_u", password="Testuser@123")
        self.os = os

    def _make(self, status, age_days):
        thread_id = uuid.uuid4()
        d = self.root / "runs" / str(thread_id)
        d.mkdir()
        (d / "f.txt").write_text("x")
        AgentRun.objects.create(user=self.user, thread_id=thread_id, task="t", status=status)
        old = __import__("time").time() - age_days * 86400
        self.os.utime(d, (old, old))
        return d

    def _run(self, *args):
        from django.core.management import call_command
        from io import StringIO

        out = StringIO()
        call_command("cleanup_workspaces", *args, stdout=out)
        return out.getvalue()

    def test_removes_only_old_finished_runs(self):
        old_done = self._make("completed", 30)
        old_paused = self._make("waiting_for_approval", 30)
        recent_done = self._make("completed", 1)

        self._run("--days", "7")

        self.assertFalse(old_done.exists())
        self.assertTrue(old_paused.exists(), "paused runs can still be resumed")
        self.assertTrue(recent_done.exists())

    def test_dry_run_deletes_nothing(self):
        old_done = self._make("rejected", 30)
        self._run("--days", "7", "--dry-run")
        self.assertTrue(old_done.exists())

    def test_ignores_directories_that_are_not_run_ids(self):
        stray = self.root / "runs" / "not-a-uuid"
        stray.mkdir()
        self.os.utime(stray, (1, 1))
        self._run("--days", "1")
        self.assertTrue(stray.exists())


class SerializeResultTests(TestCase):
    """A failed branch step must be reported, not end silently."""

    def test_branch_failure_is_reported_as_error_with_reason(self):
        from core.agent_service import serialize_result

        payload = serialize_result({"branch_result": "error: fatal: something broke"})

        self.assertEqual(payload["status"], "error")
        self.assertIn("something broke", payload["error"])
        self.assertTrue(payload["branch_result"].startswith("error:"))

    def test_successful_branch_does_not_change_status(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "branch_result": "success: devpilot/x",
            "final_result": "{'status': 'success'}",
        })

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["branch_result"], "success: devpilot/x")


class TestOutcomeStatusTests(TestCase):
    """Results must not claim success when tests failed or never ran."""

    def test_failed_tests_are_not_reported_as_completed(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "final_result": "{'status': 'success'}",
            "test_result": "{'status': 'failed', 'stdout': 'x'}",
        })
        self.assertEqual(payload["status"], "tests_failed")

    def test_tool_error_is_not_reported_as_completed(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "final_result": "{'status': 'success'}",
            "test_result": "{'status': 'error', 'message': 'timed out'}",
        })
        self.assertEqual(payload["status"], "tests_failed")

    def test_no_tests_run_tests_status(self):
        import pathlib
        import tempfile
        import shutil
        from core.tools import run_tests, set_workspace, reset_workspace

        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "hello.py").write_text("print('hi')\n")

        token = set_workspace(tmp)
        try:
            self.assertEqual(run_tests.invoke({})["status"], "no_tests")
        finally:
            reset_workspace(token)

    def test_failed_push_is_not_reported_as_completed(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "final_result": "{'status': 'success'}",
            "test_result": "{'status': 'no_tests'}",
            "commit_result": "{'status': 'success'}",
            "push_result": "{'status': 'error', 'message': 'GITHUB_TOKEN is not set; cannot push to GitHub.'}",
        })

        self.assertEqual(payload["status"], "push_failed")
        self.assertIn("GITHUB_TOKEN is not set", payload["error"])
        self.assertEqual(payload["pr_url"], "")

    def test_failed_pull_request_is_reported_separately(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "final_result": "{'status': 'success'}",
            "push_result": "{'status': 'success'}",
            "pr_result": "{'status': 'error', 'message': 'Resource not accessible'}",
        })

        self.assertEqual(payload["status"], "pr_failed")
        self.assertIn("Resource not accessible", payload["error"])

    def test_successful_delivery_is_completed(self):
        from core.agent_service import serialize_result

        payload = serialize_result({
            "final_result": "ok",
            "push_result": "{'status': 'success'}",
            "pr_result": "{'status': 'success', 'url': 'https://github.com/o/r/pull/1'}",
            "pr_url": "https://github.com/o/r/pull/1",
        })

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["error"], "")