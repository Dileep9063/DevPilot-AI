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
from django.test import TestCase, override_settings

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from rest_framework.test import APIClient

from .models import AgentRun, GitHubInstallation

from core.agent import (
    extract_proposal_node,
    route_after_test,
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
            name="propose_file_change",
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
            "test_failed",
        )

        # No tests found must not block delivery; a real error must.
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'no_tests'}"}),
            "commit",
        )
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'error'}"}),
            "test_failed",
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

    def test_agent_run_rejects_reusing_owned_thread_to_protect_checkpoint(self):
        existing_thread_id = uuid.uuid4()
        existing_run = AgentRun.objects.create(
            user=self.user,
            thread_id=existing_thread_id,
            task="existing task",
            repo_url="https://github.com/owner/repo",
        )
        before = AgentRun.objects.count()

        response = self.client.post(
            "/api/agent/run/",
            {
                "task": "start a different task",
                "repo_url": "https://github.com/owner/repo",
                "thread_id": str(existing_thread_id),
            },
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Use /api/agent/resume/", response.data["error"])
        self.assertEqual(AgentRun.objects.count(), before)
        existing_run.refresh_from_db()
        self.assertEqual(existing_run.task, "existing task")

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
    def test_agent_run_rejects_non_string_task(self):
        response = self.client.post(
            "/api/agent/run/",
            {"task": ["unexpected"], "repo_url": "https://github.com/owner/repo"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_agent_run_rejects_oversized_task(self):
        response = self.client.post(
            "/api/agent/run/",
            {"task": "x" * 4001, "repo_url": "https://github.com/owner/repo"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)

    def test_agent_run_rejects_non_github_url_before_creating_run(self):
        before = AgentRun.objects.count()
        response = self.client.post(
            "/api/agent/run/",
            {"task": "test", "repo_url": "https://example.com/owner/repo"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(AgentRun.objects.count(), before)

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

        # After every write the agent is consulted again; replying DONE ends the
        # task phase. Stubbing it keeps these tests offline and deterministic.
        from unittest import mock
        from langchain_core.messages import AIMessage
        import core.agent as agent_module

        llm = mock.patch.object(
            agent_module, "invoke_llm", return_value=AIMessage(content="DONE")
        )
        llm.start()
        self.addCleanup(llm.stop)

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

    def _start(self, path="calc.py", content=None, proposal_type="existing_file", installation_id=0):
        from core import agent_service
        from core.agent import agent

        if content is None:
            content = 'def add(a, b):\n    """Return a + b."""\n    return a + b\n'

        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        agent.update_state(
            cfg,
            agent_service._initial_state("Document add()", self.work, thread_id, installation_id),
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


    def _fake_github_api(self):
        from unittest import mock

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
        fake.post.return_value = Resp(201, {"html_url": "https://github.com/test/repo/pull/9", "number": 9})
        return fake

    def test_github_app_mode_mints_fresh_tokens_and_never_stores_them(self):
        import json
        import os
        from unittest import mock
        from langgraph.types import Command
        from core import github_app, tools

        agent, cfg, thread_id = self._start(installation_id=4242)
        agent.invoke(None, cfg)
        agent.invoke(Command(resume="approve"), cfg)

        fake = self._fake_github_api()
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}

        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(tools, "requests", fake), \
                mock.patch.object(github_app, "installation_token", return_value="ghs_secret") as mint:
            final = agent.invoke(Command(resume="approve"), cfg)

        # Minted separately for the push and for the PR, scoped to this repo.
        self.assertEqual(mint.call_count, 2)
        mint.assert_called_with(4242, "work")

        # The token reached the GitHub API even though GITHUB_TOKEN is unset.
        self.assertEqual(fake.get.call_args.kwargs["headers"]["Authorization"], "Bearer ghs_secret")
        self.assertEqual(final["pr_url"], "https://github.com/test/repo/pull/9")

        # ...and was never persisted: not in the checkpoint, not in .git/config.
        saved = json.dumps(agent.get_state(cfg).values, default=str)
        self.assertNotIn("ghs_secret", saved)
        self.assertIn("4242", saved)
        with open(f"{self.work}/.git/config") as f:
            self.assertNotIn("ghs_secret", f.read())

    def test_losing_the_installation_is_reported_as_push_failed(self):
        from unittest import mock
        from langgraph.types import Command
        from core import agent_service, github_app

        agent, cfg, thread_id = self._start(installation_id=4242)
        agent.invoke(None, cfg)
        agent.invoke(Command(resume="approve"), cfg)

        with mock.patch.object(
            github_app, "installation_token",
            side_effect=github_app.GitHubAccessError("App is no longer installed"),
        ):
            final = agent.invoke(Command(resume="approve"), cfg)

        payload = agent_service.serialize_result(final)
        self.assertEqual(payload["status"], "push_failed")
        self.assertIn("no longer installed", payload["error"])
        self.assertEqual(self._remote_branches(), ["main"])

    def test_build_output_created_during_tests_is_not_committed(self):
        """
        Regression: `npm run build` created dist/ and `git add -A` put it in the
        pull request. Only files the agent wrote may be committed.
        """
        import os
        import subprocess
        from unittest import mock
        from langchain_core.messages import AIMessage
        from langgraph.types import Command
        import core.agent as agent_module
        from core import agent_service

        # The project's test run creates build output, like `npm run build` does.
        with open(f"{self.work}/test_calc.py", "w") as f:
            f.write(
                "import os\nfrom calc import add\n\n"
                "def test_add():\n"
                "    os.makedirs('dist', exist_ok=True)\n"
                "    open('dist/bundle.js', 'w').write('x')\n"
                "    assert add(2, 3) == 5\n"
            )

        def propose(path, content, call_id):
            return AIMessage(content="", tool_calls=[{
                "name": "propose_new_file",
                "args": {"path": path, "content": content},
                "id": call_id,
                "type": "tool_call",
            }])

        replies = [
            propose("toggle.js", "export const toggle = () => {};\n", "c1"),
            propose("styles.css", ".dark { color: white; }\n", "c2"),
            AIMessage(content="DONE added the toggle and its styles."),
        ]

        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        init = agent_service._initial_state("Add a dark toggle", self.work, thread_id)

        with mock.patch.object(agent_module, "invoke_llm", side_effect=replies):
            graph = agent_module.agent
            first = graph.invoke(init, cfg)["__interrupt__"][0].value
            second = graph.invoke(Command(resume="approve"), cfg)["__interrupt__"][0].value
            push_request = graph.invoke(Command(resume="approve"), cfg)["__interrupt__"][0].value

        self.assertEqual((first["path"], second["path"]), ("toggle.js", "styles.css"))
        self.assertEqual(push_request["type"], "push_approval")

        # The build output really was created in the working tree...
        self.assertTrue(os.path.exists(f"{self.work}/dist/bundle.js"))

        # ...but the commit holds only the two files the agent wrote.
        files = subprocess.run(
            ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
            cwd=self.work, capture_output=True, text=True,
        ).stdout.split()
        self.assertEqual(sorted(files), ["styles.css", "toggle.js"])


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


# ============================================================
# GitHub App (per-user repository access)
# ============================================================

def _rsa_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private, public


APP_ENV_KEYS = (
    "GITHUB_APP_ID", "GITHUB_APP_SLUG", "GITHUB_APP_CLIENT_ID",
    "GITHUB_APP_CLIENT_SECRET", "GITHUB_APP_PRIVATE_KEY", "GITHUB_APP_PRIVATE_KEY_PATH",
)


class GitHubAppTestBase(TestCase):
    """Sets (or clears) the GitHub App environment for a test."""

    def configure_app(self, enabled=True):
        import os
        from unittest import mock

        env = {k: v for k, v in os.environ.items() if k not in APP_ENV_KEYS}
        self.private_pem, self.public_pem = _rsa_pem()

        if enabled:
            env.update({
                "GITHUB_APP_ID": "123456",
                "GITHUB_APP_SLUG": "devpilot-test-app",
                "GITHUB_APP_CLIENT_ID": "Iv1.client",
                "GITHUB_APP_CLIENT_SECRET": "client-secret",
                "GITHUB_APP_PRIVATE_KEY": self.private_pem.replace("\n", "\\n"),
            })

        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class GitHubAppUnitTests(GitHubAppTestBase):

    def test_not_configured_without_environment(self):
        from core import github_app

        self.configure_app(enabled=False)
        self.assertFalse(github_app.is_configured())

    def test_configured_with_escaped_newline_key(self):
        from core import github_app

        self.configure_app()
        self.assertTrue(github_app.is_configured())

    def test_private_key_can_come_from_a_file(self):
        import os
        import tempfile
        from core import github_app

        self.configure_app()
        os.environ.pop("GITHUB_APP_PRIVATE_KEY")
        with tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False) as f:
            f.write(self.private_pem)
        self.addCleanup(os.unlink, f.name)
        os.environ["GITHUB_APP_PRIVATE_KEY_PATH"] = f.name

        self.assertTrue(github_app.is_configured())

    def test_app_jwt_is_rs256_and_short_lived(self):
        import jwt
        from core import github_app

        self.configure_app()
        token = github_app.app_jwt()
        claims = jwt.decode(token, self.public_pem, algorithms=["RS256"], options={"verify_iat": False})

        self.assertEqual(claims["iss"], "123456")
        self.assertLessEqual(claims["exp"] - claims["iat"], 10 * 60 + 60)

    def test_state_round_trip_and_tampering(self):
        from core import github_app

        state = github_app.make_state(7)
        self.assertEqual(github_app.read_state(state), 7)
        self.assertIsNone(github_app.read_state(state + "x"))
        self.assertIsNone(github_app.read_state(""))

    def test_expired_state_is_rejected(self):
        from unittest import mock
        from core import github_app

        state = github_app.make_state(7)
        with mock.patch.object(github_app, "STATE_MAX_AGE", -1):
            self.assertIsNone(github_app.read_state(state))

    def test_connect_url_carries_state(self):
        from core import github_app

        self.configure_app()
        url = github_app.connect_url("abc:def")
        self.assertTrue(url.startswith("https://github.com/apps/devpilot-test-app/installations/new?state="))

    def _resp(self, code, data=None):
        from unittest import mock

        r = mock.MagicMock()
        r.status_code = code
        r.json.return_value = data or {}
        r.raise_for_status.side_effect = None if code < 400 else Exception(f"HTTP {code}")
        return r

    def test_installation_token_is_scoped_to_one_repository(self):
        from unittest import mock
        from core import github_app

        self.configure_app()
        with mock.patch.object(github_app.requests, "post", return_value=self._resp(201, {"token": "ghs_x"})) as post:
            token = github_app.installation_token(55, "my-repo")

        self.assertEqual(token, "ghs_x")
        self.assertEqual(post.call_args.kwargs["json"], {"repositories": ["my-repo"]})
        self.assertTrue(post.call_args.kwargs["headers"]["Authorization"].startswith("Bearer "))
        self.assertIn("/app/installations/55/access_tokens", post.call_args.args[0])

    def test_installation_token_errors_are_actionable(self):
        from unittest import mock
        from core import github_app

        self.configure_app()
        for code, fragment in ((404, "no longer installed"), (422, "does not have access")):
            with mock.patch.object(github_app.requests, "post", return_value=self._resp(code)):
                with self.assertRaises(github_app.GitHubAccessError) as ctx:
                    github_app.installation_token(55, "my-repo")
            self.assertIn(fragment, str(ctx.exception))

    def test_find_repo_installation(self):
        from unittest import mock
        from core import github_app

        self.configure_app()
        with mock.patch.object(github_app.requests, "get", return_value=self._resp(200, {"id": 99})):
            self.assertEqual(github_app.find_repo_installation("o", "r"), 99)
        with mock.patch.object(github_app.requests, "get", return_value=self._resp(404)):
            self.assertIsNone(github_app.find_repo_installation("o", "r"))

    def test_clone_credential_is_not_written_to_git_config(self):
        from unittest import mock
        from core import agent_service

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return mock.MagicMock()

        thread_id = agent_service.new_thread_id()
        with mock.patch.object(agent_service.subprocess, "run", side_effect=fake_run):
            agent_service._prepare_repository("https://github.com/o/r", thread_id, "ghs_secret")

        cmd = calls[0]
        # top-level `git -c ... clone` is per-process; `clone -c` would persist it
        self.assertLess(cmd.index("-c"), cmd.index("clone"))
        self.assertTrue(cmd[cmd.index("-c") + 1].startswith("http.extraheader="))
        self.assertNotIn("ghs_secret", " ".join(cmd))  # base64-encoded, not plain

        import shutil
        shutil.rmtree(agent_service._workspace_for_thread(thread_id), ignore_errors=True)

    def test_clone_failure_message_hides_the_token(self):
        import subprocess
        from unittest import mock
        from core import agent_service

        err = subprocess.CalledProcessError(128, "git", stderr="fatal: bad ghs_secret creds")
        thread_id = agent_service.new_thread_id()
        with mock.patch.object(agent_service.subprocess, "run", side_effect=err):
            with self.assertRaises(ValueError) as ctx:
                agent_service._prepare_repository("https://github.com/o/r", thread_id, "ghs_secret")

        self.assertNotIn("ghs_secret", str(ctx.exception))

        import shutil
        shutil.rmtree(agent_service._workspace_for_thread(thread_id), ignore_errors=True)


class GitHubAccessResolutionTests(GitHubAppTestBase):

    def setUp(self):
        self.configure_app()
        self.alice = User.objects.create_user(username="alice_gh", password="Testuser@123")
        self.bob = User.objects.create_user(username="bob_gh", password="Testuser@123")

    def test_user_who_has_not_connected_is_told_to_connect(self):
        from core import github_app

        with self.assertRaises(github_app.GitHubAccessError) as ctx:
            github_app.resolve_repo_access(self.alice, "alice", "repo")
        self.assertIn("Connect your GitHub account", str(ctx.exception))

    def test_repo_without_the_app_installed(self):
        from unittest import mock
        from core import github_app

        GitHubInstallation.objects.create(user=self.alice, installation_id=1)
        with mock.patch.object(github_app, "find_repo_installation", return_value=None):
            with self.assertRaises(github_app.GitHubAccessError) as ctx:
                github_app.resolve_repo_access(self.alice, "alice", "repo")
        self.assertIn("not installed on alice/repo", str(ctx.exception))

    def test_cannot_use_someone_elses_installation(self):
        """Bob must not push to Alice's repo just because the App is installed there."""
        from unittest import mock
        from core import github_app

        GitHubInstallation.objects.create(user=self.alice, installation_id=1)
        GitHubInstallation.objects.create(user=self.bob, installation_id=2)

        with mock.patch.object(github_app, "find_repo_installation", return_value=1):
            self.assertEqual(github_app.resolve_repo_access(self.alice, "alice", "repo"), 1)
            with self.assertRaises(github_app.GitHubAccessError) as ctx:
                github_app.resolve_repo_access(self.bob, "alice", "repo")
        self.assertIn("not connected to your DevPilot account", str(ctx.exception))


class GitHubConnectApiTests(GitHubAppTestBase):

    def setUp(self):
        self.configure_app()
        self.client = APIClient()
        self.user = User.objects.create_user(username="gh_user", password="Testuser@123")

    def _callback(self, **params):
        from django.test import Client

        return Client().get("/api/github/callback/", params)

    def test_endpoints_require_authentication(self):
        for path in ("/api/github/status/", "/api/github/connect/"):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post("/api/github/disconnect/").status_code, 401)

    def test_status_reports_configuration_and_connections(self):
        self.client.force_authenticate(user=self.user)
        data = self.client.get("/api/github/status/").json()
        self.assertTrue(data["app_configured"])
        self.assertFalse(data["connected"])

        GitHubInstallation.objects.create(user=self.user, installation_id=5, account_login="octo")
        data = self.client.get("/api/github/status/").json()
        self.assertTrue(data["connected"])
        self.assertEqual(data["installations"][0]["account"], "octo")

    def test_connect_returns_install_url_with_signed_state(self):
        from urllib.parse import parse_qs, urlparse
        from core import github_app

        self.client.force_authenticate(user=self.user)
        url = self.client.get("/api/github/connect/").json()["url"]
        state = parse_qs(urlparse(url).query)["state"][0]
        self.assertEqual(github_app.read_state(state), self.user.id)

    def test_connect_fails_cleanly_when_app_not_configured(self):
        self.configure_app(enabled=False)
        self.client.force_authenticate(user=self.user)
        self.assertEqual(self.client.get("/api/github/connect/").status_code, 400)

    def test_callback_saves_a_verified_installation(self):
        from unittest import mock
        from core import github_app

        state = github_app.make_state(self.user.id)
        with mock.patch.object(github_app, "exchange_code", return_value="user-token"), \
                mock.patch.object(github_app, "user_installations", return_value=[
                    {"id": 77, "account": {"login": "octo"}, "repository_selection": "selected"}]):
            res = self._callback(code="c", installation_id="77", setup_action="install", state=state)

        self.assertEqual(res.status_code, 302)
        self.assertIn("github=connected", res["Location"])
        saved = GitHubInstallation.objects.get(user=self.user)
        self.assertEqual((saved.installation_id, saved.account_login), (77, "octo"))

    def test_callback_rejects_an_installation_the_user_does_not_own(self):
        """Forged installation_id: GitHub's own list for this user does not contain it."""
        from unittest import mock
        from core import github_app

        state = github_app.make_state(self.user.id)
        with mock.patch.object(github_app, "exchange_code", return_value="user-token"), \
                mock.patch.object(github_app, "user_installations", return_value=[{"id": 1, "account": {"login": "me"}}]):
            res = self._callback(code="c", installation_id="999", state=state)

        self.assertIn("github=error", res["Location"])
        self.assertEqual(GitHubInstallation.objects.count(), 0)

    def test_callback_rejects_bad_state_and_missing_code(self):
        from core import github_app

        res = self._callback(code="c", installation_id="1", state="forged")
        self.assertIn("github=error", res["Location"])

        res = self._callback(installation_id="1", state=github_app.make_state(self.user.id))
        self.assertIn("github=error", res["Location"])
        self.assertEqual(GitHubInstallation.objects.count(), 0)

    def test_disconnect_removes_connections(self):
        GitHubInstallation.objects.create(user=self.user, installation_id=5)
        self.client.force_authenticate(user=self.user)
        self.assertEqual(self.client.post("/api/github/disconnect/").json()["disconnected"], 1)
        self.assertEqual(GitHubInstallation.objects.count(), 0)

    def test_run_requires_a_connected_account_in_app_mode(self):
        self.client.force_authenticate(user=self.user)
        res = self.client.post("/api/agent/run/", {"task": "t", "repo_url": "https://github.com/o/r"}, format="json")
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["code"], "github_access")

    def test_run_passes_the_resolved_installation_to_the_agent(self):
        from unittest import mock
        from core import agent_service, github_app

        GitHubInstallation.objects.create(user=self.user, installation_id=99)
        self.client.force_authenticate(user=self.user)

        with mock.patch.object(github_app, "find_repo_installation", return_value=99), \
                mock.patch.object(agent_service, "run_agent_task", return_value={"final_result": "ok"}) as run:
            res = self.client.post("/api/agent/run/", {"task": "t", "repo_url": "https://github.com/o/r"}, format="json")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(run.call_args.args[3], 99)

    def test_single_token_mode_still_works_without_an_app(self):
        from unittest import mock
        from core import agent_service

        self.configure_app(enabled=False)
        self.client.force_authenticate(user=self.user)

        with mock.patch.object(agent_service, "run_agent_task", return_value={"final_result": "ok"}) as run:
            res = self.client.post("/api/agent/run/", {"task": "t", "repo_url": "https://github.com/o/r"}, format="json")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(run.call_args.args[3], 0)

    def test_async_run_is_queued_without_blocking_for_agent(self):
        from unittest import mock
        from core import agent_service

        self.configure_app(enabled=False)
        self.client.force_authenticate(user=self.user)

        with override_settings(DEVPILOT_ASYNC_RUNS=True), mock.patch("core.tasks.run_agent_job.delay") as enqueue:
            response = self.client.post(
                "/api/agent/run/",
                {"task": "Add a test", "repo_url": "https://github.com/o/r"},
                format="json",
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.data["status"], "queued")
        self.assertEqual(AgentRun.objects.get(thread_id=response.data["thread_id"]).status, "queued")
        enqueue.assert_called_once()

class AgentWorkerReliabilityTests(TestCase):
    """Regression tests for queue timestamps, stale-run recovery and isolation config."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="worker_reliability", password="Testuser@123"
        )
        self.thread_id = uuid.uuid4()
        self.run = AgentRun.objects.create(
            user=self.user,
            thread_id=self.thread_id,
            task="Reliability test",
            repo_url="https://github.com/example/repo",
            status="queued",
        )

    def test_claiming_queued_run_refreshes_updated_at(self):
        from datetime import timedelta
        from django.utils import timezone
        from core.tasks import _claim_queued_run

        old_time = timezone.now() - timedelta(hours=2)
        AgentRun.objects.filter(pk=self.run.pk).update(updated_at=old_time)
        self.assertTrue(_claim_queued_run(str(self.thread_id)))

        self.run.refresh_from_db()
        self.assertEqual(self.run.status, "running")
        self.assertGreater(self.run.updated_at, old_time)

    def test_stale_recovery_marks_old_runs_but_preserves_recent_runs(self):
        from datetime import timedelta
        from django.utils import timezone
        from core.tasks import recover_stale_agent_runs

        stale_id = self.thread_id
        fresh_id = uuid.uuid4()
        AgentRun.objects.create(
            user=self.user,
            thread_id=fresh_id,
            task="Still running",
            repo_url="https://github.com/example/repo",
            status="running",
        )
        old_time = timezone.now() - timedelta(hours=2)
        AgentRun.objects.filter(thread_id=stale_id).update(status="running", updated_at=old_time)

        # Worker tasks close/reopen DB connections in production. Keep that
        # cleanup outside this TestCase's transaction-managed connection.
        from unittest import mock
        with mock.patch("core.tasks.close_old_connections"):
            changed = recover_stale_agent_runs.run()
        self.assertEqual(changed, 1)

        self.run.refresh_from_db()
        fresh = AgentRun.objects.get(thread_id=fresh_id)
        self.assertEqual(self.run.status, "error")
        self.assertIn("interrupted", self.run.summary.lower())
        self.assertEqual(fresh.status, "running")

    def test_bubblewrap_command_has_a_bounded_tmpfs(self):
        from unittest import mock
        from core.tools import _build_test_command

        with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
            command = _build_test_command(
                ["npm", "test"], ".", mode="bubblewrap"
            )

        self.assertIn("--size", command)
        size_index = command.index("--size")
        self.assertEqual(int(command[size_index + 1]), 512 * 1024 * 1024)

    def test_production_can_disable_shared_github_token_fallback(self):
        from unittest import mock
        from django.test import override_settings
        from core.tools import get_github_token

        with mock.patch.dict("os.environ", {"GITHUB_TOKEN": "shared-test-token"}):
            with override_settings(DEVPILOT_ALLOW_SHARED_GITHUB_TOKEN=False):
                self.assertEqual(get_github_token(), "")
            with override_settings(DEVPILOT_ALLOW_SHARED_GITHUB_TOKEN=True):
                self.assertEqual(get_github_token(), "shared-test-token")

