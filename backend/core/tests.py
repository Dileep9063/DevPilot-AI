"""
Automated tests for DevPilot AI.

These replace the one-off manual scripts that used to live scattered
across backend/ and backend/core/ (test_ai_debug.py, hitl_test.py,
test_debug_pipeline.py, etc). Those scripts were useful during
development but called Gemini for real, printed to stdout, and had to
be run and read by hand.

Here we test the same logic in a repeatable, offline way:
  - The deterministic graph nodes/routing (no LLM calls involved).
  - The sandboxed file tools.
  - The API endpoints' request validation.

Anything that requires a real Gemini call (the LLM-driven nodes, or a
full agent_run against a live model) is intentionally left for manual
end-to-end testing with a real GEMINI_API_KEY — see the project
README for that checklist.
"""

from langchain_core.messages import ToolMessage
from langgraph.graph import END
from rest_framework.test import APIClient
from django.test import TestCase

from core.agent import (
    extract_proposal_node,
    debug_node,
    route_after_test,
    route_after_debug_fix,
    write_node,
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
        self.assertIn("def login", result["proposed_content"])

    def test_extract_proposal_ignores_non_dict_content(self):
        message = ToolMessage(content="not a dict", tool_call_id="1")

        result = extract_proposal_node({"messages": [message]})

        self.assertEqual(result, {})

    def test_debug_node_detects_failure(self):
        state = {
            "test_result": "FAILED test_auth.py::test_login - assert False"
        }

        result = debug_node(state)

        self.assertEqual(result["failed_file"], "test_auth.py")
        self.assertIn("failure_detected", result["debug_result"])

    def test_debug_node_no_failure(self):
        result = debug_node({"test_result": "1 passed in 0.05s"})

        self.assertEqual(result["failed_file"], "")
        self.assertEqual(result["debug_result"], "No test failure detected.")

    def test_route_after_test(self):
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'passed'}"}),
            "done",
        )
        self.assertEqual(
            route_after_test({"test_result": "{'status': 'failed'}"}),
            "debug",
        )

    def test_route_after_debug_fix_with_proposal(self):
        state = {"proposed_path": "auth.py", "proposed_content": "code"}

        self.assertEqual(route_after_debug_fix(state), "approval")

    def test_route_after_debug_fix_without_proposal(self):
        state = {"proposed_path": "", "proposed_content": ""}

        self.assertEqual(route_after_debug_fix(state), END)

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

            self.assertEqual(result["approval"], "")

        finally:
            (WORKSPACE / test_path).unlink(missing_ok=True)


class ToolsTests(TestCase):
    """The sandboxed file tools the agent is allowed to call."""

    def test_write_then_read_file_roundtrip(self):
        test_path = "unit_test_roundtrip.py"

        try:
            write_result = write_file.invoke({
                "path": test_path,
                "content": "print('hi')",
            })

            self.assertEqual(write_result["status"], "success")

            read_result = read_file.invoke({
                "path": str(WORKSPACE / test_path),
            })

            self.assertEqual(read_result["content"], "print('hi')")

        finally:
            (WORKSPACE / test_path).unlink(missing_ok=True)

    def test_propose_file_change_blocks_path_traversal(self):
        result = propose_file_change.invoke({
            "path": "../outside_workspace.py",
            "content": "x = 1",
        })

        self.assertEqual(result["status"], "error")

    def test_propose_file_change_returns_pending_approval(self):
        result = propose_file_change.invoke({
            "path": "auth.py",
            "content": "x = 1",
        })

        self.assertEqual(result["status"], "pending_approval")
        self.assertEqual(result["path"], "auth.py")


class AgentApiValidationTests(TestCase):
    """Request validation for the agent endpoints (no Gemini calls)."""

    def setUp(self):
        self.client = APIClient()

    def test_status_endpoint(self):
        response = self.client.get("/api/status/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["status"], "success")

    def test_agent_run_requires_task(self):
        response = self.client.post("/api/agent/run/", {}, format="json")

        self.assertEqual(response.status_code, 400)

    def test_agent_resume_requires_thread_id(self):
        response = self.client.post(
            "/api/agent/resume/",
            {"decision": "approve"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)

    def test_agent_resume_requires_valid_decision(self):
        response = self.client.post(
            "/api/agent/resume/",
            {"thread_id": "abc", "decision": "maybe"},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
