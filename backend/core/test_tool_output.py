import os
import subprocess
import sys
import shutil
import tempfile
from unittest import mock
import unittest

from django.test import SimpleTestCase

from core.tools import _MAX_TEST_OUTPUT_CHARS, _bounded_output, _build_test_command, _run_bounded_subprocess, _sanitized_env


class BoundedTestOutputTests(SimpleTestCase):
    def test_short_output_is_unchanged(self):
        self.assertEqual(_bounded_output("short output"), "short output")

    def test_bytes_are_decoded_safely(self):
        self.assertEqual(_bounded_output(b"hello \xff"), "hello \ufffd")

    def test_long_output_is_truncated_with_marker(self):
        output = "x" * (_MAX_TEST_OUTPUT_CHARS + 123)
        result = _bounded_output(output)

        self.assertTrue(result.startswith("x" * _MAX_TEST_OUTPUT_CHARS))
        self.assertIn("truncated 123 characters", result)
        self.assertLess(len(result), len(output) + 100)

    def test_sanitized_environment_drops_secrets_and_host_access_sockets(self):
        source = {
            "PATH": "/usr/bin",
            "GITHUB_TOKEN": "do-not-leak",
            "DATABASE_URL": "postgres://secret",
            "SSH_AUTH_SOCK": "/tmp/agent.sock",
            "DOCKER_HOST": "unix:///var/run/docker.sock",
            "LANG": "C.UTF-8",
        }
        with tempfile.TemporaryDirectory() as workspace:
            with mock.patch.dict(os.environ, source, clear=True):
                env = _sanitized_env(workspace)

        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["LANG"], "C.UTF-8")
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("DATABASE_URL", env)
        self.assertNotIn("SSH_AUTH_SOCK", env)
        self.assertNotIn("DOCKER_HOST", env)
        self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
        self.assertEqual(env["HOME"], os.path.join(workspace, ".devpilot-home"))


    def test_unsafe_local_mode_preserves_command_explicitly(self):
        command = ["python", "-m", "pytest"]
        self.assertEqual(
            _build_test_command(command, "/tmp/workspace", mode="unsafe-local"),
            command,
        )

    def test_bubblewrap_mode_clears_environment_and_disables_network(self):
        with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
            command = _build_test_command(
                ["npm", "test"], "/tmp/workspace", mode="bubblewrap"
            )

        self.assertIn("--unshare-all", command)
        self.assertIn("--clearenv", command)
        self.assertIn("--ro-bind", command)
        self.assertIn("--bind", command)
        self.assertIn("--cpu=120:120", command)
        self.assertIn("--as=6442450944:6442450944", command)
        self.assertEqual(command[-2:], ["npm", "test"])

    def test_bubblewrap_mode_fails_closed_without_prerequisites(self):
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(FileNotFoundError):
                _build_test_command(
                    ["npm", "test"], "/tmp/workspace", mode="bubblewrap"
                )

    def test_unknown_sandbox_mode_does_not_execute_unsandboxed(self):
        with self.assertRaises(ValueError):
            _build_test_command(
                ["npm", "test"], "/tmp/workspace", mode="disabled"
            )


    def test_subprocess_output_is_bounded_while_running(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = _run_bounded_subprocess(
                [sys.executable, "-c", "print('x' * 10000)"],
                cwd=workspace,
                env={"PATH": os.environ.get("PATH", "")},
                timeout=10,
                max_output_bytes=128,
            )
        self.assertEqual(result.returncode, 0)
        self.assertLess(len(result.stdout), 200)
        self.assertIn("capture truncated", result.stdout)

    def test_subprocess_timeout_kills_the_process(self):
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaises(subprocess.TimeoutExpired):
                _run_bounded_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    cwd=workspace,
                    env={"PATH": os.environ.get("PATH", "")},
                    timeout=0.1,
                    max_output_bytes=128,
                )


    def test_workspace_growth_limit_terminates_process(self):
        script = (
            "import pathlib,time; "
            "pathlib.Path('large.bin').write_bytes(b'x' * 4096); "
            "time.sleep(5)"
        )
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaisesRegex(RuntimeError, "workspace output exceeded"):
                _run_bounded_subprocess(
                    [sys.executable, "-c", script],
                    cwd=workspace,
                    env={"PATH": os.environ.get("PATH", "")},
                    timeout=10,
                    max_output_bytes=128,
                    max_workspace_bytes=1024,
                    max_workspace_files=10,
                )

    @unittest.skipUnless(shutil.which("bwrap") and shutil.which("prlimit"), "Bubblewrap not installed")
    def test_bubblewrap_executes_inside_isolation(self):
        with tempfile.TemporaryDirectory() as workspace:
            command = _build_test_command(
                [sys.executable, "-c", "print('sandbox-ok')"],
                workspace,
                mode="bubblewrap",
            )
            result = _run_bounded_subprocess(
                command,
                cwd=workspace,
                env=_sanitized_env(workspace),
                timeout=15,
                max_output_bytes=1024,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sandbox-ok", result.stdout)
