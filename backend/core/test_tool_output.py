import os
import tempfile
from unittest import mock

from django.test import SimpleTestCase

from core.tools import _MAX_TEST_OUTPUT_CHARS, _bounded_output, _build_test_command, _sanitized_env


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
        self.assertIn("--as=2147483648:2147483648", command)
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
