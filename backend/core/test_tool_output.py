from django.test import SimpleTestCase

from core.tools import _MAX_TEST_OUTPUT_CHARS, _bounded_output


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
