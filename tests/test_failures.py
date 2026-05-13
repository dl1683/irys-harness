from __future__ import annotations

import unittest

from irys_harness.failures import FailureTag, validate_failure_tags


class FailureTests(unittest.TestCase):
    def test_validate_failure_tags_accepts_known_tags(self) -> None:
        self.assertEqual(validate_failure_tags([FailureTag.FORMAT_ERROR.value]), ["format_error"])

    def test_validate_failure_tags_rejects_unknown_tags(self) -> None:
        with self.assertRaises(ValueError):
            validate_failure_tags(["not_real"])


if __name__ == "__main__":
    unittest.main()
