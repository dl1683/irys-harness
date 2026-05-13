from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irys_harness.doctor import check_env, read_env_keys


class DoctorTests(unittest.TestCase):
    def test_read_env_keys_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("# comment\nGEMINI_API_KEY=secret\nEMPTY=\n", encoding="utf-8")
            values = read_env_keys(env_path)
            self.assertEqual(values["GEMINI_API_KEY"], "secret")
            self.assertEqual(values["EMPTY"], "")

    def test_check_env_requires_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("GEMINI_API_KEY=\n", encoding="utf-8")
            result = check_env(root)
            self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()

