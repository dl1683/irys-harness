from __future__ import annotations

import types
import unittest

from irys_harness.models.gemini import is_transient_model_error, response_text, usage_tokens


class GeminiModelTests(unittest.TestCase):
    def test_usage_tokens_reads_google_metadata(self) -> None:
        response = types.SimpleNamespace(
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=5,
            )
        )
        self.assertEqual(usage_tokens(response), (12, 5))

    def test_response_text_uses_text_property(self) -> None:
        response = types.SimpleNamespace(text="hello")
        self.assertEqual(response_text(response), "hello")

    def test_is_transient_model_error_detects_provider_502(self) -> None:
        error = RuntimeError("ServerError: 502 Bad Gateway")
        self.assertTrue(is_transient_model_error(error))
        self.assertFalse(is_transient_model_error(ValueError("invalid request")))


if __name__ == "__main__":
    unittest.main()
