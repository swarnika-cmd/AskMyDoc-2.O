"""The Groq client is built with reasoning_effort="low" for gpt-oss models, as the evaluation harness sends it."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import llm  # noqa: E402


class GroqReasoningEffort(unittest.TestCase):
    def test_gpt_oss_gets_low_reasoning_effort(self):
        client = llm._groq("openai/gpt-oss-20b", "test-key-not-real")
        self.assertEqual(client.reasoning_effort, "low")
        self.assertAlmostEqual(client.temperature, 0, places=6)

    def test_other_groq_models_do_not_get_the_parameter(self):
        client = llm._groq("llama-3.3-70b-versatile", "test-key-not-real")
        self.assertIsNone(client.reasoning_effort)

    def test_value_reaches_the_request_payload(self):
        client = llm._groq("openai/gpt-oss-20b", "test-key-not-real")
        self.assertEqual(client._default_params.get("reasoning_effort"), "low")


if __name__ == "__main__":
    unittest.main()
