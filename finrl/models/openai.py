import os
import time
from typing import Any


class OpenAIRunner:
    def __init__(
        self,
        checkpoint: str = "gpt-5.4-mini",
        mode: str = "mock",
    ):
        self.checkpoint = checkpoint
        self.mode = mode.lower()
        self.client: Any = None

        if self.mode not in ("real", "mock"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'real' or 'mock'.")

        if self.mode == "real":
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "CRITICAL: OPENAI_API_KEY environment variable is missing.\n"
                    "In '--mode real', implicit fallback to mock mode is strictly forbidden to prevent invalid experimental reporting."
                )

            self.client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, int, int, float]:
        t0 = time.perf_counter()

        if self.mode == "mock":
            input_tokens = len(prompt.split())
            output_text = self._mock_generate(prompt)
            output_tokens = len(output_text.split())
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return output_text, input_tokens, output_tokens, latency_ms

        kwargs = {
            "model": self.checkpoint,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_new_tokens,
        }
        
        # New reasoning models do not support temperature changes
        if not (self.checkpoint.startswith("o") or self.checkpoint.startswith("gpt-5")):
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)

        output_text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else len(prompt.split())
        output_tokens = response.usage.completion_tokens if response.usage else len(output_text.split())
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return output_text, input_tokens, output_tokens, latency_ms

    def _mock_generate(self, prompt: str) -> str:
        if "submit_report" in prompt or "calculate_metrics" in prompt:
            return (
                "Thought: Analysis complete. Submitting final Rule 605 report.\n"
                'Action: {"tool_name": "submit_report", "arguments": {"report": "HEADER_OR_PIPE_DATA"}}'
            )
        elif "classify_order" in prompt:
            return (
                "Thought: Order classified. Now calculating metric spreads.\n"
                'Action: {"tool_name": "calculate_metrics", "arguments": {"order_id": "O1"}}'
            )
        else:
            return (
                "Thought: Inspecting initial order and receipt classification.\n"
                'Action: {"tool_name": "classify_order", "arguments": {"order_id": "O1"}}'
            )
