import os
import time
from typing import Any


class Qwen3_4B_Runner:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-4B-Instruct",
        device: str | None = None,
        mode: str = "mock",
    ):
        self.model_name_or_path = model_name_or_path
        self.mode = mode.lower()
        self.device = device
        self.tokenizer: Any = None
        self.model: Any = None
        self.precision_str = "float32"

        if self.mode not in ("real", "mock"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'real' or 'mock'.")

        if self.mode == "real":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.device = device_str

                if torch.cuda.is_available():
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    self.precision_str = "bfloat16" if dtype == torch.bfloat16 else "float16"
                else:
                    dtype = torch.float32
                    self.precision_str = "float32"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                if not torch.cuda.is_available() and self.device == "cpu":
                    self.model = self.model.to("cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"CRITICAL: Failed to load real model '{self.model_name_or_path}': {exc}.\n"
                    f"In '--mode real', implicit fallback to mock mode is strictly forbidden to prevent invalid experimental reporting."
                ) from exc

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

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        output_tokens = len(generated_ids)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return output_text, input_len, output_tokens, latency_ms

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
