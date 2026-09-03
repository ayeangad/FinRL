import time
from typing import Any


class Qwen3_1_7B_Runner:
    def __init__(
        self,
        checkpoint: str = "Qwen/Qwen3-1.7B",
        device: str | None = None,
        mode: str = "mock",
    ):
        self.checkpoint = checkpoint
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
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )

                if self.device is None:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint,
                    trust_remote_code=True,
                )

                if self.device == "cuda" and torch.cuda.is_available():
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    self.precision_str = "4bit-nf4"
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.checkpoint,
                        quantization_config=quantization_config,
                        device_map={"": 0},
                        trust_remote_code=True,
                    )
                else:
                    raise RuntimeError(
                        "Real Qwen3-1.7B inference requires CUDA in the current "
                        "memory-constrained configuration."
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"CRITICAL: Failed to load real model '{self.checkpoint}' on device '{self.device}': {exc}.\n"
                    f"In '--mode real', implicit fallback to mock mode is strictly forbidden to prevent invalid experimental reporting."
                ) from exc

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
