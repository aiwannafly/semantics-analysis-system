from typing import List, Optional

from huggingface_hub import InferenceClient

from semantics_analysis.tokens import tokens


class LLMAgent:

    def __init__(
            self,
            model: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            timeout: int = 8,
            use_all_tokens: bool = False,
            huggingface_hub_token: Optional[str] = None
    ):
        self.model = model
        self.token_idx = 0
        if huggingface_hub_token:
            token = huggingface_hub_token
        else:
            token = tokens[self.token_idx]

        self.llm = InferenceClient(model=model, timeout=timeout, token=token)
        self.use_all_tokens = use_all_tokens

    def __call__(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        attempt = 0

        while True:
            try:
                return self.llm.text_generation(
                    prompt,
                    do_sample=False,
                    stop_sequences=stop_sequences,
                    max_new_tokens=max_new_tokens
                ).strip()
            except Exception as e:
                if attempt >= len(tokens) or not self.use_all_tokens:
                    raise e

                attempt += 1
                self.token_idx += 1
                self.token_idx = self.token_idx % len(tokens)
                self.llm = InferenceClient(model=self.model, timeout=8, token=tokens[self.token_idx])
                continue
