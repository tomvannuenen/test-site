import pytest

from neurogen.model import tiny_qwen3


class CharTokenizer:
    """Minimal char-level tokenizer with the subset of the HF API that neurogen uses."""
    name_or_path = "char-tokenizer"
    pad_token_id = 0
    eos_token_id = 0
    chars = [chr(c) for c in range(32, 127)] + ["\n"]

    def __init__(self):
        self.to_id = {c: i + 1 for i, c in enumerate(self.chars)}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [self.to_id.get(c, len(self.chars) + 1) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        return "".join(self.chars[i - 1] if 0 < i <= len(self.chars) else "?" for i in ids)


@pytest.fixture(scope="session")
def tiny_model():
    return tiny_qwen3(seed=0)


@pytest.fixture(scope="session")
def tok():
    return CharTokenizer()
