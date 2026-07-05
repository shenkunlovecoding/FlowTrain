from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _eos_id(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else None


def _fallback_chat_text(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _alpaca_prompt(record: Mapping[str, Any]) -> str:
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input", ""))
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _truncate_pair(input_ids: list[int], labels: list[int], max_length: int | None) -> tuple[list[int], list[int]]:
    if max_length is None:
        return input_ids, labels
    return input_ids[:max_length], labels[:max_length]


class RWKVTokenizerAdapter:
    """Expose ``pyrwkv_tokenizer.RWKVTokenizer`` through the small tokenizer
    interface that :class:`SFTJsonlDataset` consumes via :func:`_tokenize`
    (``encode`` / ``decode`` / ``__len__`` / ``pad_token_id`` / ``eos_token_id``).

    The RWKV-7 tokenizer has no pad/EOS tokens and no HuggingFace-style
    ``__call__``/``apply_chat_template``; chat-round termination is the literal
    ``"\\n\\n"`` separator embedded in the prompt/completion text by the
    data-prep step, so ``eos_token_id`` stays ``None`` and the dataset's
    ``add_eos`` becomes a no-op for this adapter. ``pad_token_id=0`` is safe
    because :class:`SFTDataCollator` right-pads (after real tokens, so RWKV
    recurrence is unaffected) and masks pad positions in labels with
    :data:`IGNORE_INDEX`.
    """

    PAD_TOKEN_ID = 0
    VOCAB_SIZE = 65536  # all current RWKV-7 x070 models share this vocab

    def __init__(self) -> None:
        try:
            import pyrwkv_tokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "RWKVTokenizerAdapter requires pyrwkv_tokenizer: pip install pyrwkv_tokenizer"
            ) from exc
        self._tk = pyrwkv_tokenizer.RWKVTokenizer()

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # pyrwkv_tokenizer has no notion of special tokens; the kwarg is accepted
        # only for interface compatibility with the HF tokenizer shape _tokenize uses.
        return list(self._tk.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self._tk.decode([int(i) for i in ids])

    def __len__(self) -> int:
        return self.VOCAB_SIZE

    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int | None:
        return None


class SFTJsonlDataset(Dataset):
    """JSONL SFT dataset for causal LM batches.

    Supported record shapes:
    - {"messages": [{"role": "...", "content": "..."}]}
    - {"prompt": "...", "completion": "..."}
    - {"instruction": "...", "input": "...", "output": "..."}
    - {"text": "..."}
    - {"input_ids": [...], "labels": [...]} for pre-tokenized data
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int | None = None,
        mask_prompt: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.add_eos = add_eos
        with self.path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"{self.path} contains no JSONL records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self._encode_record(self.records[index])
        input_ids, labels = _truncate_pair(input_ids, labels, self.max_length)
        if len(input_ids) < 2:
            raise ValueError("SFT samples must produce at least two tokens")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _encode_record(self, record: Mapping[str, Any]) -> tuple[list[int], list[int]]:
        if "input_ids" in record:
            input_ids = [int(x) for x in record["input_ids"]]
            if "labels" in record:
                labels = [int(x) for x in record["labels"]]
            else:
                labels = list(input_ids)
            if len(input_ids) != len(labels):
                raise ValueError("pre-tokenized input_ids and labels must have the same length")
            return input_ids, labels

        eos = [_eos_id(self.tokenizer)] if self.add_eos and _eos_id(self.tokenizer) is not None else []
        if "messages" in record:
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_ids = list(
                    self.tokenizer.apply_chat_template(
                        record["messages"],
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
            else:
                input_ids = _tokenize(self.tokenizer, _fallback_chat_text(record["messages"]))
            input_ids += eos
            return input_ids, list(input_ids)

        if "prompt" in record and "completion" in record:
            prompt_ids = _tokenize(self.tokenizer, str(record["prompt"]))
            completion_ids = _tokenize(self.tokenizer, str(record["completion"])) + eos
            input_ids = prompt_ids + completion_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids if self.mask_prompt else list(input_ids)
            return input_ids, labels

        if "instruction" in record and "output" in record:
            prompt_ids = _tokenize(self.tokenizer, _alpaca_prompt(record))
            completion_ids = _tokenize(self.tokenizer, str(record["output"])) + eos
            input_ids = prompt_ids + completion_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids if self.mask_prompt else list(input_ids)
            return input_ids, labels

        if "text" in record:
            input_ids = _tokenize(self.tokenizer, str(record["text"])) + eos
            return input_ids, list(input_ids)

        raise ValueError("SFT record must contain messages, prompt/completion, instruction/output, text, or input_ids")


@dataclass
class SFTDataCollator:
    pad_token_id: int
    label_pad_token_id: int = IGNORE_INDEX
    max_length: int | None = None
    pad_to_multiple_of: int | None = None

    def __call__(self, features: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        max_len = max(int(feature["input_ids"].numel()) for feature in features)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for feature in features:
            input_ids = feature["input_ids"][:max_len].long()
            labels = feature["labels"][:max_len].long()
            pad_len = max_len - int(input_ids.numel())
            if pad_len > 0:
                input_ids = torch.cat(
                    (input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)),
                    dim=0,
                )
                labels = torch.cat(
                    (labels, torch.full((pad_len,), self.label_pad_token_id, dtype=torch.long)),
                    dim=0,
                )
            input_rows.append(input_ids)
            label_rows.append(labels)

        return {
            "input_ids": torch.stack(input_rows, dim=0),
            "labels": torch.stack(label_rows, dim=0),
        }
