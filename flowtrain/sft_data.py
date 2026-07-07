from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler


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


def _validate_buckets(buckets: Sequence[int]) -> list[int]:
    bucket_values = [int(bucket) for bucket in buckets]
    if any(bucket <= 0 for bucket in bucket_values):
        raise ValueError("pad_to_buckets values must be positive")
    if bucket_values != sorted(set(bucket_values)):
        raise ValueError("pad_to_buckets values must be strictly increasing")
    return bucket_values


def _pad_length_to_bucket(length: int, buckets: Sequence[int]) -> int:
    bucket_values = _validate_buckets(buckets)
    for bucket in bucket_values:
        if length <= bucket:
            return bucket
    return length


def _effective_buckets(
    buckets: Sequence[int] | None, max_length: int | None
) -> list[int]:
    """Effective bucket set shared by the collator and the length-bucket sampler.

    ``max_length`` is treated as the implicit final bucket so every sample
    length (already truncated to ``max_length`` upstream by the dataset) maps
    onto a finite set of seq_len targets — long-tail samples no longer leak
    past the largest declared bucket as a non-bucket seq_len. Buckets larger
    than ``max_length`` are dropped. With ``max_length is None`` the raw
    buckets are returned unchanged, preserving the legacy behavior where
    lengths past the largest bucket stay unrounded.
    """
    raw = list(buckets) if buckets is not None else []
    kept = [int(b) for b in raw if max_length is None or b <= max_length]
    if max_length is not None:
        kept.append(int(max_length))
    return sorted(set(kept))


def _round_seq_len_to_bucket(
    length: int, buckets: Sequence[int] | None, max_length: int | None
) -> int:
    """Round ``length`` up to the smallest effective bucket >= length.

    Single source of truth shared by :class:`SFTDataCollator` and
    :class:`LengthBucketBatchSampler`. Because both round with the same
    ``(buckets, max_length)``, the collator's within-batch-max rounding always
    lands on the same bucket the sampler assigned to that batch (no metadata
    needs to be threaded between them). With an empty effective set (no
    buckets and no ``max_length``) returns ``length`` unchanged.
    """
    if buckets is not None:
        # Validate the raw user-provided order before _effective_buckets sorts
        # it, so a non-increasing bucket list still raises (mirrors the legacy
        # _pad_length_to_bucket contract relied on by the collator).
        _validate_buckets(buckets)
    effective = _effective_buckets(buckets, max_length)
    if not effective:
        return length
    return _pad_length_to_bucket(length, effective)


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
    pad_to_buckets: Sequence[int] | None = None
    pad_to_max_length: bool = False

    def __call__(self, features: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        if self.pad_to_max_length:
            if self.max_length is None:
                raise ValueError("pad_to_max_length requires max_length")
            max_len = self.max_length
        else:
            max_len = max(int(feature["input_ids"].numel()) for feature in features)
            if self.max_length is not None:
                max_len = min(max_len, self.max_length)
        if self.pad_to_buckets:
            max_len = _round_seq_len_to_bucket(max_len, self.pad_to_buckets, self.max_length)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple
            if self.max_length is not None:
                max_len = min(max_len, self.max_length)

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


def compute_sft_lengths(dataset: SFTJsonlDataset) -> list[int]:
    """Tokenize every record once and return per-sample, post-truncation lengths.

    Mirrors :meth:`SFTJsonlDataset.__getitem__` exactly by reusing
    ``_encode_record`` + :func:`_truncate_pair`, so the lengths agree with what
    the collator sees at train time. Runs once in the main process (before
    DataLoader construction) so a length-bucket sampler can group samples
    without re-tokenizing per epoch or coupling to worker-process state.
    Records that produce fewer than two tokens raise, matching the
    ``__getitem__`` invariant.
    """
    lengths: list[int] = []
    max_length = dataset.max_length
    for record in dataset.records:
        input_ids, _labels = dataset._encode_record(record)
        input_ids, _labels = _truncate_pair(input_ids, _labels, max_length)
        if len(input_ids) < 2:
            raise ValueError("SFT samples must produce at least two tokens")
        lengths.append(len(input_ids))
    return lengths


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Group dataset indices into sequence-length-homogeneous batches.

    Each sample is assigned to a bucket via :func:`_round_seq_len_to_bucket`
    (with ``max_length`` as the implicit final bucket); batches are formed
    within a single bucket so that, after the collator pads to the *same*
    rounding function, every batch's seq_len is exactly that bucket. This keeps
    TileLang kernel shapes static (one specialization per bucket instead of one
    per random shuffle) and minimizes padding, since samples sharing a batch
    have similar length.

    Sampler/collator consistency holds as long as both round with the same
    ``(buckets, max_length)``: if every sample in a batch rounds to bucket k,
    so does the within-batch max. The bucket-batch CLI path builds both from
    the same ``--pad-to-buckets`` / ``--max-length`` to guarantee this.
    """

    def __init__(
        self,
        dataset: Dataset,
        lengths: Sequence[int],
        batch_size: int,
        buckets: Sequence[int] | None,
        *,
        max_length: int | None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if len(lengths) != len(dataset):
            raise ValueError(
                f"lengths has {len(lengths)} entries but dataset has {len(dataset)} samples"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        effective = _effective_buckets(buckets, max_length)
        if not effective:
            raise ValueError(
                "LengthBucketBatchSampler requires at least one effective bucket; "
                "pass buckets or set max_length"
            )
        if buckets is not None:
            _validate_buckets(buckets)
        self.dataset = dataset
        self.batch_size = batch_size
        self.buckets = list(buckets) if buckets is not None else None
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(lengths):
            bucket = _round_seq_len_to_bucket(length, buckets, max_length)
            groups.setdefault(bucket, []).append(index)
        # Deterministic bucket order so __len__ and iteration are stable.
        self._groups: list[tuple[int, list[int]]] = sorted(groups.items())
        self._num_batches = self._compute_num_batches()

    def _compute_num_batches(self) -> int:
        total = 0
        for _bucket, indices in self._groups:
            count = len(indices)
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += -(-count // self.batch_size)  # ceil(count / batch_size)
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        all_batches: list[list[int]] = []
        for _bucket, indices in self._groups:
            bucket_indices = list(indices)
            if self.shuffle:
                rng.shuffle(bucket_indices)
            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self) -> int:
        return self._num_batches
