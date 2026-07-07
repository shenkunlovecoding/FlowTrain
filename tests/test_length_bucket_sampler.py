from __future__ import annotations

import pytest
import torch

from flowtrain.sft_data import (
    LengthBucketBatchSampler,
    SFTDataCollator,
    SFTJsonlDataset,
    _round_seq_len_to_bucket,
    compute_sft_lengths,
)


class _FakeDataset:
    """Minimal dataset: the sampler only needs ``len(dataset)``; it never calls
    ``__getitem__`` (it emits index lists; the DataLoader/collator consume them)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


def _features(lengths: list[int]) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.ones(length, dtype=torch.long),
            "labels": torch.ones(length, dtype=torch.long),
        }
        for length in lengths
    ]


# --- _round_seq_len_to_bucket ------------------------------------------------


def test_round_seq_len_to_bucket_round_up():
    assert _round_seq_len_to_bucket(100, (128, 256), 512) == 128
    assert _round_seq_len_to_bucket(128, (128, 256), 512) == 128  # exact
    assert _round_seq_len_to_bucket(129, (128, 256), 512) == 256
    # Leak A regression: length in (max_bucket, max_length] -> max_length (implicit final bucket)
    assert _round_seq_len_to_bucket(300, (128, 256), 512) == 512
    assert _round_seq_len_to_bucket(257, (128, 256, 512), 512) == 512


def test_round_seq_len_to_bucket_effective_set_edge_cases():
    # max_length is already a bucket value -> dedup (buckets kept increasing)
    assert _round_seq_len_to_bucket(100, (64, 100, 256), 100) == 100
    # max_length smaller than some buckets -> those buckets dropped from effective set
    assert _round_seq_len_to_bucket(100, (64, 128), 100) == 100
    # only max_length is set (no explicit buckets) -> everything rounds to max_length
    assert _round_seq_len_to_bucket(100, None, 128) == 128
    # no buckets, no max_length -> identity (legacy / un-bucketed path)
    assert _round_seq_len_to_bucket(100, None, None) == 100
    # length past the largest effective bucket is returned unchanged; upstream
    # truncation (dataset / compute_sft_lengths / the collator's first clamp)
    # keeps lengths <= max_length so this branch is unreachable in-pipeline.
    assert _round_seq_len_to_bucket(700, (128, 256), 512) == 700


def test_round_seq_len_to_bucket_validation():
    with pytest.raises(ValueError, match="positive"):
        _round_seq_len_to_bucket(10, (0, 128), 512)
    with pytest.raises(ValueError, match="strictly increasing"):
        _round_seq_len_to_bucket(10, (256, 128), 512)


# --- collator Leak A/B regression -------------------------------------------


def test_collator_bucket_leaks_clamp_to_max_length():
    # Leak A: today a 300-token sample with buckets (128,256) yielded (1, 300)
    # because _pad_length_to_bucket returned the length unchanged past the last
    # bucket. Now max_length is the implicit final bucket -> (1, 512).
    out = SFTDataCollator(pad_token_id=0, max_length=512, pad_to_buckets=(128, 256))(
        _features([300])
    )
    assert out["input_ids"].shape == (1, 512)

    # Leak B: 257 used to clamp-after-bucket to (1, 257); now -> (1, 512).
    out = SFTDataCollator(pad_token_id=0, max_length=512, pad_to_buckets=(128, 256, 512))(
        _features([257])
    )
    assert out["input_ids"].shape == (1, 512)


def test_collator_bucket_legacy_without_max_length_unchanged():
    # With max_length=None the raw buckets are used as-is (no implicit final
    # bucket), so a length past the last bucket stays unrounded. This preserves
    # the pre-change contract for callers that never set max_length.
    out = SFTDataCollator(pad_token_id=0, max_length=None, pad_to_buckets=(128, 256))(
        _features([300])
    )
    assert out["input_ids"].shape == (1, 300)


def test_collator_existing_bucket_assertions_still_hold():
    pad = 0
    assert SFTDataCollator(pad_token_id=pad, max_length=512, pad_to_buckets=(128, 256, 512))(
        _features([128])
    )["input_ids"].shape == (1, 128)
    assert SFTDataCollator(pad_token_id=pad, max_length=512, pad_to_buckets=(128, 256, 512))(
        _features([129])
    )["input_ids"].shape == (1, 256)
    with pytest.raises(ValueError, match="strictly increasing"):
        SFTDataCollator(pad_token_id=pad, pad_to_buckets=(256, 128))(_features([12]))


# --- LengthBucketBatchSampler ------------------------------------------------

LENGTHS = [10, 20, 30, 130, 140, 150, 250, 260, 270]
BUCKETS = (64, 128, 256)
MAX_LENGTH = 512


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        # one token id per character so length == len(text)
        return [ord(c) % 1000 for c in text]


def test_compute_sft_lengths_matches_getitem(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"prompt":"abc","completion":"defg"}\n'
        '{"prompt":"hello","completion":"world"}\n',
        encoding="utf-8",
    )
    dataset = SFTJsonlDataset(path, _DummyTokenizer(), max_length=16, mask_prompt=True)
    lengths = compute_sft_lengths(dataset)
    assert len(lengths) == len(dataset)
    # lengths must equal what __getitem__ actually produces (prompt + completion + eos)
    for index, length in enumerate(lengths):
        sample = dataset[index]
        assert length == sample["input_ids"].numel()


def _bucket_of(index: int) -> int:
    return _round_seq_len_to_bucket(LENGTHS[index], BUCKETS, MAX_LENGTH)


def test_sampler_covers_every_index_keep_mode():
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, drop_last=False
    )
    seen: list[int] = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(len(LENGTHS)))
    # 3 samples in bucket 64 -> ceil(3/2)=2 batches; 6 in bucket 256 -> 3 batches
    assert len(sampler) == 5


def test_sampler_drop_last_drops_short_tails():
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds,
        LENGTHS,
        batch_size=2,
        buckets=BUCKETS,
        max_length=MAX_LENGTH,
        shuffle=False,
        drop_last=True,
    )
    batches = list(sampler)
    assert all(len(batch) == 2 for batch in batches)
    # bucket 64 (3 samples) yields 1 full batch; bucket 256 (6 samples) yields 3
    assert len(batches) == 4
    assert len(sampler) == 4
    # shuffle=False keeps bucket 64 in original order [0, 1, 2]; the 3rd sample
    # (index 2, length 30) is the dropped tail.
    seen = {i for batch in batches for i in batch}
    assert seen == set(range(len(LENGTHS))) - {2}
    assert LENGTHS[2] == 30 and 30 not in {LENGTHS[i] for i in seen}


def test_sampler_batches_are_bucket_homogeneous():
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, drop_last=False
    )
    for batch in sampler:
        buckets = {_bucket_of(i) for i in batch}
        assert len(buckets) == 1


def test_sampler_seeded_reproducible():
    ds = _FakeDataset(len(LENGTHS))
    s1 = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, seed=7
    )
    s2 = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, seed=7
    )
    assert list(s1) == list(s2)


def test_sampler_advances_epoch_per_iteration():
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, seed=7
    )
    assert sampler._epoch == 0
    iter(sampler)
    assert sampler._epoch == 1
    iter(sampler)
    assert sampler._epoch == 2


def test_sampler_no_shuffle_is_deterministic():
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, shuffle=False
    )
    batches = list(sampler)
    # groups emitted in ascending bucket order; indices in original order within a group
    assert batches[0] == [0, 1]  # bucket 64, first two
    # reproducible across a fresh sampler
    again = list(
        LengthBucketBatchSampler(
            ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, shuffle=False
        )
    )
    assert batches == again


def test_sampler_validation_errors():
    ds = _FakeDataset(len(LENGTHS))
    with pytest.raises(ValueError, match="entries but dataset has"):
        LengthBucketBatchSampler(ds, LENGTHS[:-1], 2, BUCKETS, max_length=MAX_LENGTH)
    with pytest.raises(ValueError, match="batch_size"):
        LengthBucketBatchSampler(ds, LENGTHS, 0, BUCKETS, max_length=MAX_LENGTH)
    with pytest.raises(ValueError, match="effective bucket"):
        LengthBucketBatchSampler(ds, LENGTHS, 2, None, max_length=None)
    with pytest.raises(ValueError, match="strictly increasing"):
        LengthBucketBatchSampler(ds, LENGTHS, 2, (256, 64), max_length=MAX_LENGTH)


def test_sampler_and_collator_are_consistent():
    # The core invariant, made executable: for every batch the sampler emits,
    # collating its (synthetic) features pads to exactly the bucket the sampler
    # grouped that batch into.
    ds = _FakeDataset(len(LENGTHS))
    sampler = LengthBucketBatchSampler(
        ds, LENGTHS, batch_size=2, buckets=BUCKETS, max_length=MAX_LENGTH, drop_last=False
    )
    collator = SFTDataCollator(pad_token_id=0, max_length=MAX_LENGTH, pad_to_buckets=BUCKETS)
    for batch in sampler:
        features = _features([LENGTHS[i] for i in batch])
        out = collator(features)
        expected_bucket = _bucket_of(batch[0])
        assert out["input_ids"].shape[1] == expected_bucket
        assert all(_bucket_of(i) == expected_bucket for i in batch)
