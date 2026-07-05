from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .rwkv7 import lora_ranks, official_rank
from .trainer import infer_rwkv7_config_from_state


ActivationOffload = Literal["none", "cpu"]
ActivationQuant = Literal["none", "int8"]
ActivationStrategy = Literal["recompute", "store_layer_inputs"]
OptimizerKind = Literal["adamw", "deepspeed_cpu_adam", "qr_muon"]


def _checkpoint_state(raw: dict) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if isinstance(raw, dict):
        return raw
    raise TypeError("RWKV-7 checkpoint must be a state_dict-like mapping")


@dataclass(frozen=True)
class RWKV7Size:
    vocab_size: int
    n_layer: int
    n_embd: int
    dim_ffn: int
    head_size: int = 64
    lora_rank_style: str = "official"


@dataclass(frozen=True)
class RWKV7ParamBreakdown:
    total_params: int
    embedding_params: int
    head_params: int
    norm_params: int
    first_layer_params: int
    regular_layer_params: int


@dataclass(frozen=True)
class BatchSizeEstimate:
    max_batch_size: int
    gpu_limit_batch_size: int
    cpu_limit_batch_size: int | None
    gpu_base_gb: float
    gpu_per_sample_gb: float
    cpu_base_gb: float
    cpu_per_sample_gb: float
    model_params_b: float
    assumptions: tuple[str, ...]



def rwkv7_param_breakdown(size: RWKV7Size) -> RWKV7ParamBreakdown:
    if size.head_size != 64:
        raise ValueError("FlowTrain v1 supports head_size=64 only")
    c = size.n_embd
    ffn = size.dim_ffn
    d_decay, d_aaa, d_mv, d_gate = lora_ranks(c, size.lora_rank_style)

    time_mix = (
        4 * c * c
        + 2 * c * (d_decay + d_aaa + d_mv + d_gate)
        + 14 * c
    )
    channel_mix = 2 * c * ffn + c
    regular_layer = time_mix + channel_mix + 4 * c
    first_layer = regular_layer + 2 * c
    embedding = size.vocab_size * c
    head = size.vocab_size * c
    norm = 2 * c
    total = embedding + head + norm + first_layer + max(0, size.n_layer - 1) * regular_layer
    return RWKV7ParamBreakdown(
        total_params=total,
        embedding_params=embedding,
        head_params=head,
        norm_params=norm,
        first_layer_params=first_layer,
        regular_layer_params=regular_layer,
    )


def _qr_muon_param_and_basis_entries(size: RWKV7Size) -> tuple[int, int]:
    c = size.n_embd
    ffn = size.dim_ffn
    ranks = lora_ranks(c, size.lora_rank_style)
    rank_sum = sum(ranks)
    rank_sq_sum = sum(rank * rank for rank in ranks)

    muon_params_per_layer = 4 * c * c + 2 * c * rank_sum + 2 * c * ffn
    basis_entries_per_layer = 4 * c * c + 2 * rank_sq_sum + 2 * min(c, ffn) ** 2
    return size.n_layer * muon_params_per_layer, size.n_layer * basis_entries_per_layer


def infer_size_from_checkpoint(
    checkpoint: str | Path,
    *,
    lora_rank_style: str = "official",
    head_size: int = 64,
) -> RWKV7Size:
    raw = torch.load(checkpoint, map_location="cpu")
    state = _checkpoint_state(raw)
    config = infer_rwkv7_config_from_state(state, ctx_len=1, backend="torch_ref")
    assert config.dim_ffn is not None
    return RWKV7Size(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        dim_ffn=config.dim_ffn,
        head_size=head_size,
        lora_rank_style=lora_rank_style,
    )


def detect_cuda_memory_gb(device: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device).total_memory / 1024**3


def detect_system_memory_gb() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
    return pages * page_size / 1024**3


def estimate_rwkv7_batch_size(
    size: RWKV7Size,
    *,
    seq_len: int,
    gpu_gb: float,
    cpu_gb: float | None = None,
    activation_offload: ActivationOffload = "cpu",
    activation_quant: ActivationQuant = "none",
    activation_strategy: ActivationStrategy = "recompute",
    checkpoint_interval: int = 4,
    gpu_utilization: float = 0.90,
    cpu_utilization: float = 0.85,
    reserve_gpu_gb: float = 4.0,
    reserve_cpu_gb: float = 8.0,
    param_dtype_bytes: int = 4,
    gpu_dtype_bytes: int = 2,
    optimizer: OptimizerKind = "adamw",
    optimizer_state_bytes_per_param: int = 8,
    logit_chunk_size: int = 128,
    hidden_gpu_copies: int = 10,
) -> BatchSizeEstimate:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    if activation_offload not in ("none", "cpu"):
        raise ValueError("activation_offload must be 'none' or 'cpu'")
    if activation_quant not in ("none", "int8"):
        raise ValueError("activation_quant must be 'none' or 'int8'")
    if activation_quant == "int8" and activation_offload != "cpu":
        raise ValueError("activation_quant='int8' requires activation_offload='cpu'")
    if activation_strategy not in ("recompute", "store_layer_inputs"):
        raise ValueError("activation_strategy must be 'recompute' or 'store_layer_inputs'")
    if optimizer not in ("adamw", "deepspeed_cpu_adam", "qr_muon"):
        raise ValueError("optimizer must be 'adamw', 'deepspeed_cpu_adam', or 'qr_muon'")

    breakdown = rwkv7_param_breakdown(size)
    max_layer = max(breakdown.first_layer_params, breakdown.regular_layer_params)

    # Current FlowTrain keeps two GPU layer modules, a layer transfer tensor,
    # one forward embedding clone, one replay embedding clone, and the head.
    gpu_base_bytes = (
        (3 * max_layer + 2 * breakdown.embedding_params + breakdown.head_params + breakdown.norm_params)
        * gpu_dtype_bytes
    )

    token_hidden_bytes = seq_len * size.n_embd * gpu_dtype_bytes
    logit_tokens = seq_len - 1 if logit_chunk_size <= 0 else min(seq_len - 1, logit_chunk_size)
    logit_bytes = logit_tokens * size.vocab_size * 4
    token_id_bytes = 2 * seq_len * 8
    gpu_per_sample_bytes = hidden_gpu_copies * token_hidden_bytes + logit_bytes + token_id_bytes

    if activation_offload == "cpu":
        if activation_strategy == "store_layer_inputs":
            n_checkpoints = size.n_layer + 1
        else:
            n_checkpoints = math.ceil(size.n_layer / checkpoint_interval) + 1
        if activation_quant == "int8":
            packed_tensor_bytes = seq_len * (size.n_embd + 2)
        else:
            packed_tensor_bytes = token_hidden_bytes
        cpu_activation_per_sample_bytes = (n_checkpoints + 1) * packed_tensor_bytes
    else:
        cpu_activation_per_sample_bytes = 0
        if activation_strategy == "store_layer_inputs":
            n_checkpoints = size.n_layer + 1
        else:
            n_checkpoints = math.ceil(size.n_layer / checkpoint_interval) + 1
        gpu_per_sample_bytes += (n_checkpoints + 1) * token_hidden_bytes

    if optimizer == "qr_muon":
        muon_params, muon_basis_entries = _qr_muon_param_and_basis_entries(size)
        adamw_params = max(0, breakdown.total_params - muon_params)
        optimizer_state_bytes = (
            adamw_params * optimizer_state_bytes_per_param
            + muon_params * param_dtype_bytes
            + muon_basis_entries * param_dtype_bytes
        )
        optimizer_assumption = "QR Muon states are used for RWKV block projection/LoRA matrices; other params use AdamW"
    else:
        optimizer_state_bytes = breakdown.total_params * optimizer_state_bytes_per_param
        if optimizer == "deepspeed_cpu_adam":
            optimizer_assumption = "CPU params, CPU grads, and DeepSpeed CPUAdam states stay on host memory"
        else:
            optimizer_assumption = "CPU params, CPU grads, and AdamW states stay on host memory"

    cpu_base_bytes = (
        breakdown.total_params * (param_dtype_bytes + param_dtype_bytes)
        + optimizer_state_bytes
        + (breakdown.first_layer_params + max(0, size.n_layer - 1) * breakdown.regular_layer_params) * gpu_dtype_bytes
    )

    gpu_budget_bytes = max(0.0, (gpu_gb * gpu_utilization - reserve_gpu_gb) * 1024**3)
    gpu_limit = math.floor((gpu_budget_bytes - gpu_base_bytes) / max(1, gpu_per_sample_bytes))
    gpu_limit = max(0, gpu_limit)

    cpu_limit: int | None = None
    if cpu_gb is not None:
        cpu_budget_bytes = max(0.0, (cpu_gb * cpu_utilization - reserve_cpu_gb) * 1024**3)
        if cpu_activation_per_sample_bytes > 0:
            cpu_limit = max(0, math.floor((cpu_budget_bytes - cpu_base_bytes) / cpu_activation_per_sample_bytes))
        else:
            cpu_limit = None if cpu_base_bytes <= cpu_budget_bytes else 0

    limits = [gpu_limit]
    if cpu_limit is not None:
        limits.append(cpu_limit)

    assumptions = (
        "single CUDA GPU, bf16 GPU weights and activations",
        optimizer_assumption,
        "current trainer computes next-token logits in chunks; pass logit_chunk_size=0 to model full-sequence logits",
        f"activation checkpoint count is {n_checkpoints}",
        f"hidden_gpu_copies={hidden_gpu_copies} is a conservative peak multiplier",
    )
    return BatchSizeEstimate(
        max_batch_size=min(limits),
        gpu_limit_batch_size=gpu_limit,
        cpu_limit_batch_size=cpu_limit,
        gpu_base_gb=gpu_base_bytes / 1024**3,
        gpu_per_sample_gb=gpu_per_sample_bytes / 1024**3,
        cpu_base_gb=cpu_base_bytes / 1024**3,
        cpu_per_sample_gb=cpu_activation_per_sample_bytes / 1024**3,
        model_params_b=breakdown.total_params / 1e9,
        assumptions=assumptions,
    )
