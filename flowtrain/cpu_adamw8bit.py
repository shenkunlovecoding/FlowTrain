from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def adamw_step_(
    master: torch.Tensor,
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    use_cpp: bool = True,
) -> bool:
    """Run the fused fp32-state AdamW update if the optional C++ extension is available."""

    if not use_cpp:
        return False
    if master.numel() < 4096:
        return False
    if not (
        master.is_cpu
        and param.is_cpu
        and grad.is_cpu
        and exp_avg.is_cpu
        and exp_avg_sq.is_cpu
    ):
        return False
    if not (
        master.is_contiguous()
        and param.is_contiguous()
        and grad.is_contiguous()
        and exp_avg.is_contiguous()
        and exp_avg_sq.is_contiguous()
    ):
        return False

    extension = _load_extension()
    if extension is None:
        return False
    extension.adamw_step(
        master,
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        int(step),
    )
    return True


def adamw8bit_step_(
    master: torch.Tensor,
    param: torch.Tensor,
    grad: torch.Tensor,
    q_exp_avg: torch.Tensor,
    exp_avg_scale: torch.Tensor,
    q_exp_avg_sq: torch.Tensor,
    exp_avg_sq_scale: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    block_size: int,
    use_cpp: bool = True,
) -> bool:
    """Run the fused int8 AdamW update if the optional C++ extension is available."""

    if not use_cpp:
        return False
    if not (
        master.is_cpu
        and param.is_cpu
        and grad.is_cpu
        and q_exp_avg.is_cpu
        and exp_avg_scale.is_cpu
        and q_exp_avg_sq.is_cpu
        and exp_avg_sq_scale.is_cpu
    ):
        return False
    if not (
        master.is_contiguous()
        and param.is_contiguous()
        and grad.is_contiguous()
        and q_exp_avg.is_contiguous()
        and exp_avg_scale.is_contiguous()
        and q_exp_avg_sq.is_contiguous()
        and exp_avg_sq_scale.is_contiguous()
    ):
        return False

    extension = _load_extension()
    if extension is None:
        return False
    extension.adamw8bit_step(
        master,
        param,
        grad,
        q_exp_avg,
        exp_avg_scale,
        q_exp_avg_sq,
        exp_avg_sq_scale,
        float(lr),
        float(beta1),
        float(beta2),
        float(eps),
        float(weight_decay),
        int(step),
        int(block_size),
    )
    return True


@lru_cache(maxsize=1)
def _load_extension():
    if os.environ.get("FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP") == "1":
        return None
    source = Path(__file__).resolve().parent / "csrc" / "cpu_adamw8bit.cpp"
    try:
        return load(
            name="flowtrain_cpu_adamw8bit",
            sources=[str(source)],
            extra_cflags=["-O3", "-fopenmp"],
            extra_ldflags=["-fopenmp"],
            verbose=os.environ.get("FLOWTRAIN_CPU_ADAMW8BIT_VERBOSE") == "1",
        )
    except Exception as exc:  # pragma: no cover - depends on local compiler/toolchain.
        warnings.warn(f"FlowTrain CPU AdamW8bit C++ extension unavailable, using Python fallback: {exc}")
        return None
