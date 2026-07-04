from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .rwkv7_core import RWKV7Block


_WARNED_FALLBACKS: set[str] = set()


@dataclass(frozen=True)
class TileLangBlockRuntime:
    hidden: int
    head_size: int
    dim_ffn: int
    chunk_len: int
    dtype: torch.dtype
    capability: tuple[int, int]


@lru_cache(maxsize=32)
def _get_runtime(
    hidden: int,
    head_size: int,
    dim_ffn: int,
    chunk_len: int,
    dtype: torch.dtype,
    capability: tuple[int, int],
) -> TileLangBlockRuntime:
    import tilelang  # noqa: F401

    return TileLangBlockRuntime(
        hidden=hidden,
        head_size=head_size,
        dim_ffn=dim_ffn,
        chunk_len=chunk_len,
        dtype=dtype,
        capability=capability,
    )


def _fallback_reason(block: RWKV7Block, x: torch.Tensor, v_first: torch.Tensor) -> str | None:
    config = block.config
    if not config.use_cuda_kernel:
        return "use_cuda_kernel is disabled"
    if x.device != v_first.device:
        return "x and v_first are on different devices"
    if not x.is_cuda:
        return "input is not a CUDA tensor"
    if x.dtype != torch.bfloat16 or v_first.dtype != torch.bfloat16:
        return "tilelang_block requires bf16 x and v_first"
    if config.head_size != 64:
        return f"tilelang_block v1 supports head_size=64, got {config.head_size}"
    if x.dim() != 3:
        return "tilelang_block expects [B, T, C] hidden states"
    if x.shape != v_first.shape:
        return "x and v_first must share shape"
    if x.shape[-1] != config.n_embd:
        return f"hidden size mismatch: input has {x.shape[-1]}, config has {config.n_embd}"
    if x.shape[1] % config.cuda_chunk_len != 0:
        return f"timesteps ({x.shape[1]}) must be divisible by cuda_chunk_len ({config.cuda_chunk_len})"
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        return "tilelang_block v1 supports single-GPU training only"
    try:
        _get_runtime(
            config.n_embd,
            config.head_size,
            config.dim_ffn,
            config.cuda_chunk_len,
            x.dtype,
            torch.cuda.get_device_capability(x.device),
        )
    except Exception as exc:  # pragma: no cover - depends on local TileLang install.
        return f"TileLang runtime unavailable: {exc}"
    return None


def _warn_fallback(reason: str) -> None:
    if reason in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(reason)
    warnings.warn(f"RWKV7 tilelang_block fallback: {reason}", RuntimeWarning, stacklevel=3)


class _RWKV7TileLangBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block: RWKV7Block, x: torch.Tensor, v_first: torch.Tensor, *params: torch.Tensor):
        ctx.block = block
        ctx.save_for_backward(x, v_first, *params)
        with torch.no_grad():
            return block._forward_standard(x, v_first)

    @staticmethod
    def backward(ctx, grad_x_out: torch.Tensor, grad_v_first_out: torch.Tensor):
        saved = ctx.saved_tensors
        x, v_first = saved[:2]
        params = saved[2:]
        block = ctx.block

        x_req = x.detach().requires_grad_(True)
        v_first_req = v_first.detach().requires_grad_(True)
        targets: list[torch.Tensor] = [x_req, v_first_req]
        param_target_indices: list[int | None] = []
        for param in params:
            if param.requires_grad:
                param_target_indices.append(len(targets))
                targets.append(param)
            else:
                param_target_indices.append(None)

        with torch.enable_grad():
            x_out, v_first_out = block._forward_standard(x_req, v_first_req)
        grads = torch.autograd.grad(
            (x_out, v_first_out),
            tuple(targets),
            (grad_x_out, grad_v_first_out),
            allow_unused=True,
        )

        dx = grads[0]
        dv_first = grads[1]
        param_grads: list[torch.Tensor | None] = []
        for param, index in zip(params, param_target_indices, strict=True):
            if index is None:
                param_grads.append(None)
            else:
                param_grads.append(grads[index])
        return (None, dx, dv_first, *param_grads)


def rwkv7_block_tilelang(block: RWKV7Block, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-aware RWKV7 block backend.

    This v1 keeps only the block inputs and parameter references on the autograd
    tape, then recomputes TimeMix and ChannelMix during backward. TileLang is
    currently used as the capability gate and specialization cache; GEMM and the
    existing recurrence backend are reused during recomputation.
    """

    reason = _fallback_reason(block, x, v_first)
    if reason is not None:
        _warn_fallback(reason)
        return block._forward_standard(x, v_first)
    params = tuple(block.parameters())
    return _RWKV7TileLangBlock.apply(block, x, v_first, *params)
