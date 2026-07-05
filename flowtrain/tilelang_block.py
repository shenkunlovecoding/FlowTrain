from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .rwkv7 import RWKV7Block


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
    if config.backend != "tilelang":
        return f"backend is {config.backend!r}"
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
    if x.shape[1] % config.chunk_len != 0:
        return f"timesteps ({x.shape[1]}) must be divisible by chunk_len ({config.chunk_len})"
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        return "tilelang_block v1 supports single-GPU training only"
    try:
        _get_runtime(
            config.n_embd,
            config.head_size,
            config.dim_ffn,
            config.chunk_len,
            x.dtype,
            torch.cuda.get_device_capability(x.device),
        )
        _tilelang_gemm_ops()
        _tilelang_recurrence_ops()
        _tilelang_time_mix_post_ops()
    except Exception as exc:  # pragma: no cover - depends on local TileLang install.
        return f"TileLang runtime unavailable: {exc}"
    return None


def _warn_fallback(reason: str) -> None:
    if reason in _WARNED_FALLBACKS:
        return
    _WARNED_FALLBACKS.add(reason)
    warnings.warn(f"RWKV7 tilelang_block fallback: {reason}", RuntimeWarning, stacklevel=3)


@lru_cache(maxsize=1)
def _tilelang_gemm_ops():
    try:
        from .tilelang_gemm import tilelang_linear, tilelang_matmul
    except ImportError:
        from tilelang_gemm import tilelang_linear, tilelang_matmul
    return tilelang_linear, tilelang_matmul


@lru_cache(maxsize=1)
def _tilelang_recurrence_ops():
    try:
        from .tilelang_recurrence import can_use_rwkv7_recurrence_tilelang, rwkv7_recurrence_tilelang
    except ImportError:
        from tilelang_recurrence import can_use_rwkv7_recurrence_tilelang, rwkv7_recurrence_tilelang
    return can_use_rwkv7_recurrence_tilelang, rwkv7_recurrence_tilelang


@lru_cache(maxsize=1)
def _tilelang_time_mix_post_ops():
    try:
        from .tilelang_time_mix import (
            can_use_time_mix_post_tilelang,
            can_use_time_mix_shift_tilelang,
            time_mix_post_tilelang,
            time_mix_shift_tilelang,
        )
    except ImportError:
        from tilelang_time_mix import (
            can_use_time_mix_post_tilelang,
            can_use_time_mix_shift_tilelang,
            time_mix_post_tilelang,
            time_mix_shift_tilelang,
        )
    return (
        can_use_time_mix_post_tilelang,
        time_mix_post_tilelang,
        can_use_time_mix_shift_tilelang,
        time_mix_shift_tilelang,
    )


@lru_cache(maxsize=1)
def _rwkv7_recurrence_ref():
    try:
        from .rwkv7 import rwkv7_recurrence
    except ImportError:
        from rwkv7 import rwkv7_recurrence
    return rwkv7_recurrence


def _matmul3(x: torch.Tensor, weight: torch.Tensor, matmul) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    out = matmul(x.reshape(-1, x.shape[-1]), weight)
    return out.view(*orig_shape, weight.shape[1])


def _time_mix_tilelang(att, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tilelang_linear, tilelang_matmul = _tilelang_gemm_ops()
    bsz, timesteps, channels = x.size()
    h = att.n_head
    can_use_post, time_mix_post_tilelang, can_use_shift, time_mix_shift_tilelang = _tilelang_time_mix_post_ops()
    if (
        os.environ.get("FLOWTRAIN_DISABLE_TIMEMIX_SHIFT_FUSION") != "1"
        and can_use_shift(x, att.x_r, att.x_w, att.x_k, att.x_v, att.x_a, att.x_g)
    ):
        xr, xw, xk, xv, xa, xg = time_mix_shift_tilelang(x, att.x_r, att.x_w, att.x_k, att.x_v, att.x_a, att.x_g)
    else:
        xx = att.time_shift(x) - x

        xr = x + xx * att.x_r
        xw = x + xx * att.x_w
        xk = x + xx * att.x_k
        xv = x + xx * att.x_v
        xa = x + xx * att.x_a
        xg = x + xx * att.x_g

    r = tilelang_linear(xr, att.receptance.weight)
    raw_w = att.w0 + _matmul3(torch.tanh(_matmul3(xw, att.w1, tilelang_matmul)), att.w2, tilelang_matmul)
    k = tilelang_linear(xk, att.key.weight)
    v = tilelang_linear(xv, att.value.weight)
    if att.layer_id == 0:
        v_first = v
    else:
        v_mix = torch.sigmoid(att.v0 + _matmul3(_matmul3(xv, att.v1, tilelang_matmul), att.v2, tilelang_matmul))
        v = v + (v_first - v) * v_mix
    a = torch.sigmoid(att.a0 + _matmul3(_matmul3(xa, att.a1, tilelang_matmul), att.a2, tilelang_matmul))
    g = _matmul3(torch.sigmoid(_matmul3(xg, att.g1, tilelang_matmul)), att.g2, tilelang_matmul)

    kk = k * att.k_k
    kk = F.normalize(kk.view(bsz, timesteps, h, -1), dim=-1, p=2.0).view(bsz, timesteps, channels)
    k = k * (1 + (a - 1) * att.k_a)
    neg_kk = -kk
    kk_a = kk * a

    can_use_recurrence, rwkv7_recurrence_tilelang = _tilelang_recurrence_ops()
    if can_use_recurrence(r, raw_w, k, v, neg_kk, kk_a, att.head_size, att.config.chunk_len):
        x = rwkv7_recurrence_tilelang(
            r,
            raw_w.to(dtype=r.dtype),
            k,
            v,
            neg_kk,
            kk_a,
            att.head_size,
            chunk_len=att.config.chunk_len,
        )
    else:
        rwkv7_recurrence = _rwkv7_recurrence_ref()
        w = -F.softplus(-raw_w) - 0.5
        x = rwkv7_recurrence(r, w, k, v, neg_kk, kk_a, att.head_size)

    if (
        not torch.is_grad_enabled()
        and os.environ.get("FLOWTRAIN_DISABLE_TIMEMIX_POST_FUSION") != "1"
        and can_use_post(x, r, k, v, g, att.r_k, att.ln_x.weight, att.ln_x.bias, att.head_size)
    ):
        x = time_mix_post_tilelang(x, r, k, v, g, att.r_k, att.ln_x.weight, att.ln_x.bias, att.head_size, att.ln_x.eps)
    else:
        x = att.ln_x(x.view(bsz * timesteps, channels)).view(bsz, timesteps, channels)
        x = x + (
            (r.view(bsz, timesteps, h, -1).float() * k.view(bsz, timesteps, h, -1).float() * att.r_k)
            .sum(dim=-1, keepdim=True)
            .to(dtype=r.dtype)
            * v.view(bsz, timesteps, h, -1)
        ).view(bsz, timesteps, channels)
        x = x * g
    return tilelang_linear(x, att.output.weight), v_first


def _channel_mix_tilelang(ffn, x: torch.Tensor) -> torch.Tensor:
    tilelang_linear, _ = _tilelang_gemm_ops()
    xx = ffn.time_shift(x) - x
    k = x + xx * ffn.x_k
    return tilelang_linear(torch.relu(tilelang_linear(k, ffn.key.weight)) ** 2, ffn.value.weight)


def _forward_tilelang_gemm(block: RWKV7Block, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if block.layer_id == 0:
        x = block.ln0(x)
    x_att, v_first = _time_mix_tilelang(block.att, block.ln1(x), v_first)
    x = x + x_att
    x = x + _channel_mix_tilelang(block.ffn, block.ln2(x))
    return x, v_first


class _RWKV7TileLangBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block: RWKV7Block, x: torch.Tensor, v_first: torch.Tensor, *params: torch.Tensor):
        ctx.block = block
        ctx.save_for_backward(x, v_first, *params)
        with torch.no_grad():
            return _forward_tilelang_gemm(block, x, v_first)

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
            x_out, v_first_out = _forward_tilelang_gemm(block, x_req, v_first_req)
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
        return block._forward_torch_ref(x, v_first)
    params = tuple(block.parameters())
    return _RWKV7TileLangBlock.apply(block, x, v_first, *params)
