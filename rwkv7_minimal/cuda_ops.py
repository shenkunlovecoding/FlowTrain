from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parents[1]
_CUDA_DIR = _ROOT / "cuda"


def _variant_sources(variant: str) -> tuple[Path, Path]:
    if variant == "h100":
        return _CUDA_DIR / "rwkv7_clampw_v3.cpp", _CUDA_DIR / "rwkv7_clampw_v3_for_h100.cu"
    if variant == "base":
        return _CUDA_DIR / "rwkv7_clampw.cpp", _CUDA_DIR / "rwkv7_clampw.cu"
    if variant == "head128":
        return _CUDA_DIR / "rwkv7_clampw128_v2.cpp", _CUDA_DIR / "rwkv7_clampw128_v2.cu"
    raise ValueError(f"unknown RWKV7 CUDA variant: {variant}")


def _variant_namespace(variant: str) -> str:
    if variant == "h100":
        return "rwkv7_clampw_v3"
    if variant == "base":
        return "rwkv7_clampw"
    if variant == "head128":
        return "rwkv7_clampw128_v2"
    raise ValueError(f"unknown RWKV7 CUDA variant: {variant}")


@lru_cache(maxsize=16)
def _load_clampw_extension(head_size: int, chunk_len: int, variant: str):
    if head_size == 128 and variant == "h100":
        variant = "head128"
    cpp, cu = _variant_sources(variant)
    name = f"rwkv7_clampw_{variant}_n{head_size}_c{chunk_len}"
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        f"-D_N_={head_size}",
        f"-D_CHUNK_LEN_={chunk_len}",
    ]
    extra_cflags = ["-O3", f"-D_N_={head_size}", f"-D_CHUNK_LEN_={chunk_len}"]
    load(
        name=name,
        sources=[str(cpp), str(cu)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=os.environ.get("RWKV7_CUDA_VERBOSE", "0") == "1",
        is_python_module=False,
    )
    return getattr(torch.ops, _variant_namespace(variant))


def _check_inputs(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
    chunk_len: int,
) -> None:
    tensors = (r, raw_w, k, v, a, b)
    if not all(t.is_cuda for t in tensors):
        raise RuntimeError("RWKV7 CUDA recurrence requires CUDA tensors")
    if not all(t.dtype == torch.bfloat16 for t in tensors):
        raise RuntimeError("RWKV7 CUDA recurrence currently requires bf16 tensors")
    if not all(t.shape == r.shape for t in tensors):
        raise RuntimeError("RWKV7 CUDA recurrence inputs must share [B, T, C] shape")
    if r.dim() != 3:
        raise RuntimeError("RWKV7 CUDA recurrence inputs must be [B, T, C]")
    if r.shape[2] % head_size != 0:
        raise RuntimeError(f"channels ({r.shape[2]}) must be divisible by head_size ({head_size})")
    if r.shape[1] % chunk_len != 0:
        raise RuntimeError(f"timesteps ({r.shape[1]}) must be divisible by chunk_len ({chunk_len})")
    if head_size not in (32, 64, 128):
        raise RuntimeError(f"unsupported head_size for bundled kernels: {head_size}")


class _RWKV7ClampW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, raw_w, k, v, a, b, head_size: int, chunk_len: int, variant: str):
        _check_inputs(r, raw_w, k, v, a, b, head_size, chunk_len)
        ext = _load_clampw_extension(head_size, chunk_len, variant)

        bsz, timesteps, channels = r.shape
        n_head = channels // head_size
        shape4 = (bsz, timesteps, n_head, head_size)

        r4 = r.contiguous().view(shape4)
        w4 = raw_w.contiguous().view(shape4)
        k4 = k.contiguous().view(shape4)
        v4 = v.contiguous().view(shape4)
        a4 = a.contiguous().view(shape4)
        b4 = b.contiguous().view(shape4)

        y4 = torch.empty_like(r4)
        state = torch.empty(
            (bsz, n_head, timesteps // chunk_len, head_size, head_size),
            device=r.device,
            dtype=torch.float32,
        )
        sa4 = torch.empty_like(r4, dtype=torch.float32)

        ext.forward(r4, w4, k4, v4, a4, b4, y4, state, sa4)
        ctx.save_for_backward(r4, w4, k4, v4, a4, b4, state, sa4)
        ctx.head_size = head_size
        ctx.chunk_len = chunk_len
        ctx.variant = variant
        return y4.view_as(r)

    @staticmethod
    def backward(ctx, grad_y):
        r4, w4, k4, v4, a4, b4, state, sa4 = ctx.saved_tensors
        ext = _load_clampw_extension(ctx.head_size, ctx.chunk_len, ctx.variant)
        dy4 = grad_y.contiguous().view_as(r4)

        dr = torch.empty_like(r4)
        dw = torch.empty_like(w4)
        dk = torch.empty_like(k4)
        dv = torch.empty_like(v4)
        da = torch.empty_like(a4)
        db = torch.empty_like(b4)
        ext.backward(r4, w4, k4, v4, a4, b4, dy4, state, sa4, dr, dw, dk, dv, da, db)
        return (
            dr.view_as(grad_y),
            dw.view_as(grad_y),
            dk.view_as(grad_y),
            dv.view_as(grad_y),
            da.view_as(grad_y),
            db.view_as(grad_y),
            None,
            None,
            None,
        )


def rwkv7_recurrence_cuda(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
    chunk_len: int = 16,
    variant: str = "h100",
) -> torch.Tensor:
    """Fused RWKV-7 recurrence for raw decay logits.

    raw_w is the pre-softplus value used by the official clampw kernels. The
    pure PyTorch reference takes transformed w, so callers must pass raw_w here.
    """

    return _RWKV7ClampW.apply(r, raw_w, k, v, a, b, head_size, chunk_len, variant)


def can_use_rwkv7_cuda(x: torch.Tensor, head_size: int, chunk_len: int) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.shape[2] % head_size == 0
        and x.shape[1] % chunk_len == 0
        and head_size in (32, 64, 128)
    )
