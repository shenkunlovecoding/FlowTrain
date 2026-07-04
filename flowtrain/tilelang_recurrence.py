from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F


def can_use_rwkv7_recurrence_tilelang(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
    chunk_len: int,
) -> bool:
    return (
        head_size == 64
        and chunk_len > 0
        and r.is_cuda
        and raw_w.is_cuda
        and k.is_cuda
        and v.is_cuda
        and a.is_cuda
        and b.is_cuda
        and r.dtype == torch.bfloat16
        and raw_w.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and r.dim() == 3
        and raw_w.shape == r.shape
        and k.shape == r.shape
        and v.shape == r.shape
        and a.shape == r.shape
        and b.shape == r.shape
        and r.shape[-1] % head_size == 0
        and r.shape[1] % chunk_len == 0
    )


@lru_cache(maxsize=64)
def _get_recurrence_forward_kernel(batch: int, timesteps: int, channels: int, head_size: int):
    import tilelang
    import tilelang.language as T

    n_head = channels // head_size

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
    def _build_kernel():
        @T.prim_func
        def kernel(
            r: T.Tensor((batch, timesteps, channels), T.bfloat16),
            decay: T.Tensor((batch, timesteps, channels), T.bfloat16),
            k: T.Tensor((batch, timesteps, channels), T.bfloat16),
            v: T.Tensor((batch, timesteps, channels), T.bfloat16),
            a: T.Tensor((batch, timesteps, channels), T.bfloat16),
            b: T.Tensor((batch, timesteps, channels), T.bfloat16),
            out: T.Tensor((batch, timesteps, channels), T.bfloat16),
        ):
            with T.Kernel(batch, n_head, threads=256) as (pid_b, pid_h):
                state = T.alloc_fragment((head_size, head_size), T.float32)
                T.clear(state)
                for t in T.serial(timesteps):
                    for i, j in T.Parallel(head_size, head_size):
                        base = pid_h * head_size
                        mix = T.alloc_var(T.float32)
                        mix = T.cast(0, T.float32)
                        for inner in T.serial(head_size):
                            mix += state[i, inner] * T.cast(a[pid_b, t, base + inner], T.float32)
                        state[i, j] = (
                            state[i, j] * T.cast(decay[pid_b, t, base + j], T.float32)
                            + mix * T.cast(b[pid_b, t, base + j], T.float32)
                            + T.cast(v[pid_b, t, base + i], T.float32) * T.cast(k[pid_b, t, base + j], T.float32)
                        )
                    for i in T.Parallel(head_size):
                        base = pid_h * head_size
                        acc = T.alloc_var(T.float32)
                        acc = T.cast(0, T.float32)
                        for j in T.serial(head_size):
                            acc += state[i, j] * T.cast(r[pid_b, t, base + j], T.float32)
                        out[pid_b, t, base + i] = T.cast(acc, T.bfloat16)

        return kernel

    return _build_kernel()


def _torch_ref_from_raw_w(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    try:
        from .rwkv7 import rwkv7_recurrence
    except ImportError:
        from rwkv7 import rwkv7_recurrence

    w = -F.softplus(-raw_w) - 0.5
    return rwkv7_recurrence(r, w, k, v, a, b, head_size)


class _RWKV7RecurrenceTileLang(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        r: torch.Tensor,
        raw_w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        head_size: int,
        chunk_len: int,
    ):
        ctx.head_size = head_size
        ctx.chunk_len = chunk_len
        ctx.save_for_backward(r, raw_w, k, v, a, b)

        decay = torch.exp(-torch.exp((-F.softplus(-raw_w.float()) - 0.5))).to(dtype=r.dtype)
        out = torch.empty_like(r)
        kernel = _get_recurrence_forward_kernel(r.shape[0], r.shape[1], r.shape[2], head_size)
        kernel(
            r.contiguous(),
            decay.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a.contiguous(),
            b.contiguous(),
            out,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        r, raw_w, k, v, a, b = ctx.saved_tensors
        detached = [x.detach().requires_grad_(True) for x in (r, raw_w, k, v, a, b)]
        with torch.enable_grad():
            out = _torch_ref_from_raw_w(*detached, ctx.head_size)
        grads = torch.autograd.grad(out, tuple(detached), grad_out, allow_unused=True)
        return (*grads, None, None)


def rwkv7_recurrence_tilelang(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    head_size: int,
    chunk_len: int,
) -> torch.Tensor:
    """RWKV-7 recurrence with a TileLang forward kernel and recompute backward."""

    if not can_use_rwkv7_recurrence_tilelang(r, raw_w, k, v, a, b, head_size, chunk_len):
        return _torch_ref_from_raw_w(r, raw_w, k, v, a, b, head_size)
    return _RWKV7RecurrenceTileLang.apply(r, raw_w, k, v, a, b, head_size, chunk_len)
