from __future__ import annotations

from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def _build_bf16_matmul_kernel(m: int, n: int, k: int, bm: int, bn: int, bk: int, num_warps: int):
    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.bfloat16),
        b: T.Tensor((k, n), T.bfloat16),
        c: T.Tensor((m, n), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), threads=num_warps * 32) as (pid_m, pid_n):
            a_shared = T.alloc_shared((bm, bk), T.bfloat16)
            b_shared = T.alloc_shared((bk, bn), T.bfloat16)
            c_local = T.alloc_fragment((bm, bn), T.float32)
            c_cast = T.alloc_fragment((bm, bn), T.bfloat16)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=1):
                T.copy(a[pid_m * bm:pid_m * bm + bm, ko * bk:ko * bk + bk], a_shared)
                T.copy(b[ko * bk:ko * bk + bk, pid_n * bn:pid_n * bn + bn], b_shared)
                T.gemm(a_shared, b_shared, c_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(c_local, c_cast)
            T.copy(c_cast, c[pid_m * bm:pid_m * bm + bm, pid_n * bn:pid_n * bn + bn])

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def _build_bf16_scalar_matmul_kernel(m: int, n: int, k: int, bm: int, bn: int, num_warps: int):
    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.bfloat16),
        b: T.Tensor((k, n), T.bfloat16),
        c: T.Tensor((m, n), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(m, bm), T.ceildiv(n, bn), threads=num_warps * 32) as (pid_m, pid_n):
            for i, j in T.Parallel(bm, bn):
                row = pid_m * bm + i
                col = pid_n * bn + j
                acc = T.alloc_var(T.float32)
                acc = T.cast(0, T.float32)
                for kk in T.serial(k):
                    acc += T.cast(a[row, kk], T.float32) * T.cast(b[kk, col], T.float32)
                if row < m and col < n:
                    c[row, col] = T.cast(acc, T.bfloat16)

    return kernel


def _tile_config(m: int, n: int, k: int) -> tuple[int, int, int, int] | None:
    bm = 64 if m >= 64 else 16
    bn = 64
    bk = 32 if k >= 32 else 16
    num_warps = 4
    return bm, bn, bk, num_warps


def can_use_tilelang_matmul(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.is_cuda
        and b.is_cuda
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.dim() == 2
        and b.dim() == 2
        and a.shape[1] == b.shape[0]
        and _tile_config(a.shape[0], b.shape[1], a.shape[1]) is not None
    )


@lru_cache(maxsize=128)
def _get_bf16_matmul_kernel(m: int, n: int, k: int, bm: int, bn: int, bk: int, num_warps: int):
    return _build_bf16_matmul_kernel(m, n, k, bm, bn, bk, num_warps)


@lru_cache(maxsize=64)
def _get_bf16_scalar_matmul_kernel(m: int, n: int, k: int, bm: int, bn: int, num_warps: int):
    return _build_bf16_scalar_matmul_kernel(m, n, k, bm, bn, num_warps)


def _tilelang_matmul_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k_b, n = b.shape
    if k_b != k:
        raise RuntimeError(f"matmul shape mismatch: {tuple(a.shape)} x {tuple(b.shape)}")
    config = _tile_config(m, n, k)
    if config is None:
        return a @ b
    bm, bn, bk, num_warps = config
    a_c = a.contiguous()
    b_c = b.contiguous()
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    if k < 16:
        kernel = _get_bf16_scalar_matmul_kernel(m, n, k, bm, bn, num_warps)
        kernel(a_c, b_c, out)
        return out
    kernel = _get_bf16_matmul_kernel(m, n, k, bm, bn, bk, num_warps)
    kernel(a_c, b_c, out)
    return out


class _TileLangMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        ctx.save_for_backward(a, b)
        return _tilelang_matmul_forward(a, b)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors
        grad_out_c = grad_out.contiguous()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = tilelang_matmul(grad_out_c, b.transpose(0, 1).contiguous())
        if ctx.needs_input_grad[1]:
            grad_b = tilelang_matmul(a.transpose(0, 1).contiguous(), grad_out_c)
        return grad_a, grad_b


def tilelang_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not can_use_tilelang_matmul(a, b):
        return a @ b
    return _TileLangMatmul.apply(a, b)


def tilelang_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    out = tilelang_matmul(x_2d, weight.transpose(0, 1).contiguous())
    return out.view(*orig_shape, weight.shape[0])
