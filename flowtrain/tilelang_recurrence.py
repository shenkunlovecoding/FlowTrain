# NOTE: deliberately no `from __future__ import annotations` here. With PEP 563
# the kernel parameter annotations (e.g. T.Tensor((batch, timesteps, channels),
# T.bfloat16)) become strings, and dimension names that appear ONLY in
# annotations (not in the kernel body) are never captured as closure cells, so
# TileLang's get_type_hints raises NameError at JIT time. Eager annotations
# resolve the dimensions against the enclosing scope as expected.

import os
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

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
    def _build_kernel(batch: int, timesteps: int, channels: int, head_size: int):
        n_head = channels // head_size

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
                mix_vec = T.alloc_fragment((head_size,), T.float32)
                T.clear(state)
                for t in T.serial(timesteps):
                    base = pid_h * head_size
                    # stage 1: mix[i] = sum_p state[i,p] * a[p]  (row dot, own parallel region)
                    for i in T.Parallel(head_size):
                        acc = T.alloc_var(T.float32)
                        acc = T.cast(0, T.float32)
                        for p in T.serial(head_size):
                            acc += state[i, p] * T.cast(a[pid_b, t, base + p], T.float32)
                        mix_vec[i] = acc
                    # stage 2: elementwise state update (consistent (i,j) indexing)
                    for i, j in T.Parallel(head_size, head_size):
                        state[i, j] = (
                            state[i, j] * T.cast(decay[pid_b, t, base + j], T.float32)
                            + mix_vec[i] * T.cast(b[pid_b, t, base + j], T.float32)
                            + T.cast(v[pid_b, t, base + i], T.float32)
                            * T.cast(k[pid_b, t, base + j], T.float32)
                        )
                    # output: out[i] = sum_j state[i,j] * r[j]
                    for i in T.Parallel(head_size):
                        acc = T.alloc_var(T.float32)
                        acc = T.cast(0, T.float32)
                        for j in T.serial(head_size):
                            acc += state[i, j] * T.cast(r[pid_b, t, base + j], T.float32)
                        out[pid_b, t, base + i] = T.cast(acc, T.bfloat16)

        return kernel

    return _build_kernel(batch, timesteps, channels, head_size)


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


def _pick_recurrence_chunk(timesteps: int, target: int | None = None) -> int:
    """Pick a checkpoint chunk size that divides ``timesteps``.

    The backward kernel is JIT-specialized on a static chunk size, so the segment
    length must divide the sequence length exactly (no per-step remainder mask in
    the kernel). We prefer ``target`` and fall back through smaller divisors; if
    nothing divides, chunk == timesteps collapses to a single segment.
    """
    if target is None:
        target = int(os.environ.get("FLOWTRAIN_RECURRENCE_CHUNK_TARGET", "128"))
    for candidate in (target, target // 2, 32, 16, 8, 4, 2, 1):
        if candidate >= 1 and timesteps % candidate == 0:
            return min(candidate, timesteps)
    return timesteps


@lru_cache(maxsize=64)
def _get_recurrence_backward_kernel(
    batch: int,
    timesteps: int,
    channels: int,
    head_size: int,
    chunk: int,
    num_seg: int,
):
    """Fused RWKV-7 recurrence backward kernel.

    Phase A re-runs the forward scan and dumps the pre-update state at every
    chunk boundary (every ``chunk`` timesteps) into ``s_ckpt``. Phase B walks
    chunks in reverse; per chunk it sub-recomputes the segment states into
    ``s_seg`` from the boundary and runs the adjoint scan producing gradients for
    r, decay, k, v, a, b. The adjoint state H is carried across chunk boundaries.

    Workspace is O((T/chunk + chunk) * B * n_head * head_size^2 * 4) bytes, i.e.
    checkpoint-recompute rather than storing all T forward states.
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
    def _build_kernel(
        batch: int,
        timesteps: int,
        channels: int,
        head_size: int,
        chunk: int,
        num_seg: int,
    ):
        n_head = channels // head_size
        n = head_size

        @T.prim_func
        def kernel(
            r: T.Tensor((batch, timesteps, channels), T.bfloat16),
            decay: T.Tensor((batch, timesteps, channels), T.bfloat16),
            k: T.Tensor((batch, timesteps, channels), T.bfloat16),
            v: T.Tensor((batch, timesteps, channels), T.bfloat16),
            a: T.Tensor((batch, timesteps, channels), T.bfloat16),
            b: T.Tensor((batch, timesteps, channels), T.bfloat16),
            gout: T.Tensor((batch, timesteps, channels), T.bfloat16),
            s_ckpt: T.Tensor((batch, num_seg, n_head, n, n), T.float32),
            s_seg: T.Tensor((batch, n_head, chunk + 1, n, n), T.float32),
            gr: T.Tensor((batch, timesteps, channels), T.bfloat16),
            gw: T.Tensor((batch, timesteps, channels), T.bfloat16),
            gk: T.Tensor((batch, timesteps, channels), T.bfloat16),
            gv: T.Tensor((batch, timesteps, channels), T.bfloat16),
            ga: T.Tensor((batch, timesteps, channels), T.bfloat16),
            gb: T.Tensor((batch, timesteps, channels), T.bfloat16),
        ):
            with T.Kernel(batch, n_head, threads=256) as (pid_b, pid_h):
                state = T.alloc_fragment((n, n), T.float32)
                H = T.alloc_fragment((n, n), T.float32)
                mix_vec = T.alloc_fragment((n,), T.float32)
                pvec = T.alloc_fragment((n,), T.float32)
                gpvec = T.alloc_fragment((n,), T.float32)
                g_ws = T.alloc_shared((n, n), T.float32)

                # ---------------- Phase A: forward scan, dump boundary states ----------------
                T.clear(state)
                for seg in T.serial(num_seg):
                    for i, j in T.Parallel(n, n):
                        s_ckpt[pid_b, seg, pid_h, i, j] = state[i, j]
                    t_start = seg * chunk
                    for local in T.serial(chunk):
                        t = t_start + local
                        base = pid_h * n
                        # stage 1: mix[i] = sum_p state[i,p] * a[p]
                        for i in T.Parallel(n):
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for p in T.serial(n):
                                acc += state[i, p] * T.cast(a[pid_b, t, base + p], T.float32)
                            mix_vec[i] = acc
                        # stage 2: elementwise state update
                        for i, j in T.Parallel(n, n):
                            state[i, j] = (
                                state[i, j] * T.cast(decay[pid_b, t, base + j], T.float32)
                                + mix_vec[i] * T.cast(b[pid_b, t, base + j], T.float32)
                                + T.cast(v[pid_b, t, base + i], T.float32)
                                * T.cast(k[pid_b, t, base + j], T.float32)
                            )

                # ---------------- Phase B: chunked reverse adjoint scan ----------------
                T.clear(H)
                for idx in T.serial(num_seg):
                    seg = num_seg - 1 - idx
                    t_start = seg * chunk
                    for i, j in T.Parallel(n, n):
                        state[i, j] = s_ckpt[pid_b, seg, pid_h, i, j]
                        s_seg[pid_b, pid_h, 0, i, j] = state[i, j]
                    # sub-forward: fill s_seg[1 .. chunk]
                    for local in T.serial(chunk):
                        t = t_start + local
                        base = pid_h * n
                        # stage 1: mix[i] = sum_p state[i,p] * a[p]
                        for i in T.Parallel(n):
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for p in T.serial(n):
                                acc += state[i, p] * T.cast(a[pid_b, t, base + p], T.float32)
                            mix_vec[i] = acc
                        # stage 2: elementwise state update
                        for i, j in T.Parallel(n, n):
                            state[i, j] = (
                                state[i, j] * T.cast(decay[pid_b, t, base + j], T.float32)
                                + mix_vec[i] * T.cast(b[pid_b, t, base + j], T.float32)
                                + T.cast(v[pid_b, t, base + i], T.float32)
                                * T.cast(k[pid_b, t, base + j], T.float32)
                            )
                        for i, j in T.Parallel(n, n):
                            s_seg[pid_b, pid_h, local + 1, i, j] = state[i, j]
                    # sub-backward over local steps (inv = chunk-1 .. 0).
                    # S_pre/S_post are read directly from s_seg (global) to keep only two
                    # matrix fragments (state, H) live, matching the forward's footprint.
                    for bidx in T.serial(chunk):
                        inv = chunk - 1 - bidx
                        t = t_start + inv
                        # 1. grad r: gr[col] = sum_i S_post[i,col] * gout[i]
                        for col in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for i in T.serial(n):
                                acc += s_seg[pid_b, pid_h, inv + 1, i, col] * T.cast(
                                    gout[pid_b, t, base + i], T.float32
                                )
                            gr[pid_b, t, base + col] = T.cast(acc, T.bfloat16)
                        # 2. G = H + gout (x) r.  G is staged in SHARED memory (g_ws),
                        # not a fragment: G is reduced along BOTH axes below (gw/gk/gb
                        # reduce over rows, gv/gpvec over cols), and a single fragment
                        # layout cannot satisfy both.  Shared memory tolerates mixed-axis
                        # reads with ~20× lower latency than global memory.
                        # H stays a fragment, used only elementwise.
                        for i, j in T.Parallel(n, n):
                            base = pid_h * n
                            g_ws[i, j] = H[i, j] + T.cast(
                                gout[pid_b, t, base + i], T.float32
                            ) * T.cast(r[pid_b, t, base + j], T.float32)
                        # 3. grad decay: gw[col] = sum_i G[i,col] * Spre[i,col]
                        for col in T.Parallel(n):
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for i in T.serial(n):
                                acc += g_ws[i, col] * s_seg[pid_b, pid_h, inv, i, col]
                            base = pid_h * n
                            gw[pid_b, t, base + col] = T.cast(acc, T.bfloat16)
                        # 4. grad v: gv[i] = sum_j G[i,j] * k[j]
                        for row in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for j in T.serial(n):
                                acc += g_ws[row, j] * T.cast(k[pid_b, t, base + j], T.float32)
                            gv[pid_b, t, base + row] = T.cast(acc, T.bfloat16)
                        # 5. grad k: gk[col] = sum_i G[i,col] * v[i]
                        for col in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for i in T.serial(n):
                                acc += g_ws[i, col] * T.cast(v[pid_b, t, base + i], T.float32)
                            gk[pid_b, t, base + col] = T.cast(acc, T.bfloat16)
                        # 6. pvec = Spre @ a   (pvec[i] = sum_q Spre[i,q] * a[q])
                        for row in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for q in T.serial(n):
                                acc += s_seg[pid_b, pid_h, inv, row, q] * T.cast(
                                    a[pid_b, t, base + q], T.float32
                                )
                            pvec[row] = acc
                        # 7. grad b: gb[col] = sum_i G[i,col] * pvec[i]
                        for col in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for i in T.serial(n):
                                acc += g_ws[i, col] * pvec[i]
                            gb[pid_b, t, base + col] = T.cast(acc, T.bfloat16)
                        # 8. gpvec = G @ b   (gpvec[i] = sum_j G[i,j] * b[j])
                        for row in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for j in T.serial(n):
                                acc += g_ws[row, j] * T.cast(b[pid_b, t, base + j], T.float32)
                            gpvec[row] = acc
                        # 9. grad a: ga[col] = sum_i Spre[i,col] * gpvec[i]
                        for col in T.Parallel(n):
                            base = pid_h * n
                            acc = T.alloc_var(T.float32)
                            acc = T.cast(0, T.float32)
                            for i in T.serial(n):
                                acc += s_seg[pid_b, pid_h, inv, i, col] * gpvec[i]
                            ga[pid_b, t, base + col] = T.cast(acc, T.bfloat16)
                        # 10. carry: H = G * decay + gpvec (x) a   (G in g_ws)
                        for i, j in T.Parallel(n, n):
                            base = pid_h * n
                            H[i, j] = g_ws[i, j] * T.cast(
                                decay[pid_b, t, base + j], T.float32
                            ) + gpvec[i] * T.cast(a[pid_b, t, base + j], T.float32)

        return kernel

    return _build_kernel(batch, timesteps, channels, head_size, chunk, num_seg)


def _can_use_fused_backward(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gout: torch.Tensor,
    head_size: int,
    chunk_len: int,
) -> bool:
    return can_use_rwkv7_recurrence_tilelang(r, raw_w, k, v, a, b, head_size, chunk_len) and (
        gout.is_cuda
        and gout.dtype == torch.bfloat16
        and gout.dim() == 3
        and gout.shape == r.shape
    )


def _fused_recurrence_backward(
    r: torch.Tensor,
    raw_w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gout: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused backward kernel and chain grad(decay) -> grad(raw_w).

    Returns (gr, graw_w, gk, gv, ga, gb) matching the forward's input order
    (r, raw_w, k, v, a, b).
    """
    batch, timesteps, channels = r.shape
    n_head = channels // head_size
    chunk = _pick_recurrence_chunk(timesteps)
    num_seg = timesteps // chunk

    w_mid = -F.softplus(-raw_w.float()) - 0.5
    exp_w = torch.exp(w_mid)
    decay = torch.exp(-exp_w).to(dtype=r.dtype)

    s_ckpt = torch.zeros(
        (batch, num_seg, n_head, head_size, head_size), device=r.device, dtype=torch.float32
    )
    s_seg = torch.zeros(
        (batch, n_head, chunk + 1, head_size, head_size), device=r.device, dtype=torch.float32
    )
    gr = torch.zeros_like(r)
    gw = torch.zeros_like(r)
    gk = torch.zeros_like(k)
    gv = torch.zeros_like(v)
    ga = torch.zeros_like(a)
    gb = torch.zeros_like(b)

    kernel = _get_recurrence_backward_kernel(batch, timesteps, channels, head_size, chunk, num_seg)
    kernel(
        r.contiguous(),
        decay.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        gout.contiguous(),
        s_ckpt,
        s_seg,
        gr,
        gw,
        gk,
        gv,
        ga,
        gb,
    )

    # grad(raw_w) = grad(decay) * d(decay)/d(raw_w)
    # decay = exp(-exp(w_mid)), w_mid = -softplus(-raw_w) - 0.5
    # d(decay)/d(raw_w) = -decay * exp(w_mid) * sigmoid(-raw_w)
    sigmoid_neg = torch.sigmoid(-raw_w.float())
    graw_w = (gw.float() * (decay.float() * (-exp_w) * sigmoid_neg)).to(dtype=raw_w.dtype)
    return gr, graw_w, gk, gv, ga, gb


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
        head_size = ctx.head_size
        chunk_len = ctx.chunk_len
        if _can_use_fused_backward(r, raw_w, k, v, a, b, grad_out, head_size, chunk_len):
            return (*_fused_recurrence_backward(r, raw_w, k, v, a, b, grad_out, head_size), None, None)
        # Reference fallback: recompute via the pure-PyTorch path under autograd.
        detached = [x.detach().requires_grad_(True) for x in (r, raw_w, k, v, a, b)]
        with torch.enable_grad():
            out = _torch_ref_from_raw_w(*detached, head_size)
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
