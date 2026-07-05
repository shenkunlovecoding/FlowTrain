from functools import lru_cache

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def _build_time_mix_shift_kernel(batch: int, timesteps: int, channels: int, block_size: int):
    @T.prim_func
    def kernel(
        x: T.Tensor((batch, timesteps, channels), T.bfloat16),
        x_r: T.Tensor((1, 1, channels), T.bfloat16),
        x_w: T.Tensor((1, 1, channels), T.bfloat16),
        x_k: T.Tensor((1, 1, channels), T.bfloat16),
        x_v: T.Tensor((1, 1, channels), T.bfloat16),
        x_a: T.Tensor((1, 1, channels), T.bfloat16),
        x_g: T.Tensor((1, 1, channels), T.bfloat16),
        xr: T.Tensor((batch, timesteps, channels), T.bfloat16),
        xw: T.Tensor((batch, timesteps, channels), T.bfloat16),
        xk: T.Tensor((batch, timesteps, channels), T.bfloat16),
        xv: T.Tensor((batch, timesteps, channels), T.bfloat16),
        xa: T.Tensor((batch, timesteps, channels), T.bfloat16),
        xg: T.Tensor((batch, timesteps, channels), T.bfloat16),
    ):
        total = batch * timesteps * channels
        with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as pid:
            for i in T.Parallel(block_size):
                linear = pid * block_size + i
                if linear < total:
                    channel = linear % channels
                    tmp = linear // channels
                    time = tmp % timesteps
                    batch_idx = tmp // timesteps
                    current = T.cast(x[batch_idx, time, channel], T.float32)
                    previous = T.if_then_else(
                        time > 0,
                        T.cast(x[batch_idx, time - 1, channel], T.float32),
                        T.cast(0, T.float32),
                    )
                    delta = previous - current
                    xr[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_r[0, 0, channel], T.float32),
                        T.bfloat16,
                    )
                    xw[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_w[0, 0, channel], T.float32),
                        T.bfloat16,
                    )
                    xk[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_k[0, 0, channel], T.float32),
                        T.bfloat16,
                    )
                    xv[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_v[0, 0, channel], T.float32),
                        T.bfloat16,
                    )
                    xa[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_a[0, 0, channel], T.float32),
                        T.bfloat16,
                    )
                    xg[batch_idx, time, channel] = T.cast(
                        current + delta * T.cast(x_g[0, 0, channel], T.float32),
                        T.bfloat16,
                    )

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def _build_time_mix_post_kernel(batch: int, timesteps: int, channels: int, head_size: int, eps: float):
    n_head = channels // head_size

    @T.prim_func
    def kernel(
        recurrence: T.Tensor((batch, timesteps, channels), T.bfloat16),
        r: T.Tensor((batch, timesteps, channels), T.bfloat16),
        k: T.Tensor((batch, timesteps, channels), T.bfloat16),
        v: T.Tensor((batch, timesteps, channels), T.bfloat16),
        g: T.Tensor((batch, timesteps, channels), T.bfloat16),
        r_k: T.Tensor((n_head, head_size), T.bfloat16),
        ln_weight: T.Tensor((channels,), T.bfloat16),
        ln_bias: T.Tensor((channels,), T.bfloat16),
        out: T.Tensor((batch, timesteps, channels), T.bfloat16),
    ):
        with T.Kernel(batch, timesteps, n_head, threads=128) as (pid_b, pid_t, pid_h):
            base = pid_h * head_size
            mean = T.alloc_var(T.float32)
            var = T.alloc_var(T.float32)
            bonus = T.alloc_var(T.float32)
            mean = T.cast(0, T.float32)
            var = T.cast(0, T.float32)
            bonus = T.cast(0, T.float32)

            for i in T.serial(head_size):
                value = T.cast(recurrence[pid_b, pid_t, base + i], T.float32)
                mean += value
                bonus += (
                    T.cast(r[pid_b, pid_t, base + i], T.float32)
                    * T.cast(k[pid_b, pid_t, base + i], T.float32)
                    * T.cast(r_k[pid_h, i], T.float32)
                )
            mean = mean / T.cast(head_size, T.float32)

            for i in T.serial(head_size):
                centered = T.cast(recurrence[pid_b, pid_t, base + i], T.float32) - mean
                var += centered * centered
            var = var / T.cast(head_size, T.float32)
            inv_std = T.rsqrt(var + T.cast(eps, T.float32))

            for i in T.Parallel(head_size):
                channel = base + i
                normed = (
                    (T.cast(recurrence[pid_b, pid_t, channel], T.float32) - mean)
                    * inv_std
                    * T.cast(ln_weight[channel], T.float32)
                    + T.cast(ln_bias[channel], T.float32)
                )
                mixed = normed + bonus * T.cast(v[pid_b, pid_t, channel], T.float32)
                out[pid_b, pid_t, channel] = T.cast(
                    mixed * T.cast(g[pid_b, pid_t, channel], T.float32),
                    T.bfloat16,
                )

    return kernel


@lru_cache(maxsize=64)
def _get_time_mix_shift_kernel(batch: int, timesteps: int, channels: int, block_size: int):
    return _build_time_mix_shift_kernel(batch, timesteps, channels, block_size)


@lru_cache(maxsize=64)
def _get_time_mix_post_kernel(batch: int, timesteps: int, channels: int, head_size: int, eps: float):
    return _build_time_mix_post_kernel(batch, timesteps, channels, head_size, eps)


def can_use_time_mix_shift_tilelang(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> bool:
    tensors = (x, x_r, x_w, x_k, x_v, x_a, x_g)
    if not all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors):
        return False
    if x.dim() != 3:
        return False
    channels = x.shape[-1]
    return all(t.shape == (1, 1, channels) for t in (x_r, x_w, x_k, x_v, x_a, x_g))


def can_use_time_mix_post_tilelang(
    recurrence: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    head_size: int,
) -> bool:
    tensors = (recurrence, r, k, v, g, r_k, ln_weight, ln_bias)
    if not all(t.is_cuda and t.dtype == torch.bfloat16 for t in tensors):
        return False
    if recurrence.dim() != 3 or any(t.shape != recurrence.shape for t in (r, k, v, g)):
        return False
    batch, timesteps, channels = recurrence.shape
    if head_size != 64 or channels % head_size != 0:
        return False
    if r_k.shape != (channels // head_size, head_size):
        return False
    return ln_weight.shape == (channels,) and ln_bias.shape == (channels,)


def time_mix_shift_tilelang(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not can_use_time_mix_shift_tilelang(x, x_r, x_w, x_k, x_v, x_a, x_g):
        raise RuntimeError("time_mix_shift_tilelang received unsupported tensors")
    x_c = x.contiguous()
    batch, timesteps, channels = x.shape
    outs = tuple(torch.empty_like(x) for _ in range(6))
    block_size = 256
    kernel = _get_time_mix_shift_kernel(batch, timesteps, channels, block_size)
    kernel(
        x_c,
        x_r.contiguous(),
        x_w.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        x_a.contiguous(),
        x_g.contiguous(),
        *outs,
    )
    return outs


def time_mix_post_tilelang(
    recurrence: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    r_k: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    head_size: int,
    eps: float,
) -> torch.Tensor:
    if not can_use_time_mix_post_tilelang(recurrence, r, k, v, g, r_k, ln_weight, ln_bias, head_size):
        raise RuntimeError("time_mix_post_tilelang received unsupported tensors")
    recurrence_c = recurrence.contiguous()
    r_c = r.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    r_k_c = r_k.contiguous()
    ln_weight_c = ln_weight.contiguous()
    ln_bias_c = ln_bias.contiguous()
    batch, timesteps, channels = recurrence.shape
    out = torch.empty_like(recurrence)
    kernel = _get_time_mix_post_kernel(batch, timesteps, channels, head_size, float(eps))
    kernel(recurrence_c, r_c, k_c, v_c, g_c, r_k_c, ln_weight_c, ln_bias_c, out)
    return out
