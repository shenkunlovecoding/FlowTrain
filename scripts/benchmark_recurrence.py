#!/usr/bin/env python3
"""Benchmark TileLang RWKV-7 recurrence forward+backward chunk choices."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from flowtrain.tilelang_recurrence import (  # noqa: E402
    _get_recurrence_backward_kernel,
    _pick_recurrence_chunk,
    rwkv7_recurrence_tilelang,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--targets", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_inputs(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(args.seed)
    names = ("r", "raw_w", "k", "v", "a", "b")
    del names
    return tuple(
        (torch.randn(args.batch, args.timesteps, args.channels, device=device, dtype=torch.bfloat16) * 0.1)
        .detach()
        .requires_grad_(True)
        for _ in range(6)
    )


def run_once(inputs: tuple[torch.Tensor, ...], args: argparse.Namespace) -> None:
    for tensor in inputs:
        tensor.grad = None
    out = rwkv7_recurrence_tilelang(*inputs, args.head_size, chunk_len=args.chunk_len)
    loss = out.float().sum()
    loss.backward()


def benchmark_target(target: int, args: argparse.Namespace, device: torch.device) -> tuple[int, float, float]:
    os.environ["FLOWTRAIN_RECURRENCE_CHUNK_TARGET"] = str(target)
    _get_recurrence_backward_kernel.cache_clear()
    picked = _pick_recurrence_chunk(args.timesteps)
    inputs = make_inputs(args, device)

    for _ in range(args.warmup):
        run_once(inputs, args)
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        run_once(inputs, args)
        end.record()
        torch.cuda.synchronize(device)
        times_ms.append(start.elapsed_time(end))
    return picked, statistics.median(times_ms), statistics.mean(times_ms)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.channels % args.head_size != 0:
        raise SystemExit("--channels must be divisible by --head-size")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    print("target,picked_chunk,batch,timesteps,channels,median_ms,mean_ms")
    for target in args.targets:
        picked, median_ms, mean_ms = benchmark_target(target, args, device)
        print(
            f"{target},{picked},{args.batch},{args.timesteps},{args.channels},"
            f"{median_ms:.4f},{mean_ms:.4f}"
        )


if __name__ == "__main__":
    main()
