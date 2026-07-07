#!/usr/bin/env python3
"""Benchmark host<->GPU PCIe copy bandwidth with PyTorch."""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--pageable", action="store_true", help="Use pageable CPU memory instead of pinned memory")
    parser.add_argument("--bidirectional", action="store_true", help="Also measure simultaneous H2D and D2H copies")
    return parser.parse_args()


def gbps(num_bytes: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return float("inf")
    return num_bytes / (elapsed_ms / 1000.0) / 1e9


def make_cpu_buffer(num_bytes: int, pinned: bool) -> torch.Tensor:
    try:
        return torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=pinned)
    except RuntimeError:
        if pinned:
            raise RuntimeError("pinned CPU allocation failed; retry with --pageable") from None
        raise


def measure_one_way(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    num_bytes: int,
    warmup: int,
    iters: int,
    stream: torch.cuda.Stream,
) -> tuple[float, float]:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        for _ in range(iters):
            start.record()
            dst.copy_(src, non_blocking=True)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
    return gbps(num_bytes, statistics.median(times_ms)), gbps(num_bytes, statistics.mean(times_ms))


def measure_bidirectional(
    h2d_src: torch.Tensor,
    h2d_dst: torch.Tensor,
    d2h_src: torch.Tensor,
    d2h_dst: torch.Tensor,
    *,
    num_bytes: int,
    warmup: int,
    iters: int,
) -> float:
    h2d_stream = torch.cuda.Stream(device=h2d_dst.device)
    d2h_stream = torch.cuda.Stream(device=d2h_src.device)
    for _ in range(warmup):
        with torch.cuda.stream(h2d_stream):
            h2d_dst.copy_(h2d_src, non_blocking=True)
        with torch.cuda.stream(d2h_stream):
            d2h_dst.copy_(d2h_src, non_blocking=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.cuda.stream(h2d_stream):
            h2d_dst.copy_(h2d_src, non_blocking=True)
        with torch.cuda.stream(d2h_stream):
            d2h_dst.copy_(d2h_src, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (2 * num_bytes * iters) / elapsed / 1e9


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.iters < 1:
        raise ValueError("iters must be >= 1")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    pinned = not args.pageable
    stream = torch.cuda.Stream(device=device)

    print(f"device={torch.cuda.get_device_name(device)} pinned={pinned} iters={args.iters}")
    if args.bidirectional:
        print(f"{'size_mb':>8} {'h2d_med':>10} {'h2d_mean':>10} {'d2h_med':>10} {'d2h_mean':>10} {'bidir':>10}")
    else:
        print(f"{'size_mb':>8} {'h2d_med':>10} {'h2d_mean':>10} {'d2h_med':>10} {'d2h_mean':>10}")

    for size_mb in args.sizes_mb:
        if size_mb < 1:
            raise ValueError("all sizes must be >= 1 MB")
        num_bytes = size_mb * 1024 * 1024
        cpu_h2d = make_cpu_buffer(num_bytes, pinned)
        cpu_d2h = make_cpu_buffer(num_bytes, pinned)
        gpu_h2d = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        gpu_d2h = torch.empty(num_bytes, dtype=torch.uint8, device=device)

        h2d_med, h2d_mean = measure_one_way(
            cpu_h2d,
            gpu_h2d,
            num_bytes=num_bytes,
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        d2h_med, d2h_mean = measure_one_way(
            gpu_d2h,
            cpu_d2h,
            num_bytes=num_bytes,
            warmup=args.warmup,
            iters=args.iters,
            stream=stream,
        )
        if args.bidirectional:
            bidir = measure_bidirectional(
                cpu_h2d,
                gpu_h2d,
                gpu_d2h,
                cpu_d2h,
                num_bytes=num_bytes,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(f"{size_mb:8d} {h2d_med:10.2f} {h2d_mean:10.2f} {d2h_med:10.2f} {d2h_mean:10.2f} {bidir:10.2f}")
        else:
            print(f"{size_mb:8d} {h2d_med:10.2f} {h2d_mean:10.2f} {d2h_med:10.2f} {d2h_mean:10.2f}")


if __name__ == "__main__":
    main()
