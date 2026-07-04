#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_minimal.rwkv7_core import RWKV7, RWKV7Config
from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig
from scripts.calc_resource import calc_activation_per_sample_gb


def infer_config(state: dict[str, torch.Tensor], ctx_len: int, use_cuda_kernel: bool, variant: str, chunk_len: int) -> RWKV7Config:
    vocab_size, n_embd = state["emb.weight"].shape
    layers = [int(k.split(".")[1]) for k in state if k.startswith("blocks.")]
    n_layer = max(layers) + 1
    head_size = state["blocks.0.att.r_k"].shape[1]
    dim_ffn = state["blocks.0.ffn.key.weight"].shape[0]
    return RWKV7Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        ctx_len=ctx_len,
        head_size=head_size,
        dim_ffn=dim_ffn,
        lora_rank_style="official",
        use_cuda_kernel=use_cuda_kernel,
        cuda_kernel_variant=variant,
        cuda_chunk_len=chunk_len,
    )


def measure_one(model: RWKV7, batch_size: int, seq_len: int, device: torch.device) -> float:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    hidden = model.forward_features(input_ids)
    loss = hidden.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    peak = torch.cuda.max_memory_allocated(device)
    del input_ids, hidden, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
    return (peak - baseline) / 1024**2


def measure_one_cpumaster(cpu_master: CPUMasterModel, batch_size: int, seq_len: int, device: torch.device) -> float:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)

    input_ids = torch.randint(0, cpu_master.vocab_size, (batch_size, seq_len), device="cpu")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    cpu_master.forward_and_backward(input_ids, attention_mask, labels)
    torch.cuda.synchronize(device)

    peak = torch.cuda.max_memory_allocated(device)
    del input_ids, attention_mask, labels
    cpu_master.zero_grad()
    torch.cuda.empty_cache()
    gc.collect()
    return (peak - baseline) / 1024**2


def fit_slope(xs: list[int], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return ys[0] / xs[0], 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RWKV7 activation memory slope from a real checkpoint")
    parser.add_argument("--checkpoint", default="rwkv7_minimal/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batches", type=str, default="1,2,4,8")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variant", choices=("base", "h100", "head128", "tilelang_block"), default="base")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--disable-cuda-kernel", action="store_true")
    parser.add_argument("--mode", choices=("native", "cpumaster"), default="cpumaster")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-grad-slabs", type=int, default=4)
    parser.add_argument("--activation-offload", choices=("none", "cpu"), default="none")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    state = torch.load(args.checkpoint, map_location="cpu")
    config = infer_config(
        state,
        ctx_len=max(8192, args.seq_len),
        use_cuda_kernel=not args.disable_cuda_kernel,
        variant=args.variant,
        chunk_len=args.chunk_len,
    )
    model = RWKV7(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.train()
    del state

    cpu_master = None
    if args.mode == "cpumaster":
        mt_config = CPUMasterConfig(
            model_name="rwkv7_activation_probe",
            dataset_path="__rwkv7_probe__",
            max_seq_len=args.seq_len,
            batch_size=1,
            gradient_accumulation_steps=1,
            num_steps=1,
            checkpoint_interval=args.checkpoint_interval,
            num_grad_slabs=args.num_grad_slabs,
            activation_offload=args.activation_offload,
            device=args.device,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        model = model.cpu()
        cpu_master = CPUMasterModel(model, mt_config)
        del model

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    if args.seq_len % args.chunk_len != 0 and config.use_cuda_kernel:
        raise SystemExit(f"--seq-len must be divisible by --chunk-len when CUDA kernel is enabled")

    # Warmup includes extension dispatch and allocator setup; exclude it from measurements.
    if args.mode == "cpumaster":
        assert cpu_master is not None
        measure_one_cpumaster(cpu_master, batches[0], args.seq_len, device)
    else:
        measure_one(model, batches[0], args.seq_len, device)

    rows = []
    for bs in batches:
        if args.mode == "cpumaster":
            assert cpu_master is not None
            mb = measure_one_cpumaster(cpu_master, bs, args.seq_len, device)
        else:
            mb = measure_one(model, bs, args.seq_len, device)
        rows.append((bs, mb))

    slope_mb, intercept_mb = fit_slope([x for x, _ in rows], [y for _, y in rows])
    measured_gb = slope_mb / 1024
    estimated_gb = calc_activation_per_sample_gb(
        config.n_embd,
        config.n_layer,
        args.seq_len,
        checkpoint_interval=args.checkpoint_interval,
        arch="rwkv7",
        head_size=config.head_size,
        chunk_len=args.chunk_len,
        activation_offload=args.activation_offload,
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"mode: {args.mode}")
    if args.mode == "cpumaster":
        print(f"activation_offload: {args.activation_offload}")
    print(f"config: layers={config.n_layer} hidden={config.n_embd} head_size={config.head_size} seq={args.seq_len}")
    print(f"kernel: enabled={config.use_cuda_kernel} variant={args.variant} chunk_len={args.chunk_len}")
    print("peak_delta_mb:")
    for bs, mb in rows:
        print(f"  bs={bs:<4} {mb:10.1f} MB")
    print(f"slope: {slope_mb:.1f} MB / batch item")
    print(f"intercept: {intercept_mb:.1f} MB")
    print(f"measured_activation_per_sample: {measured_gb:.4f} GB")
    print(f"calculator_estimate_per_sample: {estimated_gb:.4f} GB")
    if measured_gb > 0:
        print(f"estimate/measured: {estimated_gb / measured_gb:.2f}x")


if __name__ == "__main__":
    main()
