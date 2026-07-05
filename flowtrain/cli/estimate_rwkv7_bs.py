from __future__ import annotations

import argparse
from pathlib import Path

from flowtrain.estimator import (
    RWKV7Size,
    detect_cuda_memory_gb,
    detect_system_memory_gb,
    estimate_rwkv7_batch_size,
    infer_size_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FlowTrain RWKV-7 single-GPU batch size")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--n-embd", type=int, default=2048)
    parser.add_argument("--dim-ffn", type=int, default=None)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--lora-rank-style", choices=("official", "simplified"), default="official")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--activation-offload", choices=("none", "cpu"), default="cpu")
    parser.add_argument("--activation-quant", choices=("none", "int8"), default="none")
    parser.add_argument("--activation-strategy", choices=("recompute", "store_layer_inputs"), default="recompute")
    parser.add_argument("--optimizer", choices=("adamw", "deepspeed_cpu_adam", "qr_muon", "adamw8bit"), default="adamw")
    parser.add_argument("--gpu-gb", type=float, default=None)
    parser.add_argument("--cpu-gb", type=float, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-utilization", type=float, default=0.85)
    parser.add_argument("--reserve-gpu-gb", type=float, default=4.0)
    parser.add_argument("--reserve-cpu-gb", type=float, default=8.0)
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--hidden-gpu-copies", type=int, default=10)
    return parser.parse_args()


def _size_from_args(args: argparse.Namespace) -> RWKV7Size:
    if args.checkpoint is not None:
        return infer_size_from_checkpoint(
            args.checkpoint,
            lora_rank_style=args.lora_rank_style,
            head_size=args.head_size,
        )
    dim_ffn = args.dim_ffn
    if dim_ffn is None:
        dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    return RWKV7Size(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        dim_ffn=dim_ffn,
        head_size=args.head_size,
        lora_rank_style=args.lora_rank_style,
    )


def main() -> None:
    args = parse_args()
    size = _size_from_args(args)

    gpu_gb = args.gpu_gb
    if gpu_gb is None:
        gpu_gb = detect_cuda_memory_gb(args.device)
    if gpu_gb is None:
        raise SystemExit("GPU memory is unknown; pass --gpu-gb explicitly")

    cpu_gb = args.cpu_gb
    if cpu_gb is None:
        cpu_gb = detect_system_memory_gb()

    estimate = estimate_rwkv7_batch_size(
        size,
        seq_len=args.seq_len,
        gpu_gb=gpu_gb,
        cpu_gb=cpu_gb,
        activation_offload=args.activation_offload,
        activation_quant=args.activation_quant,
        activation_strategy=args.activation_strategy,
        checkpoint_interval=args.checkpoint_interval,
        gpu_utilization=args.gpu_utilization,
        cpu_utilization=args.cpu_utilization,
        reserve_gpu_gb=args.reserve_gpu_gb,
        reserve_cpu_gb=args.reserve_cpu_gb,
        optimizer=args.optimizer,
        logit_chunk_size=args.logit_chunk_size,
        hidden_gpu_copies=args.hidden_gpu_copies,
    )

    print("FlowTrain RWKV-7 batch size estimate")
    print("-------------------------------------")
    print(f"model: layers={size.n_layer} emb={size.n_embd} ffn={size.dim_ffn} vocab={size.vocab_size}")
    print(f"params: {estimate.model_params_b:.3f}B")
    print(f"seq_len: {args.seq_len}")
    print(f"gpu_memory: {gpu_gb:.2f} GB")
    if cpu_gb is not None:
        print(f"cpu_memory: {cpu_gb:.2f} GB")
    print(f"optimizer: {args.optimizer}")
    print(f"activation: offload={args.activation_offload} quant={args.activation_quant} strategy={args.activation_strategy}")
    print(f"logit_chunk_size: {args.logit_chunk_size if args.logit_chunk_size > 0 else 'full_sequence'}")
    print()
    print(f"estimated_max_batch_size: {estimate.max_batch_size}")
    print(f"gpu_limited_batch_size: {estimate.gpu_limit_batch_size}")
    if estimate.cpu_limit_batch_size is None:
        print("cpu_limited_batch_size: not batch-limited")
    else:
        print(f"cpu_limited_batch_size: {estimate.cpu_limit_batch_size}")
    print()
    print(f"gpu_base: {estimate.gpu_base_gb:.3f} GB")
    print(f"gpu_per_sample: {estimate.gpu_per_sample_gb:.4f} GB/sample")
    print(f"cpu_base: {estimate.cpu_base_gb:.3f} GB")
    print(f"cpu_per_sample: {estimate.cpu_per_sample_gb:.4f} GB/sample")
    print()
    print("assumptions:")
    for assumption in estimate.assumptions:
        print(f"- {assumption}")


if __name__ == "__main__":
    main()
