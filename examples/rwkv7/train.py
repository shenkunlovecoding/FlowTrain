from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infinity import CPUMasterModel
from infinity.config import CPUMasterConfig
from rwkv7_minimal.rwkv7_core import RWKV7, RWKV7Config
from rwkv7_minimal.rwkv7_simplified_minimal import batch as digit_batch


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RWKV-7 with MegaTrain CPU-backed layer streaming")
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--dim-ffn", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-grad-slabs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--disable-cuda-kernel", action="store_true")
    parser.add_argument("--cuda-kernel-variant", choices=("h100", "base", "head128", "tilelang_block"), default="h100")
    parser.add_argument("--cuda-chunk-len", type=int, default=16)
    parser.add_argument("--activation-offload", choices=("none", "cpu"), default="none")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _apply_lr_scale(groups: list[dict], base_lr: float) -> list[dict]:
    scaled = []
    for group in groups:
        group = dict(group)
        group.pop("names", None)
        scale = group.pop("my_lr_scale", 1.0)
        group["lr"] = base_lr * scale
        scaled.append(group)
    return scaled


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = _dtype(args.dtype)
    if dtype != torch.bfloat16 and not args.disable_cuda_kernel:
        logger.warning("Bundled RWKV7 CUDA recurrence is bf16-only; disabling it for dtype=%s", args.dtype)
        args.disable_cuda_kernel = True

    rwkv_config = RWKV7Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        ctx_len=args.ctx_len,
        head_size=args.head_size,
        dim_ffn=args.dim_ffn,
        lora_rank_style="simplified" if args.n_embd <= 128 else "official",
        use_cuda_kernel=not args.disable_cuda_kernel,
        cuda_kernel_variant=args.cuda_kernel_variant,
        cuda_chunk_len=args.cuda_chunk_len,
    )
    model = RWKV7(rwkv_config).to(dtype=dtype)

    mt_config = CPUMasterConfig(
        model_name="rwkv7_minimal",
        dataset_path="__rwkv7_toy__",
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        activation_offload=args.activation_offload,
        device=args.device,
        dtype=dtype,
        seed=args.seed,
        log_interval=1,
        attn_implementation="eager",
    )
    cpu_master = CPUMasterModel(model, mt_config)
    del model

    optimizer = torch.optim.AdamW(
        _apply_lr_scale(cpu_master.optimizer_groups(args.weight_decay), args.lr),
        betas=(0.9, 0.99),
        eps=1e-18,
    )

    logger.info(
        "RWKV7 training: layers=%d emb=%d head_size=%d kernel=%s variant=%s chunk=%d",
        args.n_layer,
        args.n_embd,
        args.head_size,
        not args.disable_cuda_kernel,
        args.cuda_kernel_variant,
        args.cuda_chunk_len,
    )

    for step in range(args.num_steps):
        input_ids = digit_batch(args.batch_size, args.seq_len)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)

        loss, total_tokens, timing = cpu_master.forward_and_backward(input_ids, attention_mask, labels)
        torch.nn.utils.clip_grad_norm_(cpu_master.get_parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cpu_master.zero_grad()

        logger.info(
            "step=%d loss=%.4f tokens=%d time=%.3fs fwd=%.3fs bwd=%.3fs",
            step + 1,
            loss,
            total_tokens,
            timing["total"],
            timing["forward"],
            timing["backward"],
        )


if __name__ == "__main__":
    main()
