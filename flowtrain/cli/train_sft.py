from __future__ import annotations

import argparse
import json
import logging
import random
import time

import torch
from torch.utils.data import DataLoader

from flowtrain import FlowTrainConfig, FlowTrainTrainer, RWKV7, RWKV7Config, make_optimizer
from flowtrain.sft_data import RWKVTokenizerAdapter, SFTDataCollator, SFTJsonlDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _load_tokenizer(name_or_path: str | None, tokenizer_type: str):
    if tokenizer_type == "pyrwkv":
        return RWKVTokenizerAdapter()
    if not name_or_path:
        raise ValueError("--tokenizer is required when --tokenizer-type hf")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("flowtrain-train-sft requires transformers: pip install transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tune RWKV-7 with FlowTrain")
    parser.add_argument("--dataset", required=True, help="JSONL file with messages, prompt/completion, instruction/output, or text")
    parser.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer name or local path (required when --tokenizer-type hf)")
    parser.add_argument("--tokenizer-type", choices=("hf", "pyrwkv"), default="hf", help="Tokenizer backend; pyrwkv wraps the official RWKV-7 tokenizer")
    parser.add_argument("--checkpoint", default=None, help="RWKV-7 checkpoint to fine-tune; omit to initialize from scratch")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size for from-scratch training")
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--dim-ffn", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("adamw", "deepspeed_cpu_adam", "qr_muon", "adamw8bit"), default="adamw")
    parser.add_argument("--muon-beta", type=float, default=0.95)
    parser.add_argument("--muon-eps", type=float, default=1e-9)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-grad-slabs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("tilelang", "torch_ref"), default="tilelang")
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--activation-offload", choices=("none", "cpu"), default="cpu")
    parser.add_argument("--activation-quant", choices=("none", "int8"), default="none")
    parser.add_argument("--activation-strategy", choices=("recompute", "store_layer_inputs"), default="recompute")
    parser.add_argument("--logit-chunk-size", type=int, default=128)
    parser.add_argument("--pad-to-multiple-of", type=int, default=None)
    parser.add_argument(
        "--pad-to-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Pad each SFT batch to the smallest listed sequence-length bucket, e.g. 128 256 512",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad every SFT batch to --max-length for static sequence shapes",
    )
    parser.add_argument(
        "--debug-finite-checks",
        action="store_true",
        help="Raise at the first non-finite loss/logit/gradient/CPU-master parameter",
    )
    parser.add_argument("--full-sequence-loss", action="store_true", help="Train on prompt tokens too; default masks prompt tokens")
    parser.add_argument("--profile", action="store_true", help="Collect per-step profile (timing/memory/throughput) for infra planning")
    parser.add_argument("--profile-dir", default="runs", help="Directory for profile artifacts")
    parser.add_argument(
        "--profile-gpu-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample GPU utilization while profiling (use --no-profile-gpu-samples to disable)",
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace, vocab_size: int) -> RWKV7 | str:
    if args.checkpoint:
        return args.checkpoint
    return RWKV7(
        RWKV7Config(
            vocab_size=vocab_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            ctx_len=args.ctx_len,
            head_size=args.head_size,
            dim_ffn=args.dim_ffn,
            lora_rank_style="simplified" if args.n_embd <= 128 else "official",
            backend=args.backend,
            chunk_len=args.chunk_len,
        )
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_type)
    vocab_size = args.vocab_size or len(tokenizer)
    dataset = SFTJsonlDataset(
        args.dataset,
        tokenizer,
        max_length=args.max_length,
        mask_prompt=not args.full_sequence_loss,
        add_eos=True,
    )
    collator = SFTDataCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=args.max_length,
        pad_to_multiple_of=args.pad_to_multiple_of,
        pad_to_buckets=args.pad_to_buckets,
        pad_to_max_length=args.pad_to_max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    trainer = FlowTrainTrainer(
        _build_model(args, vocab_size),
        FlowTrainConfig(
            device=args.device,
            backend=args.backend,
            chunk_len=args.chunk_len,
            checkpoint_interval=args.checkpoint_interval,
            num_grad_slabs=args.num_grad_slabs,
            activation_offload=args.activation_offload,
            activation_quant=args.activation_quant,
            activation_strategy=args.activation_strategy,
            logit_chunk_size=args.logit_chunk_size,
            debug_finite_checks=args.debug_finite_checks,
        ),
        checkpoint_ctx_len=args.max_length,
    )
    optimizer = make_optimizer(
        trainer.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-18,
        optimizer=args.optimizer,
        muon_beta=args.muon_beta,
        muon_eps=args.muon_eps,
        debug_finite_checks=args.debug_finite_checks,
    )

    logger.info(
        "FlowTrain SFT: records=%d batch=%d max_length=%d backend=%s optimizer=%s activation=%s/%s/%s",
        len(dataset),
        args.batch_size,
        args.max_length,
        args.backend,
        args.optimizer,
        args.activation_offload,
        args.activation_quant,
        args.activation_strategy,
    )

    profiler = None
    if args.profile:
        from flowtrain.cli.sft_profile import StepProfiler

        total_params = int(sum(p.numel() for p in trainer.model.parameters()))
        profiler = StepProfiler(
            args.profile_dir,
            total_params=total_params,
            gpu_samples=args.profile_gpu_samples,
        )
        profiler.start_gpu_sampler()

    iterator = iter(dataloader)
    for step in range(args.num_steps):
        step_start = time.perf_counter()
        data_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        t_data = time.perf_counter() - data_start

        loss, total_tokens, timing = trainer.forward_and_backward(batch["input_ids"], batch["labels"])

        opt_start = time.perf_counter()
        if args.debug_finite_checks:
            trainer.check_finite_parameters("before_optimizer")
            trainer.check_finite_gradients("before_optimizer")
        optimizer.step()
        if args.debug_finite_checks:
            trainer.check_finite_parameters("after_optimizer")
        t_opt = time.perf_counter() - opt_start

        zero_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        trainer.zero_grad()
        t_zero = time.perf_counter() - zero_start

        t_step = time.perf_counter() - step_start

        valid_targets = int((batch["labels"][:, 1:] != -100).sum().item())
        logger.info(
            "step=%d loss=%.4f tokens=%d valid_targets=%d time=%.3fs fwd=%.3fs bwd=%.3fs opt=%.3fs act=%.4f GB/sample",
            step + 1,
            loss,
            total_tokens,
            valid_targets,
            timing["total"],
            timing["forward"],
            timing["backward"],
            t_opt,
            timing["activation_bytes_per_sample"] / 1024**3,
        )
        if profiler is not None:
            profiler.record(
                step=step + 1,
                loss=loss,
                total_tokens=total_tokens,
                valid_targets=valid_targets,
                t_step=t_step,
                t_data=t_data,
                t_fwd=timing["forward"],
                t_bwd=timing["backward"],
                t_opt=t_opt,
                t_zero=t_zero,
            )

    if profiler is not None:
        profiler.stop_gpu_sampler()
        summary = profiler.summarize_and_write(
            batch_size=args.batch_size,
            seq_len=args.max_length,
            optimizer_name=args.optimizer,
            model_label=args.checkpoint or "scratch",
        )
        logger.info("profile summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
