from __future__ import annotations

import argparse
import json
import logging
import random
import time

import torch
from torch.utils.data import DataLoader

from flowtrain import FlowTrainConfig, FlowTrainTrainer, RWKV7, RWKV7Config, make_optimizer
from flowtrain.sft_data import (
    LengthBucketBatchSampler,
    RWKVTokenizerAdapter,
    SFTDataCollator,
    SFTJsonlDataset,
    compute_sft_lengths,
)


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
    parser.add_argument("--optimizer-eps", type=float, default=None)
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
        "--length-bucket-batch",
        action="store_true",
        help="Group SFT samples into length-homogeneous batches (one bucket per batch) to "
        "minimize padding and keep TileLang kernel shapes static. Requires --pad-to-buckets; "
        "every --pad-to-buckets value and --max-length must be a multiple of --chunk-len "
        "for the tilelang backend.",
    )
    parser.add_argument(
        "--length-bucket-drop-tail",
        action="store_true",
        help="With --length-bucket-batch, drop each bucket's final short batch so every batch "
        "is exactly --batch-size (keeps the batch dimension a static TileLang JIT key). Default "
        "keeps all samples, accepting ~one extra per-bucket (remainder, bucket) specialization.",
    )
    parser.add_argument(
        "--debug-finite-checks",
        action="store_true",
        help="Raise at the first non-finite loss/logit/gradient/CPU-master parameter",
    )
    parser.add_argument(
        "--debug-stats",
        action="store_true",
        help="Log internal loss/logit/gradient/parameter statistics for diagnosing silent saturation",
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


def _log_forward_debug(step: int, debug: dict[str, float]) -> None:
    logger.info(
        (
            "debug step=%d loss=%.8e ce_sum=%.6e valid=%d "
            "hidden_rms=%.6g hidden_absmax=%.6g "
            "logit_min=%.6g logit_max=%.6g logit_absmax=%.6g "
            "target_logit_mean=%.6g target_gap_mean=%.6g target_gap_min=%.6g "
            "nll_mean=%.6g nll_min=%.6g nll_max=%.6g"
        ),
        step,
        debug.get("loss", float("nan")),
        debug.get("ce_sum", float("nan")),
        int(debug.get("valid_tokens", 0.0)),
        debug.get("hidden_before_head_rms", float("nan")),
        debug.get("hidden_before_head_absmax", float("nan")),
        debug.get("logit_min", float("nan")),
        debug.get("logit_max", float("nan")),
        debug.get("logit_absmax", float("nan")),
        debug.get("target_logit_mean", float("nan")),
        debug.get("target_gap_mean", float("nan")),
        debug.get("target_gap_min", float("nan")),
        debug.get("nll_mean", float("nan")),
        debug.get("nll_min", float("nan")),
        debug.get("nll_max", float("nan")),
    )


def _log_optimizer_debug(
    step: int,
    grad_stats: dict[str, float],
    param_before: dict[str, float],
    param_after: dict[str, float],
    opt_grad_norm: object,
) -> None:
    try:
        opt_grad_norm_float = float(opt_grad_norm) if opt_grad_norm is not None else float("nan")
    except (TypeError, ValueError):
        opt_grad_norm_float = float("nan")
    logger.info(
        (
            "debug step=%d grad_norm=%.6g grad_absmax=%.6g grad_bad=%d "
            "opt_grad_norm=%.6g param_norm_before=%.6g param_norm_after=%.6g "
            "param_absmax_before=%.6g param_absmax_after=%.6g param_bad_after=%d"
        ),
        step,
        grad_stats.get("grad_norm", float("nan")),
        grad_stats.get("grad_absmax", float("nan")),
        int(grad_stats.get("grad_bad", 0.0)),
        opt_grad_norm_float,
        param_before.get("param_before_norm", float("nan")),
        param_after.get("param_after_norm", float("nan")),
        param_before.get("param_before_absmax", float("nan")),
        param_after.get("param_after_absmax", float("nan")),
        int(param_after.get("param_after_bad", 0.0)),
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
    if args.length_bucket_batch:
        if not args.pad_to_buckets:
            raise SystemExit(
                "--length-bucket-batch requires --pad-to-buckets (e.g. --pad-to-buckets 64 128 256)"
            )
        if args.pad_to_max_length:
            logger.warning(
                "--pad-to-max-length is redundant under --length-bucket-batch "
                "(--max-length is already the implicit final bucket)"
            )
        if args.backend == "tilelang":
            for value in (*args.pad_to_buckets, args.max_length):
                if value % args.chunk_len != 0:
                    raise SystemExit(
                        f"--pad-to-buckets/--max-length value {value} must be a multiple of "
                        f"--chunk-len ({args.chunk_len}) for the tilelang backend"
                    )
    if args.length_bucket_batch:
        sampler = LengthBucketBatchSampler(
            dataset,
            lengths=compute_sft_lengths(dataset),
            batch_size=args.batch_size,
            buckets=args.pad_to_buckets,
            max_length=args.max_length,
            shuffle=True,
            drop_last=args.length_bucket_drop_tail,
            seed=args.seed,
        )
        if len(sampler) == 0:
            raise SystemExit(
                "--length-bucket-batch: no full batches can be formed; "
                "lower --batch-size or add more data"
            )
        batches_per_epoch = len(sampler)
        logger.info(
            "length-bucket sampler: batches_per_epoch=%d drop_tail=%s buckets=%s",
            batches_per_epoch,
            args.length_bucket_drop_tail,
            args.pad_to_buckets,
        )
        if batches_per_epoch < args.num_steps:
            logger.warning(
                "--num-steps (%d) > batches_per_epoch (%d); the bucket sampler will be "
                "iterated multiple times (reseeded each epoch)",
                args.num_steps,
                batches_per_epoch,
            )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
    else:
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
            debug_stats=args.debug_stats,
        ),
        checkpoint_ctx_len=args.max_length,
    )
    optimizer = make_optimizer(
        trainer.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=args.optimizer_eps if args.optimizer_eps is not None else (1e-8 if args.optimizer == "deepspeed_cpu_adam" else 1e-18),
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
        if args.debug_stats:
            _log_forward_debug(step + 1, timing.get("debug", {}))

        opt_start = time.perf_counter()
        if args.debug_finite_checks:
            trainer.check_finite_parameters("before_optimizer")
            trainer.check_finite_gradients("before_optimizer")
        grad_stats = trainer.debug_gradient_stats("grad") if args.debug_stats else {}
        param_before = trainer.debug_parameter_stats("param_before") if args.debug_stats else {}
        opt_grad_norm = optimizer.step()
        if args.debug_finite_checks:
            trainer.check_finite_parameters("after_optimizer")
        if args.debug_stats:
            param_after = trainer.debug_parameter_stats("param_after")
            _log_optimizer_debug(step + 1, grad_stats, param_before, param_after, opt_grad_norm)
        t_opt = time.perf_counter() - opt_start

        zero_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        trainer.zero_grad()
        t_zero = time.perf_counter() - zero_start

        t_step = time.perf_counter() - step_start

        valid_targets = int((batch["labels"][:, 1:] != -100).sum().item())
        util = 100.0 * valid_targets / total_tokens if total_tokens else 0.0
        logger.info(
            "step=%d loss=%.4f tokens=%d valid_targets=%d util=%.1f%% time=%.3fs fwd=%.3fs bwd=%.3fs opt=%.3fs act=%.4f GB/sample",
            step + 1,
            loss,
            total_tokens,
            valid_targets,
            util,
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
