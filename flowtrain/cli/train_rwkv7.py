from __future__ import annotations

import argparse
import logging
import random

import torch

from flowtrain import FlowTrainConfig, FlowTrainTrainer, RWKV7, RWKV7Config, make_optimizer

TOK = {**{str(i): i for i in range(10)}, ",": 10, "#": 11}
DIGIT_MAX = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _digits(value: int) -> list[int]:
    return [TOK[ch] for ch in str(value)]


def digit_batch(batch_size: int, seq_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
    rows = []
    for _ in range(batch_size):
        row = []
        while len(row) < seq_len:
            k = random.randint(1, DIGIT_MAX)
            lo = 0 if k == 1 else 10 ** (k - 1)
            n = random.randint(lo, 10**k - 1)
            digits = _digits(n)
            row += digits + [TOK[","]] + digits[::-1] + [TOK["#"]]
        rows.append(row[:seq_len])
    return torch.tensor(rows, device=device, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RWKV-7 with FlowTrain")
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
    parser.add_argument(
        "--debug-finite-checks",
        action="store_true",
        help="Raise at the first non-finite loss/logit/gradient/CPU-master parameter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rwkv_config = RWKV7Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        ctx_len=args.ctx_len,
        head_size=args.head_size,
        dim_ffn=args.dim_ffn,
        lora_rank_style="simplified" if args.n_embd <= 128 else "official",
        backend=args.backend,
        chunk_len=args.chunk_len,
    )
    model = RWKV7(rwkv_config)
    trainer = FlowTrainTrainer(
        model,
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
    )
    del model

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
        "FlowTrain RWKV7: layers=%d emb=%d backend=%s optimizer=%s chunk=%d logits=%d activation=%s/%s/%s",
        args.n_layer,
        args.n_embd,
        args.backend,
        args.optimizer,
        args.chunk_len,
        args.logit_chunk_size,
        args.activation_offload,
        args.activation_quant,
        args.activation_strategy,
    )

    for step in range(args.num_steps):
        input_ids = digit_batch(args.batch_size, args.seq_len)
        labels = input_ids.clone()

        loss, total_tokens, timing = trainer.forward_and_backward(input_ids, labels)
        if args.debug_finite_checks:
            trainer.check_finite_parameters("before_optimizer")
            trainer.check_finite_gradients("before_optimizer")
        optimizer.step()
        if args.debug_finite_checks:
            trainer.check_finite_parameters("after_optimizer")
        optimizer.zero_grad(set_to_none=True)
        trainer.zero_grad()

        logger.info(
            "step=%d loss=%.4f tokens=%d time=%.3fs fwd=%.3fs bwd=%.3fs act=%.4f GB/sample",
            step + 1,
            loss,
            total_tokens,
            timing["total"],
            timing["forward"],
            timing["backward"],
            timing["activation_bytes_per_sample"] / 1024**3,
        )


if __name__ == "__main__":
    main()
