from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowtrain import FlowTrainConfig, FlowTrainTrainer, RWKV7, RWKV7Config, load_rwkv7_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FlowTrain RWKV-7 activation memory")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-len", type=int, default=16)
    return parser.parse_args()


def make_model(args: argparse.Namespace, backend: str) -> RWKV7:
    if args.checkpoint is not None:
        return load_rwkv7_checkpoint(
            args.checkpoint,
            ctx_len=args.ctx_len,
            backend=backend,
            chunk_len=args.chunk_len,
        )
    return RWKV7(
        RWKV7Config(
            vocab_size=args.vocab_size,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            ctx_len=args.ctx_len,
            head_size=64,
            lora_rank_style="simplified" if args.n_embd <= 128 else "official",
            backend=backend,
            chunk_len=args.chunk_len,
        )
    )


def measure_direct(model: RWKV7, batch_size: int, seq_len: int, device: torch.device) -> tuple[float, float]:
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, model.config.vocab_size).float(), labels[:, 1:].reshape(-1))
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    return peak / batch_size / 1024**3, elapsed


def measure_flowtrain(
    model: RWKV7,
    batch_size: int,
    seq_len: int,
    device_id: int,
    activation_quant: str,
    activation_strategy: str = "recompute",
) -> tuple[float, float]:
    trainer = FlowTrainTrainer(
        model,
        FlowTrainConfig(
            device=device_id,
            backend="tilelang",
            chunk_len=model.config.chunk_len,
            checkpoint_interval=1,
            activation_offload="cpu",
            activation_quant=activation_quant,
            activation_strategy=activation_strategy,
        ),
    )
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device="cpu")
    labels = input_ids.clone()
    _, _, timing = trainer.forward_and_backward(input_ids, labels)
    return timing["activation_bytes_per_sample"] / 1024**3, timing["total"]


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")

    rows = []
    for batch_size in args.batch_sizes:
        torch_ref_model = make_model(args, "torch_ref")
        rows.append(("torch_ref", batch_size, *measure_direct(torch_ref_model, batch_size, args.seq_len, device)))
        del torch_ref_model

        tilelang_model = make_model(args, "tilelang")
        rows.append(("tilelang", batch_size, *measure_direct(tilelang_model, batch_size, args.seq_len, device)))
        del tilelang_model

        singleton_model = make_model(args, "tilelang")
        rows.append(("tilelang+singleton", batch_size, *measure_flowtrain(singleton_model, batch_size, args.seq_len, args.device, "none")))
        del singleton_model

        int8_model = make_model(args, "tilelang")
        rows.append(("tilelang+singleton+int8", batch_size, *measure_flowtrain(int8_model, batch_size, args.seq_len, args.device, "int8")))
        del int8_model

        stored_model = make_model(args, "tilelang")
        rows.append(
            (
                "tilelang+store_layer_inputs+int8",
                batch_size,
                *measure_flowtrain(stored_model, batch_size, args.seq_len, args.device, "int8", "store_layer_inputs"),
            )
        )
        del stored_model

    print("mode,batch_size,gb_per_sample,seconds")
    for mode, batch_size, gb_per_sample, seconds in rows:
        print(f"{mode},{batch_size},{gb_per_sample:.6f},{seconds:.4f}")


if __name__ == "__main__":
    main()
