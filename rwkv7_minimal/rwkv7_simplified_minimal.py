from __future__ import annotations

import random

import torch
import torch.nn.functional as F

try:
    from .rwkv7_core import RWKV7, RWKV7Config, make_optimizer
except ImportError:
    from rwkv7_core import RWKV7, RWKV7Config, make_optimizer


TOK = {**{str(i): i for i in range(10)}, ",": 10, "#": 11}
DIGIT_MAX = 20


def _digits(n: int) -> list[int]:
    return [TOK[c] for c in str(n)]


def batch(batch_size: int, seq_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
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


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    config = RWKV7Config(
        vocab_size=12,
        n_layer=2,
        n_embd=32,
        ctx_len=128,
        head_size=16,
        dim_ffn=32 * 4,
        lora_rank_style="simplified",
    )
    model = RWKV7(config)
    opt = make_optimizer(model, lr=4e-3, weight_decay=0.1)

    x = batch(4, 33)
    targets = x[:, 1:]
    logits = model(x[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    print("simplified logits", tuple(logits.shape))
    print("simplified loss", float(loss.detach()))


if __name__ == "__main__":
    main()
