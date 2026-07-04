from __future__ import annotations

import torch

from rwkv7_core import RWKV7, RWKV7Config, count_parameters, make_optimizer


def main() -> None:
    torch.manual_seed(42)
    config = RWKV7Config(
        vocab_size=128,
        n_layer=2,
        n_embd=64,
        ctx_len=32,
        head_size=16,
        lora_rank_style="official",
    )
    model = RWKV7(config)
    x = torch.randint(0, config.vocab_size, (2, 17))
    y = model(x)
    loss = torch.nn.functional.cross_entropy(y[:, :-1].reshape(-1, config.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.1)
    opt.step()
    print("full_minimal logits", tuple(y.shape))
    print("full_minimal loss", float(loss.detach()))
    print("full_minimal params", count_parameters(model.parameters()))
    print("optimizer groups", [len(g["params"]) for g in opt.param_groups], [g["weight_decay"] for g in opt.param_groups])


if __name__ == "__main__":
    main()
