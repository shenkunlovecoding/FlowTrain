from __future__ import annotations

import torch

from flowtrain import FlowTrainConfig, RWKV7, RWKV7ActivationStore, RWKV7Config, make_optimizer


def test_public_rwkv7_torch_ref_forward_backward():
    config = RWKV7Config(
        vocab_size=12,
        n_layer=2,
        n_embd=64,
        ctx_len=16,
        lora_rank_style="simplified",
        backend="torch_ref",
    )
    model = RWKV7(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(tokens)
    loss = logits.float().mean()
    loss.backward()
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert sum(param.grad is not None for param in model.parameters()) > 0


def test_make_optimizer_returns_adamw():
    model = RWKV7(
        RWKV7Config(
            vocab_size=12,
            n_layer=1,
            n_embd=64,
            ctx_len=8,
            lora_rank_style="simplified",
            backend="torch_ref",
        )
    )
    optimizer = make_optimizer(model)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_flowtrain_config_rejects_int8_without_cpu_offload():
    try:
        FlowTrainConfig(activation_offload="none", activation_quant="int8")
    except ValueError as exc:
        assert "requires activation_offload='cpu'" in str(exc)
    else:
        raise AssertionError("expected invalid activation config to raise")


def test_activation_store_int8_roundtrip_cpu():
    store = RWKV7ActivationStore(offload="cpu", quant="int8")
    hidden = torch.randn(2, 4, 64, dtype=torch.bfloat16)
    v_first = torch.randn(2, 4, 64, dtype=torch.bfloat16)
    store.save_v_first(v_first)
    checkpoint = store.checkpoint(hidden, has_v_first=True)
    hidden_out, v_first_out = store.unpack(checkpoint, torch.device("cpu"))
    assert hidden_out.shape == hidden.shape
    assert v_first_out.shape == v_first.shape
    assert hidden_out.dtype == torch.bfloat16
    assert v_first_out.dtype == torch.bfloat16


if __name__ == "__main__":
    test_public_rwkv7_torch_ref_forward_backward()
    test_make_optimizer_returns_adamw()
    test_flowtrain_config_rejects_int8_without_cpu_offload()
    test_activation_store_int8_roundtrip_cpu()
    print("smoke ok")
