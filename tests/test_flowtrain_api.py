from __future__ import annotations

import torch

from flowtrain import CPUAdamW, CPUQRMuon, FlowTrainConfig, RWKV7, RWKV7ActivationStore, RWKV7Config, make_optimizer
from flowtrain.estimator import RWKV7Size, estimate_rwkv7_batch_size, rwkv7_param_breakdown


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


def test_make_optimizer_returns_cpu_adamw():
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
    assert isinstance(optimizer, CPUAdamW)
    before = model.head.weight.detach().clone()
    loss = model(torch.randint(0, 12, (1, 4))).float().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head.weight.detach())


def test_make_optimizer_returns_cpu_qr_muon():
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
    optimizer = make_optimizer(model, optimizer="qr_muon")
    assert isinstance(optimizer, CPUQRMuon)
    assert any(group.get("use_muon") for group in optimizer.param_groups)
    assert any(not group.get("use_muon") for group in optimizer.param_groups)

    target = next(group["params"][0] for group in optimizer.param_groups if group.get("use_muon") and group["params"])
    before = target.detach().clone()
    target.grad = torch.ones_like(target)
    optimizer.step()
    assert not torch.equal(before, target.detach())


def test_flowtrain_config_rejects_int8_without_cpu_offload():
    try:
        FlowTrainConfig(activation_offload="none", activation_quant="int8")
    except ValueError as exc:
        assert "requires activation_offload='cpu'" in str(exc)
    else:
        raise AssertionError("expected invalid activation config to raise")


def test_flowtrain_config_accepts_store_layer_inputs():
    config = FlowTrainConfig(activation_strategy="store_layer_inputs")
    assert config.activation_strategy == "store_layer_inputs"


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


def test_estimator_reports_positive_batch_for_tiny_model():
    size = RWKV7Size(vocab_size=12, n_layer=2, n_embd=64, dim_ffn=224, lora_rank_style="simplified")
    breakdown = rwkv7_param_breakdown(size)
    estimate = estimate_rwkv7_batch_size(size, seq_len=32, gpu_gb=8, cpu_gb=32)
    store_inputs = estimate_rwkv7_batch_size(
        size,
        seq_len=32,
        gpu_gb=8,
        cpu_gb=32,
        activation_strategy="store_layer_inputs",
    )
    qr_muon = estimate_rwkv7_batch_size(
        size,
        seq_len=32,
        gpu_gb=8,
        cpu_gb=32,
        optimizer="qr_muon",
    )
    assert breakdown.total_params > 0
    assert estimate.max_batch_size > 0
    assert estimate.gpu_per_sample_gb > 0
    assert store_inputs.cpu_per_sample_gb > estimate.cpu_per_sample_gb
    assert qr_muon.cpu_base_gb != estimate.cpu_base_gb


if __name__ == "__main__":
    test_public_rwkv7_torch_ref_forward_backward()
    test_make_optimizer_returns_cpu_adamw()
    test_make_optimizer_returns_cpu_qr_muon()
    test_flowtrain_config_rejects_int8_without_cpu_offload()
    test_flowtrain_config_accepts_store_layer_inputs()
    test_activation_store_int8_roundtrip_cpu()
    test_estimator_reports_positive_batch_for_tiny_model()
    print("smoke ok")
