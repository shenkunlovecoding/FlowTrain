from __future__ import annotations

import torch

from flowtrain import CPUAdamW, CPUQRMuon, DeepSpeedCPUAdamW, FlowTrainConfig, RWKV7, RWKV7ActivationStore, RWKV7Config, make_optimizer
from flowtrain.copy_ops import batched_copy_
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


def test_make_optimizer_returns_deepspeed_cpu_adam_when_available():
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        return

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
    optimizer = make_optimizer(model, optimizer="deepspeed_cpu_adam")
    assert isinstance(optimizer, DeepSpeedCPUAdamW)
    before = model.head.weight.detach().clone()
    loss = model(torch.randint(0, 12, (1, 4))).float().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head.weight.detach())


def test_batched_copy_cpu_roundtrip():
    destinations = [torch.zeros(2), torch.zeros(3)]
    sources = [torch.ones(2), torch.arange(3, dtype=torch.float32)]
    batched_copy_(destinations, sources)
    assert torch.equal(destinations[0], sources[0])
    assert torch.equal(destinations[1], sources[1])


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
    deepspeed_adam = estimate_rwkv7_batch_size(
        size,
        seq_len=32,
        gpu_gb=8,
        cpu_gb=32,
        optimizer="deepspeed_cpu_adam",
    )
    assert breakdown.total_params > 0
    assert estimate.max_batch_size > 0
    assert estimate.gpu_per_sample_gb > 0
    assert store_inputs.cpu_per_sample_gb > estimate.cpu_per_sample_gb
    assert qr_muon.cpu_base_gb != estimate.cpu_base_gb
    assert deepspeed_adam.cpu_base_gb == estimate.cpu_base_gb


def test_tilelang_time_mix_post_matches_torch_reference():
    if not torch.cuda.is_available():
        return

    from flowtrain.tilelang_time_mix import time_mix_post_tilelang

    batch, timesteps, heads, head_size = 1, 16, 2, 64
    channels = heads * head_size
    torch.manual_seed(0)

    def randn(*shape):
        return (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()

    recurrence = randn(batch, timesteps, channels)
    r = randn(batch, timesteps, channels)
    k = randn(batch, timesteps, channels)
    v = randn(batch, timesteps, channels)
    g = randn(batch, timesteps, channels)
    r_k = randn(heads, head_size)
    ln_weight = (torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1 + 1).contiguous()
    ln_bias = randn(channels)

    out = time_mix_post_tilelang(recurrence, r, k, v, g, r_k, ln_weight, ln_bias, head_size, 64e-5)
    ref = torch.nn.functional.group_norm(
        recurrence.view(batch * timesteps, channels),
        heads,
        ln_weight,
        ln_bias,
        eps=64e-5,
    ).view(batch, timesteps, channels)
    bonus = (
        r.view(batch, timesteps, heads, head_size).float()
        * k.view(batch, timesteps, heads, head_size).float()
        * r_k.float()
    ).sum(dim=-1, keepdim=True).to(dtype=recurrence.dtype) * v.view(batch, timesteps, heads, head_size)
    ref = ((ref + bonus.view(batch, timesteps, channels)) * g).bfloat16()

    torch.cuda.synchronize()
    rel = (out.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-9)
    assert rel < 0.01


def test_tilelang_time_mix_shift_matches_torch_reference():
    if not torch.cuda.is_available():
        return

    from flowtrain.tilelang_time_mix import time_mix_shift_tilelang

    batch, timesteps, channels = 2, 17, 128
    torch.manual_seed(1)
    x = (torch.randn(batch, timesteps, channels, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    params = tuple(
        (torch.randn(1, 1, channels, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        for _ in range(6)
    )

    outs = time_mix_shift_tilelang(x, *params)
    shifted = torch.nn.ZeroPad2d((0, 0, 1, -1))(x) - x
    refs = tuple((x + shifted * param).bfloat16() for param in params)
    torch.cuda.synchronize()

    for out, ref in zip(outs, refs, strict=True):
        rel = (out.float() - ref.float()).norm().item() / max(ref.float().norm().item(), 1e-9)
        assert rel < 0.01


if __name__ == "__main__":
    test_public_rwkv7_torch_ref_forward_backward()
    test_make_optimizer_returns_cpu_adamw()
    test_make_optimizer_returns_cpu_qr_muon()
    test_make_optimizer_returns_deepspeed_cpu_adam_when_available()
    test_batched_copy_cpu_roundtrip()
    test_flowtrain_config_rejects_int8_without_cpu_offload()
    test_flowtrain_config_accepts_store_layer_inputs()
    test_activation_store_int8_roundtrip_cpu()
    test_estimator_reports_positive_batch_for_tiny_model()
    test_tilelang_time_mix_post_matches_torch_reference()
    test_tilelang_time_mix_shift_matches_torch_reference()
    print("smoke ok")
