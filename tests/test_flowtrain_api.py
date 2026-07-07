from __future__ import annotations

import torch

from flowtrain import (
    CPU8bitAdamW,
    CPUAdamW,
    CPUQRMuon,
    DeepSpeedCPUAdamW,
    FlowTrainConfig,
    IGNORE_INDEX,
    RWKV7,
    RWKV7ActivationStore,
    RWKV7Config,
    RWKVTokenizerAdapter,
    SFTDataCollator,
    SFTJsonlDataset,
    make_optimizer,
)
from flowtrain.cpu_accum import accumulate_grad_slab_
from flowtrain.cpu_adamw8bit import _load_extension as _load_adamw8bit_extension
from flowtrain.copy_ops import batched_copy_
from flowtrain.estimator import RWKV7Size, estimate_rwkv7_batch_size, rwkv7_param_breakdown


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(ch) % 47 + 2 for ch in text]


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


def test_deepspeed_cpu_adam_uses_fp32_master_for_bf16_params():
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        return

    param = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    optimizer = DeepSpeedCPUAdamW([param], lr=0.1, weight_decay=0.0)
    assert optimizer.original_to_master[param].dtype == torch.float32
    assert optimizer.optimizer.param_groups[0]["params"][0] is optimizer.original_to_master[param]
    param.grad = torch.ones_like(param)
    optimizer.step()
    assert param.dtype == torch.bfloat16
    assert torch.isfinite(param).all()
    assert not torch.equal(param.detach(), torch.ones_like(param))


def test_make_optimizer_returns_cpu_adamw8bit():
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

    # Force int8 state on every param so the assertion is independent of model size.
    optimizer = make_optimizer(model, optimizer="adamw8bit", min_quantized_numel=0)
    assert isinstance(optimizer, CPU8bitAdamW)

    head_param = model.head.weight
    head_param.grad = torch.ones_like(head_param)
    before = head_param.detach().clone()
    optimizer.step()
    assert not torch.equal(before, head_param.detach())

    # Large-param path produced int8 quantized state with an fp32 master.
    state = optimizer.state[head_param]
    assert state["q_exp_avg"].dtype == torch.int8
    assert state["q_exp_avg_sq"].dtype == torch.int8
    assert state["exp_avg_scale"].dtype == torch.float32
    assert state["master"].dtype == torch.float32
    assert "exp_avg" not in state

    # Small-param fallback: a threshold above every param keeps fp32 state.
    tiny_opt = make_optimizer(model, optimizer="adamw8bit", min_quantized_numel=10**9)
    assert isinstance(tiny_opt, CPU8bitAdamW)
    model.zero_grad()
    head_param.grad = torch.ones_like(head_param)
    tiny_opt.step()
    tiny_state = tiny_opt.state[head_param]
    assert tiny_state["exp_avg"].dtype == torch.float32
    assert tiny_state["exp_avg_sq"].dtype == torch.float32
    assert "q_exp_avg" not in tiny_state


def test_cpu_adamw8bit_debug_checks_negative_second_moment():
    import pytest

    param = torch.nn.Parameter(torch.ones(8))
    optimizer = CPU8bitAdamW(
        [{"params": [param], "names": ["weight"]}],
        lr=1e-3,
        min_quantized_numel=0,
        block_size=8,
        debug_finite_checks=True,
    )
    param.grad = torch.ones_like(param)
    optimizer.step()

    state = optimizer.state[param]
    state["q_exp_avg_sq"].fill_(-1)
    state["exp_avg_sq_scale"].fill_(1.0)
    param.grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="negative second moment"):
        optimizer.step()


def test_cpu_adamw8bit_keeps_positive_second_moment_nonzero():
    param = torch.nn.Parameter(torch.zeros(256))
    optimizer = CPU8bitAdamW([param], lr=0.0, min_quantized_numel=0, block_size=256)
    grad = torch.zeros_like(param)
    grad[0] = 1e-3
    grad[1] = 1.0
    param.grad = grad
    optimizer.step()

    state = optimizer.state[param]
    q_exp_avg_sq = state["q_exp_avg_sq"].view(-1)
    assert q_exp_avg_sq[0].item() > 0
    assert q_exp_avg_sq[1].item() > 0
    assert q_exp_avg_sq[2:].abs().sum().item() == 0


def test_cpu_adamw8bit_cpp_matches_python_fallback():
    import os
    import pytest

    original_disable = os.environ.get("FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP")
    os.environ.pop("FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP", None)
    _load_adamw8bit_extension.cache_clear()
    if _load_adamw8bit_extension() is None:
        pytest.skip("CPU AdamW8bit C++ extension is unavailable")

    torch.manual_seed(0)
    initial = torch.randn(1031, dtype=torch.float32)
    grads = [torch.randn_like(initial) for _ in range(3)]

    def run(*, disable_cpp: bool):
        if disable_cpp:
            os.environ["FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP"] = "1"
        else:
            os.environ.pop("FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP", None)
        _load_adamw8bit_extension.cache_clear()

        param = torch.nn.Parameter(initial.clone())
        optimizer = CPU8bitAdamW(
            [param],
            lr=1e-3,
            weight_decay=0.01,
            min_quantized_numel=0,
            block_size=256,
        )
        for grad in grads:
            param.grad = grad.clone()
            optimizer.step()
        state = optimizer.state[param]
        return (
            param.detach().clone(),
            state["master"].clone(),
            state["q_exp_avg"].view(-1).clone(),
            state["exp_avg_scale"].clone(),
            state["q_exp_avg_sq"].view(-1).clone(),
            state["exp_avg_sq_scale"].clone(),
        )

    try:
        cpp = run(disable_cpp=False)
        fallback = run(disable_cpp=True)
    finally:
        if original_disable is None:
            os.environ.pop("FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP", None)
        else:
            os.environ["FLOWTRAIN_DISABLE_CPU_ADAMW8BIT_CPP"] = original_disable
        _load_adamw8bit_extension.cache_clear()

    torch.testing.assert_close(cpp[0], fallback[0], rtol=0, atol=3e-7)
    torch.testing.assert_close(cpp[1], fallback[1], rtol=0, atol=3e-7)
    assert torch.equal(cpp[2], fallback[2])
    torch.testing.assert_close(cpp[3], fallback[3], rtol=0, atol=5e-11)
    assert torch.equal(cpp[4], fallback[4])
    torch.testing.assert_close(cpp[5], fallback[5], rtol=0, atol=5e-11)


def test_batched_copy_cpu_roundtrip():
    destinations = [torch.zeros(2), torch.zeros(3)]
    sources = [torch.ones(2), torch.arange(3, dtype=torch.float32)]
    batched_copy_(destinations, sources)
    assert torch.equal(destinations[0], sources[0])
    assert torch.equal(destinations[1], sources[1])


def test_accumulate_grad_slab_python_fallback():
    grads = [torch.zeros(2, 2), torch.ones(3)]
    slab = torch.arange(7, dtype=torch.bfloat16)
    accumulate_grad_slab_(grads, slab, [4, 3], use_cpp=False)
    assert torch.equal(grads[0], torch.arange(4, dtype=torch.float32).view(2, 2))
    assert torch.equal(grads[1], torch.ones(3) + torch.arange(4, 7, dtype=torch.float32))


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


def test_activation_store_int8_scale_stays_finite_for_large_values():
    store = RWKV7ActivationStore(offload="cpu", quant="int8")
    hidden = torch.full((1, 1, 64), 1.0e8, dtype=torch.bfloat16)
    checkpoint = store.checkpoint(hidden, has_v_first=False)
    assert checkpoint.hidden.scale is not None
    assert checkpoint.hidden.scale.dtype == torch.float32
    hidden_out, _ = store.unpack(checkpoint, torch.device("cpu"))
    assert torch.isfinite(hidden_out).all()


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


def test_sft_jsonl_dataset_masks_prompt_and_collates(tmp_path):
    import pytest

    path = tmp_path / "sft.jsonl"
    path.write_text(
        '{"prompt":"ab","completion":"cd"}\n'
        '{"input_ids":[5,6,7],"labels":[-100,6,7]}\n',
        encoding="utf-8",
    )
    tokenizer = _DummyTokenizer()
    dataset = SFTJsonlDataset(path, tokenizer, max_length=16, mask_prompt=True)

    first = dataset[0]
    prompt_len = len(tokenizer.encode("ab", add_special_tokens=False))
    assert first["labels"][:prompt_len].tolist() == [IGNORE_INDEX] * prompt_len
    assert first["labels"][prompt_len:].tolist() == first["input_ids"][prompt_len:].tolist()
    assert first["input_ids"][-1].item() == tokenizer.eos_token_id

    second = dataset[1]
    assert second["labels"].tolist() == [IGNORE_INDEX, 6, 7]

    batch = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)([first, second])
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"][1, -1].item() == tokenizer.pad_token_id
    assert batch["labels"][1, -1].item() == IGNORE_INDEX
    next_token_targets = batch["labels"][:, 1:]
    current_tokens = batch["input_ids"][:, :-1]
    assert not torch.equal(next_token_targets, current_tokens)
    assert next_token_targets[1][next_token_targets[1] != IGNORE_INDEX].tolist() == [6, 7]

    fixed_batch = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=16,
        pad_to_multiple_of=8,
        pad_to_max_length=True,
    )([first, second])
    assert fixed_batch["input_ids"].shape == (2, 16)
    assert fixed_batch["labels"].shape == (2, 16)

    bucket_128 = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=512,
        pad_to_buckets=(128, 256, 512),
    )([{"input_ids": torch.ones(128), "labels": torch.ones(128)}])
    assert bucket_128["input_ids"].shape == (1, 128)
    assert bucket_128["labels"].shape == (1, 128)

    bucket_256 = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=512,
        pad_to_buckets=(128, 256, 512),
    )([{"input_ids": torch.ones(129), "labels": torch.ones(129)}])
    assert bucket_256["input_ids"].shape == (1, 256)

    bucket_512 = SFTDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=512,
        pad_to_buckets=(128, 256, 512),
    )([{"input_ids": torch.ones(257), "labels": torch.ones(257)}])
    assert bucket_512["input_ids"].shape == (1, 512)

    with pytest.raises(ValueError, match="strictly increasing"):
        SFTDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            pad_to_buckets=(256, 128),
        )([{"input_ids": torch.ones(12), "labels": torch.ones(12)}])


def test_rwkv_tokenizer_adapter_matches_sft_interface():
    try:
        import pyrwkv_tokenizer  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("pyrwkv_tokenizer not installed")

    adapter = RWKVTokenizerAdapter()

    # _tokenize() calls encode(text, add_special_tokens=False); the adapter must
    # accept that kwarg even though the underlying tokenizer has no notion of it.
    ids = adapter.encode("User: hi\n\nAssistant:", add_special_tokens=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert len(ids) > 0

    # decode is the inverse of encode (round-trips through the RWKV tokenizer).
    assert adapter.decode(ids) == "User: hi\n\nAssistant:"

    assert len(adapter) == 65536
    assert isinstance(adapter.pad_token_id, int)
    assert adapter.pad_token_id == 0
    assert adapter.eos_token_id is None


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


def test_tilelang_time_mix_shift_backward_matches_torch_reference():
    if not torch.cuda.is_available():
        return

    from flowtrain.tilelang_time_mix import time_mix_shift_tilelang

    batch, timesteps, channels = 2, 17, 128
    torch.manual_seed(2)

    def leaf(*shape):
        return (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous().requires_grad_(True)

    x = leaf(batch, timesteps, channels)
    params = tuple(leaf(1, 1, channels) for _ in range(6))
    x_ref = x.detach().clone().requires_grad_(True)
    params_ref = tuple(param.detach().clone().requires_grad_(True) for param in params)
    grad_outs = tuple(
        (torch.randn(batch, timesteps, channels, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
        for _ in range(6)
    )

    outs = time_mix_shift_tilelang(x, *params)
    loss = sum((out.float() * grad.float()).sum() for out, grad in zip(outs, grad_outs, strict=True))
    loss.backward()

    shifted = torch.nn.ZeroPad2d((0, 0, 1, -1))(x_ref) - x_ref
    refs = tuple((x_ref + shifted * param).bfloat16() for param in params_ref)
    ref_loss = sum((out.float() * grad.float()).sum() for out, grad in zip(refs, grad_outs, strict=True))
    ref_loss.backward()
    torch.cuda.synchronize()

    rel_x = (x.grad.float() - x_ref.grad.float()).norm().item() / max(x_ref.grad.float().norm().item(), 1e-9)
    assert rel_x < 0.03
    for grad, ref_grad in zip((param.grad for param in params), (param.grad for param in params_ref), strict=True):
        rel = (grad.float() - ref_grad.float()).norm().item() / max(ref_grad.float().norm().item(), 1e-9)
        assert rel < 0.03


if __name__ == "__main__":
    test_public_rwkv7_torch_ref_forward_backward()
    test_make_optimizer_returns_cpu_adamw()
    test_make_optimizer_returns_cpu_qr_muon()
    test_make_optimizer_returns_deepspeed_cpu_adam_when_available()
    test_make_optimizer_returns_cpu_adamw8bit()
    test_batched_copy_cpu_roundtrip()
    test_accumulate_grad_slab_python_fallback()
    test_flowtrain_config_rejects_int8_without_cpu_offload()
    test_flowtrain_config_accepts_store_layer_inputs()
    test_activation_store_int8_roundtrip_cpu()
    test_estimator_reports_positive_batch_for_tiny_model()
    test_tilelang_time_mix_post_matches_torch_reference()
    test_tilelang_time_mix_shift_matches_torch_reference()
    test_tilelang_time_mix_shift_backward_matches_torch_reference()
    print("smoke ok")
