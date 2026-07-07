from __future__ import annotations

import copy
import concurrent.futures
import logging
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_store import ActivationOffload, ActivationQuant, RWKV7ActivationStore
from .copy_ops import batched_copy_
from .cpu_accum import accumulate_grad_slab_
from .rwkv7 import RWKV7, RWKV7Config, lora_ranks

logger = logging.getLogger(__name__)


@dataclass
class FlowTrainConfig:
    device: int | str = 0
    dtype: torch.dtype = torch.bfloat16
    backend: Literal["tilelang", "torch_ref"] = "tilelang"
    chunk_len: int = 16
    checkpoint_interval: int = 4
    num_grad_slabs: int = 12
    activation_offload: ActivationOffload = "cpu"
    activation_quant: ActivationQuant = "none"
    activation_strategy: Literal["recompute", "store_layer_inputs"] = "recompute"
    max_grad_norm: float | None = 1.0
    logit_chunk_size: int = 128
    debug_finite_checks: bool = False
    debug_stats: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("tilelang", "torch_ref"):
            raise ValueError("backend must be 'tilelang' or 'torch_ref'")
        if self.dtype != torch.bfloat16:
            raise ValueError("FlowTrain v1 supports bf16 training only")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        if self.num_grad_slabs < 1:
            raise ValueError("num_grad_slabs must be >= 1")
        if self.logit_chunk_size < 1:
            raise ValueError("logit_chunk_size must be >= 1")
        if self.activation_offload not in ("none", "cpu"):
            raise ValueError("activation_offload must be 'none' or 'cpu'")
        if self.activation_quant not in ("none", "int8"):
            raise ValueError("activation_quant must be 'none' or 'int8'")
        if self.activation_quant == "int8" and self.activation_offload != "cpu":
            raise ValueError("activation_quant='int8' requires activation_offload='cpu'")
        if self.activation_strategy not in ("recompute", "store_layer_inputs"):
            raise ValueError("activation_strategy must be 'recompute' or 'store_layer_inputs'")


def _device(value: int | str) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    if value == "cpu":
        return torch.device("cpu")
    if value.startswith("cuda"):
        return torch.device(value)
    return torch.device(f"cuda:{value}")


def _pin_empty(numel: int, dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty(numel, dtype=dtype, device="cpu")


def _raise_if_nonfinite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    finite_mask = torch.isfinite(tensor)
    bad = int((~finite_mask).sum().item())
    finite = tensor.detach()[finite_mask]
    if finite.numel():
        min_val = float(finite.min().item())
        max_val = float(finite.max().item())
        range_text = f" finite_min={min_val:.6g} finite_max={max_val:.6g}"
    else:
        range_text = " no_finite_values"
    raise RuntimeError(
        f"non-finite tensor: {name} bad={bad} shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device}{range_text}"
    )


def _tensor_debug_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    bad = int((~finite).sum().item())
    if not bool(finite.any().item()):
        return {
            f"{prefix}_numel": float(data.numel()),
            f"{prefix}_bad": float(bad),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_absmax": float("nan"),
        }
    values = data[finite]
    return {
        f"{prefix}_numel": float(data.numel()),
        f"{prefix}_bad": float(bad),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_rms": float(values.square().mean().sqrt().item()),
        f"{prefix}_absmax": float(values.abs().max().item()),
    }


def _module_tensor_debug_stats(prefix: str, tensors: list[torch.Tensor]) -> dict[str, float]:
    total_sq = 0.0
    total_numel = 0
    max_abs = 0.0
    bad = 0
    for tensor in tensors:
        data = tensor.detach().float()
        finite = torch.isfinite(data)
        bad += int((~finite).sum().item())
        if not bool(finite.any().item()):
            total_numel += data.numel()
            continue
        values = data[finite]
        total_sq += float(values.square().sum().item())
        total_numel += data.numel()
        max_abs = max(max_abs, float(values.abs().max().item()))
    norm = total_sq**0.5
    return {
        f"{prefix}_numel": float(total_numel),
        f"{prefix}_bad": float(bad),
        f"{prefix}_norm": norm,
        f"{prefix}_absmax": max_abs,
    }


def infer_rwkv7_config_from_state(
    state: dict[str, torch.Tensor],
    *,
    ctx_len: int,
    backend: Literal["tilelang", "torch_ref"] = "tilelang",
    chunk_len: int = 16,
) -> RWKV7Config:
    vocab_size, n_embd = state["emb.weight"].shape
    layer_ids = {
        int(name.split(".")[1])
        for name in state
        if name.startswith("blocks.") and name.split(".")[1].isdigit()
    }
    if not layer_ids:
        raise ValueError("checkpoint does not look like an RWKV-7 state_dict: no blocks.* tensors found")
    n_layer = max(layer_ids) + 1
    dim_ffn = state.get("blocks.0.ffn.key.weight")
    if dim_ffn is None:
        raise ValueError("checkpoint does not look like an RWKV-7 state_dict: missing blocks.0.ffn.key.weight")

    # Detect per-layer lora rank overrides. Some RWKV-7 checkpoints (e.g. g1g)
    # use different lora dimensions across layer groups; read the actual shape
    # of each layer's ``att.w1`` (first LoRA projection) to derive overrides.
    lora_rank_overrides: dict[int, tuple[int, int, int, int]] = {}
    default_ranks = lora_ranks(n_embd, "official")
    for lid in sorted(layer_ids):
        w1_key = f"blocks.{lid}.att.w1"
        a1_key = f"blocks.{lid}.att.a1"
        v1_key = f"blocks.{lid}.att.v1"
        g1_key = f"blocks.{lid}.att.g1"
        w1 = state.get(w1_key)
        a1 = state.get(a1_key)
        v1 = state.get(v1_key)
        g1 = state.get(g1_key)
        if w1 is None or a1 is None or v1 is None or g1 is None:
            continue
        ranks = (w1.shape[1], a1.shape[1], v1.shape[1], g1.shape[1])
        if ranks != default_ranks:
            lora_rank_overrides[lid] = ranks
    if not lora_rank_overrides:
        lora_rank_overrides.clear()  # keep None if no overrides needed

    return RWKV7Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        ctx_len=ctx_len,
        head_size=64,
        dim_ffn=dim_ffn.shape[0],
        lora_rank_overrides=lora_rank_overrides if lora_rank_overrides else None,
        backend=backend,
        chunk_len=chunk_len,
    )


def load_rwkv7_checkpoint(
    checkpoint: str | Path,
    *,
    config: RWKV7Config | None = None,
    ctx_len: int = 8192,
    backend: Literal["tilelang", "torch_ref"] = "tilelang",
    chunk_len: int = 16,
) -> RWKV7:
    raw = torch.load(checkpoint, map_location="cpu")
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError("RWKV-7 checkpoint must be a state_dict-like mapping")

    rwkv_config = config or infer_rwkv7_config_from_state(
        state,
        ctx_len=ctx_len,
        backend=backend,
        chunk_len=chunk_len,
    )
    model = RWKV7(rwkv_config)
    model.load_state_dict(state, strict=False)
    return model


class FlowTrainTrainer:
    """RWKV-7-only CPU-master trainer with streamed replay backward."""

    def __init__(
        self,
        model_or_checkpoint: RWKV7 | str | Path,
        config: FlowTrainConfig,
        *,
        rwkv_config: RWKV7Config | None = None,
        checkpoint_ctx_len: int = 8192,
    ):
        self.config = config
        self.device = _device(config.device)
        if self.device.type != "cuda":
            raise ValueError("FlowTrain v1 requires a CUDA device")
        torch.cuda.set_device(self.device)

        if isinstance(model_or_checkpoint, RWKV7):
            model = model_or_checkpoint
        else:
            model = load_rwkv7_checkpoint(
                model_or_checkpoint,
                config=rwkv_config,
                ctx_len=checkpoint_ctx_len,
                backend=config.backend,
                chunk_len=config.chunk_len,
            )

        model.config.backend = config.backend
        model.config.chunk_len = config.chunk_len
        model.config.debug_finite_checks = config.debug_finite_checks
        if model.config.head_size != 64:
            raise ValueError("FlowTrain v1 supports RWKV-7 head_size=64 only")
        self.model = model.cpu()
        self.model.train()

        self.embedding = self.model.emb
        self.layers = list(self.model.blocks)
        self.norm = self.model.ln_out
        self.head = self.model.head
        self.vocab_size = self.model.config.vocab_size

        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.weight_stream = torch.cuda.Stream(device=self.device)
        self.grad_stream = torch.cuda.Stream(device=self.device)
        self.activation_stream = torch.cuda.Stream(device=self.device)
        self.activation_store = RWKV7ActivationStore(config.activation_offload, config.activation_quant)
        self.gpu_layer_templates: list[dict[str, nn.Module]] = [{}, {}]
        self.template_layer_idx: list[dict[str, int | None]] = [{}, {}]
        self.template_busy_events: list[dict[str, torch.cuda.Event | None]] = [{}, {}]
        self.template_ready_events: list[dict[str, torch.cuda.Event]] = [{}, {}]

        self.layer_param_shapes = [[p.shape for p in layer.parameters()] for layer in self.layers]
        self.layer_param_numels = [[p.numel() for p in layer.parameters()] for layer in self.layers]
        self.layer_numels = [sum(numels) for numels in self.layer_param_numels]
        self.max_layer_numel = max(self.layer_numels) if self.layer_numels else 0
        self.layer_pinned_flats = [_pin_empty(numel, config.dtype) for numel in self.layer_numels]
        self.layer_flat_dirty = [True for _ in self.layers]
        self.gpu_flat_buffers = [
            torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device),
            torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device),
        ]
        grad_slab_numel = max(
            [self.max_layer_numel, sum(p.numel() for p in self.embedding.parameters()), sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.norm.parameters())],
            default=1,
        )
        self.grad_slabs = [_pin_empty(grad_slab_numel, config.dtype) for _ in range(config.num_grad_slabs)]
        self.grad_slab_events = [torch.cuda.Event(enable_timing=False) for _ in range(config.num_grad_slabs)]
        self.grad_slab_free: queue.SimpleQueue[int] = queue.SimpleQueue()
        for slab_idx in range(config.num_grad_slabs):
            self.grad_slab_free.put(slab_idx)
        self._grad_tasks: list[
            tuple[int, torch.cuda.Event, torch.Tensor, list[torch.nn.Parameter], list[torch.Size], list[int], list[torch.Tensor]]
        ] = []
        self.grad_accum_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowtrain-grad")
        self._grad_accum_futures: list[concurrent.futures.Future] = []

    def parameters(self):
        return self.model.parameters()

    def get_parameters(self):
        return list(self.model.parameters())

    def optimizer_groups(self, weight_decay: float = 0.1) -> list[dict]:
        return self.model.optimizer_groups(weight_decay)

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def check_finite_parameters(self, stage: str) -> None:
        for name, param in self.model.named_parameters():
            _raise_if_nonfinite(f"{stage}:param:{name}", param.data)

    def check_finite_gradients(self, stage: str) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                _raise_if_nonfinite(f"{stage}:grad:{name}", param.grad)

    def debug_parameter_stats(self, prefix: str = "param") -> dict[str, float]:
        return _module_tensor_debug_stats(prefix, [param.data for param in self.model.parameters()])

    def debug_gradient_stats(self, prefix: str = "grad") -> dict[str, float]:
        return _module_tensor_debug_stats(
            prefix,
            [param.grad for param in self.model.parameters() if param.grad is not None],
        )

    def cleanup(self) -> None:
        self._drain_grad_tasks(wait_all=True)
        self.grad_accum_executor.shutdown(wait=True)
        self.gpu_layer_templates = [{}, {}]
        self.template_layer_idx = [{}, {}]
        self.template_busy_events = [{}, {}]
        self.template_ready_events = [{}, {}]
        self.activation_store.clear()
        torch.cuda.empty_cache()

    @property
    def activation_bytes_per_sample(self) -> float:
        return self.activation_store.offloaded_bytes

    def forward_and_backward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        if labels is None:
            labels = input_ids
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")
        if input_ids.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")

        self._drain_grad_tasks(wait_all=True)
        self.activation_store.clear()
        self.zero_grad()
        self.layer_flat_dirty = [True for _ in self.layers]
        start = time.perf_counter()

        with torch.cuda.stream(self.compute_stream):
            input_ids_gpu = input_ids.to(self.device, non_blocking=True)
            labels_gpu = labels.to(self.device, non_blocking=True)
            emb_gpu = self._clone_module_to_gpu(self.embedding)
            hidden = emb_gpu(input_ids_gpu)
            if self.config.debug_finite_checks:
                _raise_if_nonfinite("embedding", hidden)
            v_first = torch.empty_like(hidden)

        checkpoints = {}
        store_layer_inputs = self.config.activation_strategy == "store_layer_inputs"
        if self.layers:
            self._prefetch_layer_async(0, 0)
        with torch.no_grad():
            for layer_idx in range(len(self.layers)):
                buffer_idx = layer_idx % 2
                next_idx = layer_idx + 1
                if store_layer_inputs or layer_idx % self.config.checkpoint_interval == 0:
                    checkpoints[layer_idx] = self._checkpoint_activation(hidden, has_v_first=layer_idx > 0)

                if next_idx < len(self.layers):
                    self._prefetch_layer_async(next_idx, next_idx % 2)

                gpu_layer = self._wait_for_layer(layer_idx, buffer_idx)
                with torch.cuda.stream(self.compute_stream):
                    hidden, v_first = gpu_layer(hidden, v_first)
                    if self.config.debug_finite_checks:
                        _raise_if_nonfinite(f"forward_layer_{layer_idx}:hidden", hidden)
                        _raise_if_nonfinite(f"forward_layer_{layer_idx}:v_first", v_first)
                    self._record_buffer_busy(layer_idx, buffer_idx)
                if layer_idx == 0:
                    self._save_v_first(v_first)

            checkpoints[len(self.layers)] = self._checkpoint_activation(hidden, has_v_first=bool(self.layers))

        fwd_end = time.perf_counter()

        final_hidden, final_v_first = self._unpack_activation(checkpoints[len(self.layers)])
        if self.config.debug_finite_checks:
            with torch.cuda.stream(self.compute_stream):
                _raise_if_nonfinite("final_hidden_unpacked", final_hidden)
                if self.layers:
                    _raise_if_nonfinite("final_v_first_unpacked", final_v_first)
        valid_tokens = int((labels[:, 1:] != -100).sum().item())
        if valid_tokens == 0:
            raise ValueError("labels contain no valid next-token targets")
        debug_stats: dict[str, float] = {}

        with torch.cuda.stream(self.compute_stream):
            hidden_before_norm = final_hidden.detach().requires_grad_(True)
            norm_gpu = self._clone_module_to_gpu(self.norm)
            head_gpu = self._clone_module_to_gpu(self.head)
            hidden_after_norm = norm_gpu(hidden_before_norm)
            if self.config.debug_finite_checks:
                _raise_if_nonfinite("hidden_before_head", hidden_after_norm)
            if self.config.debug_stats:
                debug_stats.update(_tensor_debug_stats("hidden_before_head", hidden_after_norm))
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            logit_min = float("inf")
            logit_max = float("-inf")
            logit_absmax = 0.0
            target_logit_sum = 0.0
            target_logit_min = float("inf")
            target_logit_max = float("-inf")
            target_margin_sum = 0.0
            target_margin_min = float("inf")
            target_margin_max = float("-inf")
            target_gap_sum = 0.0
            target_gap_min = float("inf")
            target_gap_max = float("-inf")
            nll_sum = 0.0
            nll_min = float("inf")
            nll_max = float("-inf")
            debug_valid = 0
            for start_t in range(0, input_ids.shape[1] - 1, self.config.logit_chunk_size):
                end_t = min(start_t + self.config.logit_chunk_size, input_ids.shape[1] - 1)
                logits = head_gpu(hidden_after_norm[:, start_t:end_t, :]).reshape(-1, self.vocab_size).float()
                if self.config.debug_finite_checks:
                    _raise_if_nonfinite(f"logits[{start_t}:{end_t}]", logits)
                targets = labels_gpu[:, start_t + 1 : end_t + 1].reshape(-1)
                chunk_loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
                total_loss = total_loss + chunk_loss
                if self.config.debug_stats:
                    logit_min = min(logit_min, float(logits.min().item()))
                    logit_max = max(logit_max, float(logits.max().item()))
                    logit_absmax = max(logit_absmax, float(logits.abs().max().item()))
                    valid_mask = targets != -100
                    chunk_valid = int(valid_mask.sum().item())
                    if chunk_valid:
                        valid_logits = logits[valid_mask]
                        valid_targets = targets[valid_mask]
                        target_logits = valid_logits.gather(1, valid_targets[:, None]).squeeze(1)
                        max_logits = valid_logits.max(dim=1).values
                        top_count = min(2, valid_logits.shape[1])
                        top_values, top_indices = valid_logits.topk(top_count, dim=1)
                        if top_count == 1:
                            max_other = top_values[:, 0]
                        else:
                            max_other = torch.where(
                                top_indices[:, 0] == valid_targets,
                                top_values[:, 1],
                                top_values[:, 0],
                            )
                        target_margin = target_logits - max_logits
                        target_gap = target_logits - max_other
                        nll = -F.log_softmax(valid_logits, dim=1).gather(1, valid_targets[:, None]).squeeze(1)

                        debug_valid += chunk_valid
                        target_logit_sum += float(target_logits.sum().item())
                        target_logit_min = min(target_logit_min, float(target_logits.min().item()))
                        target_logit_max = max(target_logit_max, float(target_logits.max().item()))
                        target_margin_sum += float(target_margin.sum().item())
                        target_margin_min = min(target_margin_min, float(target_margin.min().item()))
                        target_margin_max = max(target_margin_max, float(target_margin.max().item()))
                        target_gap_sum += float(target_gap.sum().item())
                        target_gap_min = min(target_gap_min, float(target_gap.min().item()))
                        target_gap_max = max(target_gap_max, float(target_gap.max().item()))
                        nll_sum += float(nll.sum().item())
                        nll_min = min(nll_min, float(nll.min().item()))
                        nll_max = max(nll_max, float(nll.max().item()))
                del logits
            loss = total_loss / valid_tokens
            if self.config.debug_finite_checks:
                _raise_if_nonfinite("loss", loss)
            if self.config.debug_stats:
                denom = max(1, debug_valid)
                debug_stats.update(
                    {
                        "valid_tokens": float(valid_tokens),
                        "ce_sum": float(total_loss.detach().item()),
                        "loss": float(loss.detach().item()),
                        "logit_min": logit_min,
                        "logit_max": logit_max,
                        "logit_absmax": logit_absmax,
                        "target_logit_min": target_logit_min,
                        "target_logit_max": target_logit_max,
                        "target_logit_mean": target_logit_sum / denom,
                        "target_margin_min": target_margin_min,
                        "target_margin_max": target_margin_max,
                        "target_margin_mean": target_margin_sum / denom,
                        "target_gap_min": target_gap_min,
                        "target_gap_max": target_gap_max,
                        "target_gap_mean": target_gap_sum / denom,
                        "nll_min": nll_min,
                        "nll_max": nll_max,
                        "nll_mean": nll_sum / denom,
                    }
                )
            loss.backward()
            loss_value_tensor = loss.detach().float()

        grad_hidden = hidden_before_norm.grad.detach()
        grad_v_first = torch.zeros_like(final_v_first)
        self._copy_module_grads_to_cpu_async(head_gpu, self.head)
        self._copy_module_grads_to_cpu_async(norm_gpu, self.norm)
        self._drain_ready_grad_tasks()

        loss_val = float(loss_value_tensor.cpu())
        del hidden_after_norm, hidden_before_norm, final_hidden, final_v_first, total_loss

        if self.config.activation_strategy == "store_layer_inputs":
            for layer_idx in range(len(self.layers) - 1, -1, -1):
                layer_input_base, state_input_base = self._unpack_activation(checkpoints[layer_idx])
                grad_hidden, grad_v_first = self._replay_one_layer_backward(
                    layer_idx,
                    layer_input_base,
                    state_input_base,
                    grad_hidden,
                    grad_v_first,
                )
        else:
            num_blocks = (len(self.layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval
            for block_idx in range(num_blocks - 1, -1, -1):
                block_start = block_idx * self.config.checkpoint_interval
                block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.layers))
                checkpoint_hidden, checkpoint_v_first = self._unpack_activation(checkpoints[block_start])

                recompute_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                hidden_recompute = checkpoint_hidden
                v_first_recompute = checkpoint_v_first
                if block_start < block_end:
                    self._prefetch_layer_async(block_start, block_start % 2)
                with torch.no_grad():
                    for layer_idx in range(block_start, block_end):
                        next_idx = layer_idx + 1
                        if next_idx < block_end:
                            self._prefetch_layer_async(next_idx, next_idx % 2)
                        gpu_layer = self._wait_for_layer(layer_idx, layer_idx % 2)
                        with torch.cuda.stream(self.compute_stream):
                            hidden_recompute, v_first_recompute = gpu_layer(hidden_recompute, v_first_recompute)
                            self._record_buffer_busy(layer_idx, layer_idx % 2)
                        recompute_cache[layer_idx] = (hidden_recompute.detach(), v_first_recompute.detach())

                for layer_idx in range(block_end - 1, block_start - 1, -1):
                    if layer_idx == block_start:
                        layer_input_base = checkpoint_hidden
                        state_input_base = checkpoint_v_first
                    else:
                        layer_input_base, state_input_base = recompute_cache[layer_idx - 1]
                    grad_hidden, grad_v_first = self._replay_one_layer_backward(
                        layer_idx,
                        layer_input_base,
                        state_input_base,
                        grad_hidden,
                        grad_v_first,
                    )

                recompute_cache.clear()

        with torch.cuda.stream(self.compute_stream):
            emb_replay = self._clone_module_to_gpu(self.embedding)
            emb_out = emb_replay(input_ids_gpu)
            emb_out.backward(grad_hidden)
        self._drain_ready_grad_tasks()
        self._copy_module_grads_to_cpu_async(emb_replay, self.embedding)
        self._drain_grad_tasks(wait_all=True)

        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.config.debug_finite_checks:
            self.check_finite_gradients("after_backward")

        torch.cuda.synchronize(self.device)
        bwd_end = time.perf_counter()
        timing: dict[str, Any] = {
            "forward": fwd_end - start,
            "backward": bwd_end - fwd_end,
            "total": bwd_end - start,
            "activation_bytes_per_sample": self.activation_store.offloaded_bytes / max(1, input_ids.shape[0]),
        }
        if self.config.debug_stats:
            timing["debug"] = debug_stats
        return loss_val, input_ids.numel(), timing

    def _replay_one_layer_backward(
        self,
        layer_idx: int,
        layer_input_base: torch.Tensor,
        state_input_base: torch.Tensor,
        grad_hidden: torch.Tensor,
        grad_v_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._prefetch_layer_async(layer_idx, layer_idx % 2)
        gpu_layer = self._wait_for_layer(layer_idx, layer_idx % 2)
        with torch.cuda.stream(self.compute_stream):
            layer_input = layer_input_base.detach().requires_grad_(True)
            state_input = state_input_base.detach().requires_grad_(True)
            for param in gpu_layer.parameters():
                param.requires_grad_(True)

            layer_output, state_output = gpu_layer(layer_input, state_input)
            grad_targets = (layer_input, state_input, *tuple(gpu_layer.parameters()))
            grads = torch.autograd.grad(
                (layer_output, state_output),
                grad_targets,
                (grad_hidden, grad_v_first),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            next_grad_hidden = grads[0].detach()
            next_grad_v_first = grads[1].detach() if grads[1] is not None else torch.zeros_like(state_input)
            for param, grad in zip(gpu_layer.parameters(), grads[2:], strict=True):
                param.grad = grad
                param.requires_grad_(False)
            self._record_buffer_busy(layer_idx, layer_idx % 2)
        self._copy_layer_grads_to_cpu_async(layer_idx, gpu_layer)
        self._drain_ready_grad_tasks()
        self._drain_grad_tasks(wait_all=False, min_remaining=self.config.num_grad_slabs - 1)
        return next_grad_hidden, next_grad_v_first

    def _clone_module_to_gpu(self, module: nn.Module) -> nn.Module:
        gpu_module = copy.deepcopy(module).to(device=self.device, dtype=self.config.dtype)
        gpu_module.train(module.training)
        return gpu_module

    def _checkpoint_activation(self, hidden: torch.Tensor, has_v_first: bool):
        with torch.cuda.stream(self.activation_stream):
            self.activation_stream.wait_stream(self.compute_stream)
            return self.activation_store.checkpoint(hidden, has_v_first=has_v_first)

    def _save_v_first(self, v_first: torch.Tensor) -> None:
        with torch.cuda.stream(self.activation_stream):
            self.activation_stream.wait_stream(self.compute_stream)
            self.activation_store.save_v_first(v_first)

    def _unpack_activation(self, checkpoint):
        with torch.cuda.stream(self.compute_stream):
            return self.activation_store.unpack(checkpoint, self.device)

    def _refresh_layer_flat(self, layer_idx: int) -> None:
        if not self.layer_flat_dirty[layer_idx]:
            return
        flat = self.layer_pinned_flats[layer_idx]
        offset = 0
        destinations = []
        sources = []
        for param in self.layers[layer_idx].parameters():
            numel = param.numel()
            destinations.append(flat[offset : offset + numel])
            sources.append(param.detach().reshape(-1).to(dtype=self.config.dtype))
            offset += numel
        batched_copy_(destinations, sources)
        self.layer_flat_dirty[layer_idx] = False

    def _layer_template_kind(self, layer_idx: int) -> str:
        return "first" if layer_idx == 0 else "regular"

    def _ensure_layer_template(self, layer_idx: int, buffer_idx: int) -> nn.Module:
        kind = self._layer_template_kind(layer_idx)
        template = self.gpu_layer_templates[buffer_idx].get(kind)
        if template is not None:
            return template

        source_idx = 0 if kind == "first" else 1
        if source_idx >= len(self.layers):
            source_idx = layer_idx
        template = self._clone_module_to_gpu(self.layers[source_idx])
        if kind == "regular":
            template.layer_id = 1
            template.att.layer_id = 1
            template.ffn.layer_id = 1
        self.gpu_layer_templates[buffer_idx][kind] = template
        self.template_layer_idx[buffer_idx][kind] = None
        self.template_busy_events[buffer_idx][kind] = None
        self.template_ready_events[buffer_idx][kind] = torch.cuda.Event(enable_timing=False)
        return template

    def _prefetch_layer_async(self, layer_idx: int, buffer_idx: int) -> None:
        kind = self._layer_template_kind(layer_idx)
        if self.template_layer_idx[buffer_idx].get(kind) == layer_idx and not self.layer_flat_dirty[layer_idx]:
            return
        gpu_layer = self._ensure_layer_template(layer_idx, buffer_idx)
        self._refresh_layer_flat(layer_idx)
        layer_numel = self.layer_numels[layer_idx]
        shapes = self.layer_param_shapes[layer_idx]
        numels = self.layer_param_numels[layer_idx]
        gpu_flat = self.gpu_flat_buffers[buffer_idx][:layer_numel]
        busy_event = self.template_busy_events[buffer_idx].get(kind)
        with torch.cuda.stream(self.weight_stream):
            if busy_event is not None:
                self.weight_stream.wait_event(busy_event)
            gpu_flat.copy_(self.layer_pinned_flats[layer_idx], non_blocking=True)
            offset = 0
            destinations = []
            sources = []
            with torch.no_grad():
                for param, shape, numel in zip(gpu_layer.parameters(), shapes, numels, strict=True):
                    destinations.append(param.data)
                    sources.append(gpu_flat[offset : offset + numel].view(shape))
                    param.grad = None
                    param.requires_grad_(False)
                    offset += numel
                batched_copy_(destinations, sources, non_blocking=True)
            self.template_ready_events[buffer_idx][kind].record(self.weight_stream)
        self.template_layer_idx[buffer_idx][kind] = layer_idx

    def _wait_for_layer(self, layer_idx: int, buffer_idx: int) -> nn.Module:
        kind = self._layer_template_kind(layer_idx)
        if self.template_layer_idx[buffer_idx].get(kind) != layer_idx:
            self._prefetch_layer_async(layer_idx, buffer_idx)
        self.compute_stream.wait_event(self.template_ready_events[buffer_idx][kind])
        return self.gpu_layer_templates[buffer_idx][kind]

    def _record_buffer_busy(self, layer_idx: int, buffer_idx: int) -> None:
        kind = self._layer_template_kind(layer_idx)
        event = torch.cuda.Event(enable_timing=False)
        event.record(self.compute_stream)
        self.template_busy_events[buffer_idx][kind] = event

    def _copy_layer_grads_to_cpu_async(self, layer_idx: int, gpu_layer: nn.Module) -> None:
        self._copy_grads_to_cpu_async(list(gpu_layer.parameters()), list(self.layers[layer_idx].parameters()))

    def _copy_module_grads_to_cpu_async(self, gpu_module: nn.Module, cpu_module: nn.Module) -> None:
        self._copy_grads_to_cpu_async(list(gpu_module.parameters()), list(cpu_module.parameters()))

    def _copy_grads_to_cpu_async(self, gpu_params: list[torch.nn.Parameter], cpu_params: list[torch.nn.Parameter]) -> None:
        entries = [
            (gpu_param, cpu_param)
            for gpu_param, cpu_param in zip(gpu_params, cpu_params, strict=True)
            if gpu_param.grad is not None
        ]
        if not entries:
            return
        self._reap_grad_accum_futures(wait_all=False)
        if len(self._grad_tasks) + len(self._grad_accum_futures) >= self.config.num_grad_slabs:
            self._drain_grad_tasks(wait_all=False, min_remaining=self.config.num_grad_slabs - 1)

        shapes = [cpu_param.shape for _, cpu_param in entries]
        numels = [gpu_param.grad.numel() for gpu_param, _ in entries]
        total_numel = sum(numels)
        slab_idx = self.grad_slab_free.get()
        slab = self.grad_slabs[slab_idx][:total_numel]
        event = self.grad_slab_events[slab_idx]
        source_grads = []
        with torch.cuda.stream(self.grad_stream):
            self.grad_stream.wait_stream(self.compute_stream)
            offset = 0
            destinations = []
            sources = []
            for (gpu_param, _), numel in zip(entries, numels, strict=True):
                grad = gpu_param.grad.detach()
                source_grads.append(grad)
                destinations.append(slab[offset : offset + numel])
                sources.append(grad.reshape(-1))
                gpu_param.grad = None
                offset += numel
            batched_copy_(destinations, sources, non_blocking=True)
            event.record(self.grad_stream)
        self._grad_tasks.append((slab_idx, event, slab, [cpu for _, cpu in entries], shapes, numels, source_grads))

    def _drain_grad_tasks(self, wait_all: bool, min_remaining: int = 0) -> None:
        if wait_all:
            for task in self._grad_tasks:
                self._submit_grad_task(task)
            self._grad_tasks.clear()
            self._reap_grad_accum_futures(wait_all=True)
            return

        self._drain_ready_grad_tasks()
        self._reap_grad_accum_futures(wait_all=False)
        while self._grad_tasks and (wait_all or len(self._grad_tasks) > min_remaining):
            self._submit_grad_task(self._grad_tasks.pop(0))
            self._reap_grad_accum_futures(wait_all=False, min_remaining=min_remaining)

    def _drain_ready_grad_tasks(self) -> None:
        if not self._grad_tasks:
            return
        pending = []
        for task in self._grad_tasks:
            slab_idx, event, slab, cpu_params, shapes, numels, _source_grads = task
            if event.query():
                self._submit_grad_task(task)
            else:
                pending.append(task)
        self._grad_tasks = pending

    def _submit_grad_task(self, task) -> None:
        future = self.grad_accum_executor.submit(self._finish_grad_task, task)
        self._grad_accum_futures.append(future)

    def _finish_grad_task(self, task) -> None:
        slab_idx, event, slab, cpu_params, shapes, numels, _source_grads = task
        event.synchronize()
        self._accumulate_grad_task(slab, cpu_params, numels)
        self.grad_slab_free.put(slab_idx)

    def _reap_grad_accum_futures(self, wait_all: bool, min_remaining: int | None = None) -> None:
        pending = []
        for future in self._grad_accum_futures:
            if wait_all or future.done():
                future.result()
            else:
                pending.append(future)
        self._grad_accum_futures = pending
        if min_remaining is None:
            return
        while not wait_all and len(self._grad_tasks) + len(self._grad_accum_futures) > min_remaining:
            if not self._grad_accum_futures:
                break
            future = self._grad_accum_futures.pop(0)
            future.result()

    def _accumulate_grad_task(self, slab: torch.Tensor, cpu_params: list[torch.nn.Parameter], numels: list[int]) -> None:
        grads = []
        for cpu_param in cpu_params:
            if cpu_param.grad is None:
                cpu_param.grad = torch.zeros_like(cpu_param)
            grads.append(cpu_param.grad)
        accumulate_grad_slab_(grads, slab, numels)
