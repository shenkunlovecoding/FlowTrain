from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_store import ActivationOffload, ActivationQuant, RWKV7ActivationStore
from .rwkv7 import RWKV7, RWKV7Config

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
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.backend not in ("tilelang", "torch_ref"):
            raise ValueError("backend must be 'tilelang' or 'torch_ref'")
        if self.dtype != torch.bfloat16:
            raise ValueError("FlowTrain v1 supports bf16 training only")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.chunk_len < 1:
            raise ValueError("chunk_len must be >= 1")
        if self.activation_offload not in ("none", "cpu"):
            raise ValueError("activation_offload must be 'none' or 'cpu'")
        if self.activation_quant not in ("none", "int8"):
            raise ValueError("activation_quant must be 'none' or 'int8'")
        if self.activation_quant == "int8" and self.activation_offload != "cpu":
            raise ValueError("activation_quant='int8' requires activation_offload='cpu'")


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
    return RWKV7Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        ctx_len=ctx_len,
        head_size=64,
        dim_ffn=dim_ffn.shape[0],
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
    """RWKV-7-only CPU-master trainer with explicit replay backward."""

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
        if model.config.head_size != 64:
            raise ValueError("FlowTrain v1 supports RWKV-7 head_size=64 only")
        self.model = model.cpu()
        self.model.train()

        self.embedding = self.model.emb
        self.layers = list(self.model.blocks)
        self.norm = self.model.ln_out
        self.head = self.model.head
        self.vocab_size = self.model.config.vocab_size

        self.activation_store = RWKV7ActivationStore(config.activation_offload, config.activation_quant)
        self.gpu_layer_buffers: list[nn.Module | None] = [None, None]
        self.gpu_layer_kinds: list[str | None] = [None, None]
        self.layer_param_shapes = [[p.shape for p in layer.parameters()] for layer in self.layers]
        self.layer_param_numels = [[p.numel() for p in layer.parameters()] for layer in self.layers]
        self.layer_numels = [sum(numels) for numels in self.layer_param_numels]
        self.layer_pinned_flats = [_pin_empty(numel, config.dtype) for numel in self.layer_numels]

    def parameters(self):
        return self.model.parameters()

    def get_parameters(self):
        return list(self.model.parameters())

    def optimizer_groups(self, weight_decay: float = 0.1) -> list[dict]:
        return self.model.optimizer_groups(weight_decay)

    def zero_grad(self) -> None:
        for param in self.model.parameters():
            param.grad = None

    def cleanup(self) -> None:
        self.gpu_layer_buffers = [None, None]
        self.gpu_layer_kinds = [None, None]
        self.activation_store.clear()
        torch.cuda.empty_cache()

    @property
    def activation_bytes_per_sample(self) -> float:
        return self.activation_store.offloaded_bytes

    def forward_and_backward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[float, int, dict[str, float]]:
        if labels is None:
            labels = input_ids
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")
        if input_ids.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")

        self.activation_store.clear()
        self.zero_grad()
        start = time.perf_counter()

        input_ids_gpu = input_ids.to(self.device, non_blocking=True)
        labels_gpu = labels.to(self.device, non_blocking=True)
        emb_gpu = self._clone_module_to_gpu(self.embedding)

        checkpoints = {}
        with torch.no_grad():
            hidden = emb_gpu(input_ids_gpu)
            v_first = torch.empty_like(hidden)
            for layer_idx in range(len(self.layers)):
                if layer_idx % self.config.checkpoint_interval == 0:
                    checkpoints[layer_idx] = self.activation_store.checkpoint(hidden, has_v_first=layer_idx > 0)
                gpu_layer = self._load_layer_to_buffer(layer_idx, layer_idx % 2)
                hidden, v_first = gpu_layer(hidden, v_first)
                if layer_idx == 0:
                    self.activation_store.save_v_first(v_first)
            checkpoints[len(self.layers)] = self.activation_store.checkpoint(hidden, has_v_first=bool(self.layers))

        fwd_end = time.perf_counter()

        final_hidden, final_v_first = self.activation_store.unpack(checkpoints[len(self.layers)], self.device)
        hidden_before_norm = final_hidden.detach().requires_grad_(True)
        norm_gpu = self._clone_module_to_gpu(self.norm)
        head_gpu = self._clone_module_to_gpu(self.head)

        hidden_after_norm = norm_gpu(hidden_before_norm)
        logits = head_gpu(hidden_after_norm[:, :-1, :]).reshape(-1, self.vocab_size).float()
        targets = labels_gpu[:, 1:].reshape(-1)
        valid_tokens = int((targets != -100).sum().item())
        if valid_tokens == 0:
            raise ValueError("labels contain no valid next-token targets")
        loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum") / valid_tokens
        loss_val = float(loss.detach().cpu())
        loss.backward()

        grad_hidden = hidden_before_norm.grad.detach()
        grad_v_first = torch.zeros_like(final_v_first)
        self._copy_module_grads_to_cpu(head_gpu, self.head)
        self._copy_module_grads_to_cpu(norm_gpu, self.norm)

        del logits, hidden_after_norm, hidden_before_norm, final_hidden, final_v_first

        num_blocks = (len(self.layers) + self.config.checkpoint_interval - 1) // self.config.checkpoint_interval
        for block_idx in range(num_blocks - 1, -1, -1):
            block_start = block_idx * self.config.checkpoint_interval
            block_end = min((block_idx + 1) * self.config.checkpoint_interval, len(self.layers))
            checkpoint_hidden, checkpoint_v_first = self.activation_store.unpack(checkpoints[block_start], self.device)

            recompute_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            hidden_recompute = checkpoint_hidden
            v_first_recompute = checkpoint_v_first
            with torch.no_grad():
                for layer_idx in range(block_start, block_end):
                    gpu_layer = self._load_layer_to_buffer(layer_idx, layer_idx % 2)
                    hidden_recompute, v_first_recompute = gpu_layer(hidden_recompute, v_first_recompute)
                    recompute_cache[layer_idx] = (hidden_recompute.detach(), v_first_recompute.detach())

            for layer_idx in range(block_end - 1, block_start - 1, -1):
                if layer_idx == block_start:
                    layer_input = checkpoint_hidden.detach().requires_grad_(True)
                    state_input = checkpoint_v_first.detach().requires_grad_(True)
                else:
                    cached_hidden, cached_state = recompute_cache[layer_idx - 1]
                    layer_input = cached_hidden.detach().requires_grad_(True)
                    state_input = cached_state.detach().requires_grad_(True)

                gpu_layer = self._load_layer_to_buffer(layer_idx, layer_idx % 2)
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
                grad_hidden = grads[0].detach()
                grad_v_first = grads[1].detach() if grads[1] is not None else torch.zeros_like(state_input)
                for param, grad in zip(gpu_layer.parameters(), grads[2:], strict=True):
                    param.grad = grad
                    param.requires_grad_(False)
                self._copy_layer_grads_to_cpu(layer_idx, gpu_layer)

            recompute_cache.clear()

        emb_replay = self._clone_module_to_gpu(self.embedding)
        emb_out = emb_replay(input_ids_gpu)
        emb_out.backward(grad_hidden)
        self._copy_module_grads_to_cpu(emb_replay, self.embedding)

        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        torch.cuda.synchronize(self.device)
        bwd_end = time.perf_counter()
        return loss_val, input_ids.numel(), {
            "forward": fwd_end - start,
            "backward": bwd_end - fwd_end,
            "total": bwd_end - start,
            "activation_bytes_per_sample": self.activation_store.offloaded_bytes / max(1, input_ids.shape[0]),
        }

    def _clone_module_to_gpu(self, module: nn.Module) -> nn.Module:
        gpu_module = copy.deepcopy(module).to(device=self.device, dtype=self.config.dtype)
        gpu_module.train(module.training)
        return gpu_module

    def _refresh_layer_flat(self, layer_idx: int) -> None:
        flat = self.layer_pinned_flats[layer_idx]
        offset = 0
        for param in self.layers[layer_idx].parameters():
            numel = param.numel()
            flat[offset : offset + numel].copy_(param.detach().reshape(-1).to(dtype=self.config.dtype))
            offset += numel

    def _load_layer_to_buffer(self, layer_idx: int, buffer_idx: int) -> nn.Module:
        kind = "first" if layer_idx == 0 else "regular"
        shapes = self.layer_param_shapes[layer_idx]
        gpu_layer = self.gpu_layer_buffers[buffer_idx]
        if gpu_layer is None or self.gpu_layer_kinds[buffer_idx] != kind:
            gpu_layer = self._clone_module_to_gpu(self.layers[layer_idx])
            self.gpu_layer_buffers[buffer_idx] = gpu_layer
            self.gpu_layer_kinds[buffer_idx] = kind
        self._refresh_layer_flat(layer_idx)
        flat = self.layer_pinned_flats[layer_idx].to(self.device, non_blocking=True)
        offset = 0
        with torch.no_grad():
            for param, shape, numel in zip(gpu_layer.parameters(), shapes, self.layer_param_numels[layer_idx], strict=True):
                param.data.copy_(flat[offset : offset + numel].view(shape))
                param.grad = None
                param.requires_grad_(False)
                offset += numel
        return gpu_layer

    def _copy_layer_grads_to_cpu(self, layer_idx: int, gpu_layer: nn.Module) -> None:
        self._copy_grads_to_cpu(gpu_layer.parameters(), self.layers[layer_idx].parameters())

    def _copy_module_grads_to_cpu(self, gpu_module: nn.Module, cpu_module: nn.Module) -> None:
        self._copy_grads_to_cpu(gpu_module.parameters(), cpu_module.parameters())

    def _copy_grads_to_cpu(self, gpu_params, cpu_params) -> None:
        for gpu_param, cpu_param in zip(gpu_params, cpu_params, strict=True):
            if gpu_param.grad is None:
                continue
            grad_cpu = gpu_param.grad.detach().to(device="cpu", dtype=cpu_param.dtype)
            if cpu_param.grad is None:
                cpu_param.grad = torch.zeros_like(cpu_param)
            cpu_param.grad.add_(grad_cpu)
            gpu_param.grad = None
