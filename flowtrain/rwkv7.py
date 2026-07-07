########################################################################################################
# Minimal RWKV-7 x070 core extracted from BlinkDL/RWKV-LM RWKV-v7/train_temp.
#
# This file intentionally keeps the official train_temp math, parameter names, and initialization style,
# while replacing fused CUDA kernels / Lightning / DeepSpeed with a FlowTrain
# TileLang backend plus a pure PyTorch reference path.
########################################################################################################

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .tilelang_block import rwkv7_block_tilelang
except ImportError:  # Allows running this file directly from flowtrain/.
    from tilelang_block import rwkv7_block_tilelang


@dataclass
class RWKV7Config:
    vocab_size: int
    n_layer: int = 6
    n_embd: int = 512
    ctx_len: int = 1024
    head_size: int = 64
    dim_att: int | None = None
    dim_ffn: int | None = None
    lora_rank_style: str = "official"
    lora_rank_overrides: dict[int, tuple[int, int, int, int]] | None = None
    backend: Literal["tilelang", "torch_ref"] = "tilelang"
    chunk_len: int = 16

    def __post_init__(self) -> None:
        if self.backend not in {"tilelang", "torch_ref"}:
            raise ValueError("backend must be 'tilelang' or 'torch_ref'")
        if self.dim_att is None:
            self.dim_att = self.n_embd
        if self.dim_ffn is None:
            self.dim_ffn = int((self.n_embd * 3.5) // 32 * 32)
        assert self.dim_att is not None
        assert self.dim_ffn is not None
        assert self.n_embd % 32 == 0
        assert self.head_size == 64, "FlowTrain v1 supports RWKV-7 head_size=64 only"
        assert self.chunk_len > 0
        assert self.dim_att % self.head_size == 0
        assert self.dim_att == self.n_embd, "train_temp x070 assumes dim_att == n_embd"


def _ortho_init(x: torch.Tensor, scale: float) -> torch.Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(f"unsupported shape for orthogonal init: {shape}")
    return x


def official_rank(c: int, multiplier: float) -> int:
    return max(32, int(round((multiplier * (c**0.5)) / 32) * 32))


def lora_ranks(c: int, style: str) -> tuple[int, int, int, int]:
    if style == "official":
        return official_rank(c, 2.5), official_rank(c, 2.5), official_rank(c, 1.7), official_rank(c, 5.0)
    if style == "simplified":
        return 8, 8, 8, 8
    raise ValueError(f"unknown lora_rank_style: {style}")


def _layer_lora_ranks(config: RWKV7Config, layer_id: int) -> tuple[int, int, int, int]:
    """Resolve lora ranks for a specific layer, checking per-layer overrides first."""
    if config.lora_rank_overrides is not None and layer_id in config.lora_rank_overrides:
        return config.lora_rank_overrides[layer_id]
    return lora_ranks(config.n_embd, config.lora_rank_style)


def rwkv7_recurrence(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, head_size: int) -> torch.Tensor:
    """Pure PyTorch reference for the RWKV-7 recurrent attention kernel.

    Inputs are [B, T, C]. This follows the official demo slow path:
    state = state * exp(-exp(w)) + state @ a @ b + v @ k
    out = state @ r
    """

    orig_dtype = r.dtype
    bsz, timesteps, channels = r.shape
    n_head = channels // head_size
    r = r.view(bsz, timesteps, n_head, head_size).float()
    w = torch.exp(-torch.exp(w.view(bsz, timesteps, n_head, head_size).float()))
    k = k.view(bsz, timesteps, n_head, head_size).float()
    v = v.view(bsz, timesteps, n_head, head_size).float()
    a = a.view(bsz, timesteps, n_head, head_size).float()
    b = b.view(bsz, timesteps, n_head, head_size).float()

    out = torch.empty((bsz, timesteps, n_head, head_size), device=r.device, dtype=torch.float32)
    state = torch.zeros((bsz, n_head, head_size, head_size), device=r.device, dtype=torch.float32)

    for t in range(timesteps):
        kk = k[:, t].view(bsz, n_head, 1, head_size)
        rr = r[:, t].view(bsz, n_head, head_size, 1)
        vv = v[:, t].view(bsz, n_head, head_size, 1)
        aa = a[:, t].view(bsz, n_head, head_size, 1)
        bb = b[:, t].view(bsz, n_head, 1, head_size)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t] = (state @ rr).view(bsz, n_head, head_size)

    return out.view(bsz, timesteps, channels).to(dtype=orig_dtype)


class RWKV7TimeMix(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.n_head = config.dim_att // self.head_size
        h = self.n_head
        n = self.head_size
        c = config.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1) if config.n_layer > 1 else 0.0
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.arange(c, dtype=torch.float32).view(1, 1, c) / c

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            linear = torch.arange(c, dtype=torch.float32) / (c - 1) - 0.5
            zigzag = ((torch.arange(c, dtype=torch.float32) % n) - ((n - 1) / 2)) / ((n - 1) / 2)
            zigzag = zigzag * torch.abs(zigzag)
            www = -6 + 6 * (torch.arange(c, dtype=torch.float32) / (c - 1)) ** (1 + ratio_0_to_1**0.3)

            d_decay, d_aaa, d_mv, d_gate = _layer_lora_ranks(config, layer_id)
            self.w1 = nn.Parameter(torch.zeros(c, d_decay))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(d_decay, c), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, c) + 0.5 + zigzag * 2.5)

            self.a1 = nn.Parameter(torch.zeros(c, d_aaa))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(d_aaa, c), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, c) - 0.19 + zigzag * 0.3 + linear * 0.4)

            self.v1 = nn.Parameter(torch.zeros(c, d_mv))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(d_mv, c), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, c) + 0.73 - linear * 0.4)

            self.g1 = nn.Parameter(torch.zeros(c, d_gate))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(d_gate, c), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, c) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, c) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(h, n) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(c, c, bias=False)
            self.key = nn.Linear(c, c, bias=False)
            self.value = nn.Linear(c, c, bias=False)
            self.output = nn.Linear(c, c, bias=False)
            self.ln_x = nn.GroupNorm(h, c, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
            self.key.weight.data.uniform_(-0.05 / (c**0.5), 0.05 / (c**0.5))
            self.value.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
            self.output.weight.data.zero_()

    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, channels = x.size()
        h = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        raw_w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        w = -F.softplus(-raw_w) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(bsz, timesteps, h, -1), dim=-1, p=2.0).view(bsz, timesteps, channels)
        k = k * (1 + (a - 1) * self.k_a)

        x = rwkv7_recurrence(r, w, k, v, -kk, kk * a, self.head_size)
        x = self.ln_x(x.view(bsz * timesteps, channels)).view(bsz, timesteps, channels)
        x = x + ((r.view(bsz, timesteps, h, -1).float() * k.view(bsz, timesteps, h, -1).float() * self.r_k).sum(dim=-1, keepdim=True).to(dtype=x.dtype) * v.view(bsz, timesteps, h, -1)).view(bsz, timesteps, channels)
        return self.output(x * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.arange(config.n_embd, dtype=torch.float32).view(1, 1, config.n_embd) / config.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False)
        self.key.weight.data.uniform_(-0.5 / (config.n_embd**0.5), 0.5 / (config.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        return self.value(torch.relu(self.key(k)) ** 2)


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        self.att = RWKV7TimeMix(config, layer_id)
        self.ffn = RWKV7ChannelMix(config, layer_id)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.backend == "tilelang":
            return rwkv7_block_tilelang(self, x, v_first)
        return self._forward_torch_ref(x, v_first)

    def _forward_torch_ref(self, x: torch.Tensor, v_first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_id == 0:
            x = self.ln0(x)
        x_att, v_first = self.att(self.ln1(x), v_first)
        x = x + x_att
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKV7Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward_features(self, idx: torch.Tensor) -> torch.Tensor:
        _, timesteps = idx.size()
        assert timesteps <= self.config.ctx_len, "Cannot forward, model ctx_len is exhausted."
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.ln_out(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(idx))

    def optimizer_groups(self, weight_decay: float = 0.1) -> list[dict]:
        """Official train_temp grouping: att.w0 uses 2x lr; only large .weight tensors decay."""

        lr_decay: set[str] = set()
        lr_1x: set[str] = set()
        lr_2x: set[str] = set()
        for name, param in self.named_parameters():
            if "att.w0" in name:
                lr_2x.add(name)
            elif len(param.squeeze().shape) >= 2 and weight_decay > 0 and ".weight" in name:
                lr_decay.add(name)
            else:
                lr_1x.add(name)

        param_dict = dict(self.named_parameters())
        groups = [
            {"params": [param_dict[n] for n in sorted(lr_1x)], "weight_decay": 0.0, "my_lr_scale": 1.0, "names": sorted(lr_1x)},
            {"params": [param_dict[n] for n in sorted(lr_2x)], "weight_decay": 0.0, "my_lr_scale": 2.0, "names": sorted(lr_2x)},
        ]
        if weight_decay > 0:
            groups.append({"params": [param_dict[n] for n in sorted(lr_decay)], "weight_decay": weight_decay, "my_lr_scale": 1.0, "names": sorted(lr_decay)})
        return groups

    def generate_init_weight(self, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        """Return official-style initial weights compatible with train_temp naming."""

        state = self.state_dict()
        out: dict[str, torch.Tensor] = {}
        for name, param in state.items():
            shape = param.shape
            if (
                "ln_" in name
                or ".ln" in name
                or "time_" in name
                or "_mask" in name
                or "pos_emb" in name
                or ".mask." in name
                or name.endswith("_w")
                or name.endswith("_w1")
                or name.endswith("_w2")
                or name.endswith("_bias")
                or ".weight" not in name
            ):
                if "ln_x.weight" in name:
                    layer_scale = (1 + int(name.split(".")[1])) / self.config.n_layer
                    tensor = (param * 0.0) + (layer_scale**0.7)
                else:
                    tensor = param.clone()
            elif name == "emb.weight":
                tensor = param.clone()
                scale = -1e-4
                nn.init.uniform_(tensor, a=scale, b=-scale)
            elif name == "head.weight":
                tensor = param.clone()
                scale = 0.5 * math.sqrt(self.config.vocab_size / self.config.n_embd) if self.config.vocab_size > self.config.n_embd else 0.5
                nn.init.orthogonal_(tensor, gain=scale)
            else:
                assert name.endswith(".weight")
                scale = 1.0
                for key in [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", ".oo.", ".rr."]:
                    if key in name:
                        scale = 0.0
                for key in [".att.key.", ".att.gate."]:
                    if key in name:
                        scale = 0.1
                tensor = torch.empty(shape)
                if scale == 0:
                    nn.init.zeros_(tensor)
                elif scale < 0:
                    nn.init.uniform_(tensor, a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(tensor, gain=scale)

            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            out[name] = tensor.cpu()
        return out


def _use_qr_muon_parameter(name: str, param: torch.nn.Parameter) -> bool:
    return (
        name.startswith("blocks.")
        and param.ndim == 2
        and min(param.shape) >= 2
        and not name.endswith("att.r_k")
    )


def _split_qr_muon_groups(groups: list[dict]) -> list[dict]:
    split_groups: list[dict] = []
    for group in groups:
        names = group.get("names")
        if names is None:
            base = dict(group)
            base.pop("names", None)
            base["use_muon"] = False
            split_groups.append(base)
            continue

        muon_params = []
        adamw_params = []
        for name, param in zip(names, group["params"], strict=True):
            if _use_qr_muon_parameter(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        base = {key: value for key, value in group.items() if key not in {"params", "names"}}
        if adamw_params:
            split_groups.append({**base, "params": adamw_params, "use_muon": False})
        if muon_params:
            split_groups.append({**base, "params": muon_params, "use_muon": True})
    return split_groups


def make_optimizer(
    model: RWKV7,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.99),
    eps: float = 1e-18,
    optimizer: Literal["adamw", "deepspeed_cpu_adam", "qr_muon", "adamw8bit"] = "adamw",
    muon_beta: float = 0.95,
    muon_eps: float = 1e-9,
    muon_double_qr: bool = True,
    block_size: int = 256,
    min_quantized_numel: int = 4096,
    debug_finite_checks: bool = False,
):
    from .optimizer import CPU8bitAdamW, CPUAdamW, CPUQRMuon, DeepSpeedCPUAdamW

    groups = model.optimizer_groups(weight_decay=weight_decay)
    if optimizer == "qr_muon":
        return CPUQRMuon(
            _split_qr_muon_groups(groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            muon_beta=muon_beta,
            muon_eps=muon_eps,
            muon_double_qr=muon_double_qr,
        )
    if optimizer not in ("adamw", "deepspeed_cpu_adam", "adamw8bit"):
        raise ValueError("optimizer must be 'adamw', 'deepspeed_cpu_adam', 'qr_muon', or 'adamw8bit'")

    param_names: dict[torch.nn.Parameter, str] = {}
    for group in groups:
        names = group.get("names")
        if names is not None:
            for name, param in zip(names, group["params"], strict=True):
                param_names[param] = name
        group.pop("names", None)
    if optimizer == "deepspeed_cpu_adam":
        return DeepSpeedCPUAdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if optimizer == "adamw8bit":
        return CPU8bitAdamW(
            groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
            min_quantized_numel=min_quantized_numel,
            debug_finite_checks=debug_finite_checks,
            param_names=param_names,
        )
    return CPUAdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters)
