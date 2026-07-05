from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


_INT8_MAX = 127.0


def _quantize_int8_blockwise(
    tensor: torch.Tensor, block_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize a CPU tensor to block-wise int8.

    Returns ``(q, scale, padded_numel)`` where ``q`` is an ``int8`` tensor of
    shape ``[n_blocks, block_size]`` (the input is flattened and zero-padded to
    a multiple of ``block_size``) and ``scale`` is a float32 per-block absmax/127
    tensor of shape ``[n_blocks]``. All-zero blocks use ``scale=1.0`` so the
    stored zeros dequantize back to exactly zero.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).flatten()
    numel = flat.numel()
    padded_numel = numel + (-numel) % block_size
    if padded_numel != numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - numel))
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1)
    scale = absmax / _INT8_MAX
    # Avoid division by zero for all-zero blocks (e.g. fresh state); 1.0 scale
    # leaves the zeros untouched after round-trip.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(blocks / scale.unsqueeze(1)).clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)
    return q, scale.to(torch.float32), padded_numel


def _dequantize_int8_blockwise(
    q: torch.Tensor, scale: torch.Tensor, orig_shape: torch.Size
) -> torch.Tensor:
    """Inverse of :func:`_quantize_int8_blockwise`, returning a float32 CPU tensor."""
    numel = 1
    for dim in orig_shape:
        numel *= dim
    flat = (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).view(-1)[:numel]
    return flat.reshape(orig_shape).contiguous()


class CPUAdamW:
    """AdamW for CPU-resident FlowTrain master parameters.

    Parameters stay on CPU and all state tensors are float32 CPU tensors. The
    update itself is expressed with PyTorch vector ops, so the host backend can
    use its normal SIMD kernels.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = 1.0,
    ):
        self.defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self.max_grad_norm = max_grad_norm
        self.param_groups = self._normalize_param_groups(params)
        self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor | int]] = {}
        self.global_step = 0

    def _normalize_param_groups(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        params = list(params)
        if not params:
            raise ValueError("CPUAdamW got an empty parameter list")
        if isinstance(params[0], dict):
            groups = []
            for raw_group in params:
                group = dict(raw_group)
                group.pop("names", None)
                lr_scale = float(group.pop("my_lr_scale", 1.0))
                group.setdefault("lr", self.defaults["lr"] * lr_scale)
                group.setdefault("betas", self.defaults["betas"])
                group.setdefault("eps", self.defaults["eps"])
                group.setdefault("weight_decay", self.defaults["weight_decay"])
                group["params"] = list(group["params"])
                groups.append(group)
            return groups
        return [
            {
                "params": params,
                **self.defaults,
            }
        ]

    def _state_for(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor | int]:
        state = self.state.get(param)
        if state is None:
            master = param.detach().float().cpu().clone()
            state = {
                "step": 0,
                "master": master,
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
            self.state[param] = state
        return state

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

    def clip_gradients(self) -> float:
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return 0.0
        params_with_grad = [
            param
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not params_with_grad:
            return 0.0
        total_norm = torch.linalg.vector_norm(
            torch.stack([param.grad.detach().float().norm(2) for param in params_with_grad]),
            ord=2,
        )
        clip_coef = self.max_grad_norm / (float(total_norm) + 1e-8)
        if clip_coef < 1.0:
            for param in params_with_grad:
                param.grad.mul_(clip_coef)
        return float(total_norm)

    @torch.no_grad()
    def step(self) -> float:
        self.global_step += 1
        grad_norm = self.clip_gradients()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.device.type != "cpu":
                    raise ValueError("CPUAdamW expects CPU-resident parameters")

                state = self._state_for(param)
                state["step"] = int(state["step"]) + 1
                step = int(state["step"])
                master = state["master"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                assert isinstance(master, torch.Tensor)
                assert isinstance(exp_avg, torch.Tensor)
                assert isinstance(exp_avg_sq, torch.Tensor)

                grad = param.grad.detach().float().cpu()
                if weight_decay != 0:
                    master.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(eps)
                master.addcdiv_(exp_avg, denom, value=-step_size)
                param.data.copy_(master.to(dtype=param.dtype))

        return grad_norm

    def state_dict(self) -> dict[str, Any]:
        param_to_index = {}
        params = []
        for group in self.param_groups:
            for param in group["params"]:
                if param not in param_to_index:
                    param_to_index[param] = len(params)
                    params.append(param)

        state = {}
        for param, value in self.state.items():
            if param in param_to_index:
                state[param_to_index[param]] = value
        groups = []
        for group in self.param_groups:
            saved = {key: value for key, value in group.items() if key != "params"}
            saved["params"] = [param_to_index[param] for param in group["params"]]
            groups.append(saved)
        return {"global_step": self.global_step, "state": state, "param_groups": groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.global_step = int(state_dict["global_step"])
        params = []
        for group in self.param_groups:
            params.extend(group["params"])
        self.state.clear()
        for index, value in state_dict["state"].items():
            self.state[params[int(index)]] = value


class DeepSpeedCPUAdamW:
    """Optional DeepSpeed CPUAdam wrapper with FlowTrain-style grad clipping."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = 1.0,
    ):
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError as exc:
            raise ImportError(
                "optimizer='deepspeed_cpu_adam' requires deepspeed; install it or use optimizer='adamw'"
            ) from exc

        groups = CPUAdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        ).param_groups
        self.max_grad_norm = max_grad_norm
        self.optimizer = DeepSpeedCPUAdam(
            groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.param_groups = self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def clip_gradients(self) -> float:
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return 0.0
        params_with_grad = [
            param
            for group in self.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if not params_with_grad:
            return 0.0
        total_norm = torch.linalg.vector_norm(
            torch.stack([param.grad.detach().float().norm(2) for param in params_with_grad]),
            ord=2,
        )
        clip_coef = self.max_grad_norm / (float(total_norm) + 1e-8)
        if clip_coef < 1.0:
            for param in params_with_grad:
                param.grad.mul_(clip_coef)
        return float(total_norm)

    @torch.no_grad()
    def step(self) -> float:
        grad_norm = self.clip_gradients()
        self.optimizer.step()
        return grad_norm

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)


def _reduced_qr(a: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q


def _shifted_cholesky_qr(a: torch.Tensor, eps: float) -> torch.Tensor:
    gram = a.T @ a
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    shift = float(gram.norm()) * eps
    if not math.isfinite(shift) or shift <= 0:
        return _reduced_qr(a)
    try:
        chol = torch.linalg.cholesky(gram + shift * eye)
        q = torch.linalg.solve_triangular(chol, a.T, upper=False).T
    except RuntimeError:
        return _reduced_qr(a)
    if not bool(torch.isfinite(q).all()):
        return _reduced_qr(a)
    return q


def _col_norm(a: torch.Tensor, eps: float) -> torch.Tensor:
    return a / a.norm(dim=0, keepdim=True).clamp_min(eps)


def _orient_tall(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if matrix.shape[0] >= matrix.shape[1]:
        return matrix, False
    return matrix.T, True


class CPUQRMuon(CPUAdamW):
    """QR Muon for CPU-resident FlowTrain master parameters.

    Groups marked with ``use_muon=True`` use the streaming power-iteration
    Muon update. Other groups fall back to CPU AdamW, which keeps embeddings,
    head, norm, and scalar/vector RWKV parameters on the conservative path.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-18,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = 1.0,
        muon_beta: float = 0.95,
        muon_eps: float = 1e-9,
        muon_double_qr: bool = True,
        muon_update_scale: bool = True,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        self.muon_beta = muon_beta
        self.muon_eps = muon_eps
        self.muon_double_qr = muon_double_qr
        self.muon_update_scale = muon_update_scale
        for group in self.param_groups:
            group.setdefault("use_muon", False)
            group.setdefault("muon_beta", muon_beta)
            group.setdefault("muon_eps", muon_eps)
            group.setdefault("muon_double_qr", muon_double_qr)
            group.setdefault("muon_update_scale", muon_update_scale)

    def _adam_state_for(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor | int]:
        state = self.state.get(param)
        if state is None or "exp_avg_sq" not in state:
            master = param.detach().float().cpu().clone()
            state = {
                "step": 0,
                "master": master,
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
            self.state[param] = state
        return state

    def _muon_state_for(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor | int]:
        if param.ndim != 2:
            raise ValueError("CPUQRMuon use_muon=True groups require 2D parameters")
        state = self.state.get(param)
        short_dim = min(param.shape)
        if state is None or "basis_v" not in state:
            master = param.detach().float().cpu().clone()
            state = {
                "step": 0,
                "master": master,
                "momentum": torch.zeros_like(master),
                "basis_v": torch.eye(short_dim, dtype=torch.float32),
            }
            self.state[param] = state
        return state

    def _step_adamw_param(self, param: torch.nn.Parameter, group: dict[str, Any]) -> None:
        state = self._adam_state_for(param)
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])
        master = state["master"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        assert isinstance(master, torch.Tensor)
        assert isinstance(exp_avg, torch.Tensor)
        assert isinstance(exp_avg_sq, torch.Tensor)

        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        grad = param.grad.detach().float().cpu()

        if weight_decay != 0:
            master.mul_(1.0 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        denom = exp_avg_sq.sqrt().add_(eps)
        master.addcdiv_(exp_avg, denom, value=-step_size)
        param.data.copy_(master.to(dtype=param.dtype))

    def _streaming_muon_update(
        self,
        momentum: torch.Tensor,
        basis_v: torch.Tensor,
        eps: float,
        double_qr: bool,
        update_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        matrix, transposed = _orient_tall(momentum)
        v = basis_v.to(dtype=matrix.dtype, device=matrix.device)
        if double_qr:
            left = _shifted_cholesky_qr(matrix @ v, eps)
            v = _shifted_cholesky_qr(matrix.T @ left, eps)
        else:
            v = _shifted_cholesky_qr(matrix.T @ matrix @ v, eps)
        u = _col_norm(matrix @ v, eps)
        update = u @ v.T
        if update_scale:
            update.mul_(math.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1])))
        if transposed:
            update = update.T
        return update, v

    def _step_muon_param(self, param: torch.nn.Parameter, group: dict[str, Any]) -> None:
        state = self._muon_state_for(param)
        state["step"] = int(state["step"]) + 1
        master = state["master"]
        momentum = state["momentum"]
        basis_v = state["basis_v"]
        assert isinstance(master, torch.Tensor)
        assert isinstance(momentum, torch.Tensor)
        assert isinstance(basis_v, torch.Tensor)

        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta = float(group["muon_beta"])
        eps = float(group["muon_eps"])
        double_qr = bool(group["muon_double_qr"])
        update_scale = bool(group["muon_update_scale"])

        grad = param.grad.detach().float().cpu()
        momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
        update, next_v = self._streaming_muon_update(momentum, basis_v, eps, double_qr, update_scale)

        if weight_decay != 0:
            master.mul_(1.0 - lr * weight_decay)
        master.add_(update, alpha=-lr)
        state["basis_v"] = next_v.detach().cpu()
        param.data.copy_(master.to(dtype=param.dtype))

    @torch.no_grad()
    def step(self) -> float:
        self.global_step += 1
        grad_norm = self.clip_gradients()

        for group in self.param_groups:
            use_muon = bool(group.get("use_muon", False))
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.device.type != "cpu":
                    raise ValueError("CPUQRMuon expects CPU-resident parameters")
                if use_muon:
                    self._step_muon_param(param, group)
                else:
                    self._step_adamw_param(param, group)

        return grad_norm


class CPU8bitAdamW(CPUAdamW):
    """Experimental AdamW with block-wise int8 optimizer state for CPU-master params.

    Keeps the float32 master copy and RWKV-replay integration of :class:`CPUAdamW`,
    but quantizes the first/second moments (``exp_avg`` / ``exp_avg_sq``) to
    block-wise int8 with per-block float32 scales. Optimizer-state footprint drops
    from ~8 bytes/param (float32 m+v) to ~2 bytes/param, freeing host RAM on
    large CPU-resident models. Parameters smaller than ``min_quantized_numel``
    (norms, RWKV time-mix / vector scalars) keep float32 state to avoid quality
    loss on tiny tensors.

    Experimental: int8 ``v`` adds ~1/127 (~0.8%) relative noise to ``sqrt(v)``,
    which dwarfs the tiny RWKV ``eps`` (1e-18); treat the quantization noise as
    extra stabilization and validate loss curves before relying on it.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = 1.0,
        block_size: int = 256,
        min_quantized_numel: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if min_quantized_numel < 0:
            raise ValueError("min_quantized_numel must be >= 0")
        self.block_size = int(block_size)
        self.min_quantized_numel = int(min_quantized_numel)

    def _should_quantize(self, param: torch.nn.Parameter) -> bool:
        return param.numel() >= self.min_quantized_numel

    def _state_for(self, param: torch.nn.Parameter) -> dict[str, torch.Tensor | int]:
        state = self.state.get(param)
        if state is not None:
            return state
        master = param.detach().float().cpu().clone()
        if self._should_quantize(param):
            n_blocks = (param.numel() + self.block_size - 1) // self.block_size
            state = {
                "step": 0,
                "master": master,
                "q_exp_avg": torch.zeros(n_blocks * self.block_size, dtype=torch.int8),
                "exp_avg_scale": torch.zeros(n_blocks, dtype=torch.float32),
                "q_exp_avg_sq": torch.zeros(n_blocks * self.block_size, dtype=torch.int8),
                "exp_avg_sq_scale": torch.zeros(n_blocks, dtype=torch.float32),
            }
        else:
            state = {
                "step": 0,
                "master": master,
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
        self.state[param] = state
        return state

    def _adamw_update(
        self,
        master: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        grad: torch.Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        step: int,
    ) -> None:
        if weight_decay != 0:
            master.mul_(1.0 - lr * weight_decay)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        denom = exp_avg_sq.sqrt().add_(eps)
        master.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_quantized(self, param: torch.nn.Parameter, group: dict[str, Any]) -> None:
        state = self._state_for(param)
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])
        master = state["master"]
        assert isinstance(master, torch.Tensor)
        q_exp_avg = state["q_exp_avg"]
        exp_avg_scale = state["exp_avg_scale"]
        q_exp_avg_sq = state["q_exp_avg_sq"]
        exp_avg_sq_scale = state["exp_avg_sq_scale"]
        assert isinstance(q_exp_avg, torch.Tensor)
        assert isinstance(exp_avg_scale, torch.Tensor)
        assert isinstance(q_exp_avg_sq, torch.Tensor)
        assert isinstance(exp_avg_sq_scale, torch.Tensor)

        grad = param.grad.detach().float().cpu()
        # Dequantize into transient fp32 buffers for the update math, then
        # requantize so steady-state state stays int8.
        exp_avg = _dequantize_int8_blockwise(
            q_exp_avg.view(-1, self.block_size), exp_avg_scale, param.shape
        )
        exp_avg_sq = _dequantize_int8_blockwise(
            q_exp_avg_sq.view(-1, self.block_size), exp_avg_sq_scale, param.shape
        )
        self._adamw_update(
            master,
            exp_avg,
            exp_avg_sq,
            grad,
            float(group["lr"]),
            group["betas"][0],
            group["betas"][1],
            float(group["eps"]),
            float(group["weight_decay"]),
            step,
        )
        state["q_exp_avg"], state["exp_avg_scale"], _ = _quantize_int8_blockwise(exp_avg, self.block_size)
        state["q_exp_avg_sq"], state["exp_avg_sq_scale"], _ = _quantize_int8_blockwise(exp_avg_sq, self.block_size)
        param.data.copy_(master.to(dtype=param.dtype))

    def _step_fp32(self, param: torch.nn.Parameter, group: dict[str, Any]) -> None:
        state = self._state_for(param)
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])
        master = state["master"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        assert isinstance(master, torch.Tensor)
        assert isinstance(exp_avg, torch.Tensor)
        assert isinstance(exp_avg_sq, torch.Tensor)
        grad = param.grad.detach().float().cpu()
        self._adamw_update(
            master,
            exp_avg,
            exp_avg_sq,
            grad,
            float(group["lr"]),
            group["betas"][0],
            group["betas"][1],
            float(group["eps"]),
            float(group["weight_decay"]),
            step,
        )
        param.data.copy_(master.to(dtype=param.dtype))

    @torch.no_grad()
    def step(self) -> float:
        self.global_step += 1
        grad_norm = self.clip_gradients()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.device.type != "cpu":
                    raise ValueError("CPU8bitAdamW expects CPU-resident parameters")
                if self._should_quantize(param):
                    self._step_quantized(param, group)
                else:
                    self._step_fp32(param, group)

        return grad_norm
