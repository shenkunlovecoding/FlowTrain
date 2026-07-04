from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


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
