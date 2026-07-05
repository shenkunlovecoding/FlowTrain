from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load


def accumulate_grad_slab_(
    grads: Sequence[torch.Tensor],
    slab: torch.Tensor,
    numels: Sequence[int],
    *,
    use_cpp: bool = True,
) -> None:
    if not grads:
        return
    if use_cpp:
        extension = _load_extension()
        if extension is not None:
            extension.accumulate_grad_slab(list(grads), slab, [int(numel) for numel in numels])
            return

    offset = 0
    sources = []
    for grad, numel in zip(grads, numels, strict=True):
        sources.append(slab[offset : offset + numel].view(grad.shape).to(dtype=grad.dtype))
        offset += numel
    try:
        torch._foreach_add_(list(grads), sources)
    except (RuntimeError, TypeError, AttributeError):
        for grad, source in zip(grads, sources, strict=True):
            grad.add_(source)


@lru_cache(maxsize=1)
def _load_extension():
    if os.environ.get("FLOWTRAIN_DISABLE_CPU_ACCUM_CPP") == "1":
        return None
    source = Path(__file__).resolve().parent / "csrc" / "cpu_accum.cpp"
    try:
        return load(
            name="flowtrain_cpu_accum",
            sources=[str(source)],
            extra_cflags=["-O3"],
            verbose=os.environ.get("FLOWTRAIN_CPU_ACCUM_VERBOSE") == "1",
        )
    except Exception as exc:  # pragma: no cover - depends on local compiler/toolchain.
        warnings.warn(f"FlowTrain CPU accumulation C++ extension unavailable, using Python fallback: {exc}")
        return None
