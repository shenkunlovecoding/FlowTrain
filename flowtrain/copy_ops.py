from __future__ import annotations

from collections.abc import Sequence

import torch


def batched_copy_(
    destinations: Sequence[torch.Tensor],
    sources: Sequence[torch.Tensor],
    *,
    non_blocking: bool = False,
) -> None:
    """Copy tensor lists with foreach when available, falling back to a loop."""

    if len(destinations) != len(sources):
        raise ValueError("destinations and sources must have the same length")
    if not destinations:
        return

    foreach_copy = getattr(torch, "_foreach_copy_", None)
    if foreach_copy is not None:
        try:
            foreach_copy(list(destinations), list(sources), non_blocking=non_blocking)
            return
        except RuntimeError:
            pass
        except TypeError:
            try:
                foreach_copy(list(destinations), list(sources))
                return
            except RuntimeError:
                pass

    for destination, source in zip(destinations, sources, strict=True):
        destination.copy_(source, non_blocking=non_blocking)
