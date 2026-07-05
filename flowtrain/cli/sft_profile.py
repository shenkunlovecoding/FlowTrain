"""Lightweight per-step profiler for the SFT train loop.

Wrap :func:`flowtrain.trainer.FlowTrainTrainer.forward_and_backward` callers
from the CLI; do **not** touch the trainer hot path. Only
:func:`time.perf_counter` around loop phases and one
``torch.cuda.memory_allocated()`` call per step are added, so the profiler is
safe to leave on and introduces no extra ``.cpu()``/``.item()`` sync into the
critical path (per the AGENTS.md hot-path rule).

Produces two artifacts for infra planning:

- ``sft_profile_steps.jsonl`` — one line per step (loss, tokens, per-phase
  timings, GPU alloc/reserved, RSS), handy for plotting.
- ``sft_profile_summary.json`` — aggregates: throughput, step-time p50/p95,
  estimated SFT TFLOPS, per-phase time fractions (where is the time going?),
  peak GPU alloc GB, peak RSS GB, mean GPU utilization %.
"""

from __future__ import annotations

import json
import resource
import statistics
import threading
import time
from pathlib import Path
from typing import Any


def peak_rss_gb() -> float:
    """Peak resident set size of this process in GiB. ``ru_maxrss`` is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[index]


class StepProfiler:
    """Collect per-step timing/memory records and emit JSON artifacts."""

    def __init__(
        self,
        out_dir: str | Path,
        *,
        total_params: float | None = None,
        gpu_samples: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.out_dir / "sft_profile_steps.jsonl"
        self.summary_path = self.out_dir / "sft_profile_summary.json"
        self.gpu_samples_path = self.out_dir / "sft_profile_gpu_samples.jsonl"
        self.total_params = total_params
        self.collect_gpu_samples = gpu_samples

        self._records: list[dict[str, Any]] = []
        self._gpu_samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._sampler_thread: threading.Thread | None = None

        try:
            import torch

            self._torch = torch
        except Exception:  # pragma: no cover - torch is a hard FlowTrain dep.
            self._torch = None

        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        # Truncate any prior run's step log.
        self.steps_path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ GPU sampler
    def start_gpu_sampler(self) -> None:
        torch = self._torch
        if not self.collect_gpu_samples or torch is None or not torch.cuda.is_available():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread = thread
        thread.start()

    def _sample_loop(self) -> None:
        torch = self._torch
        assert torch is not None
        while not self._stop.is_set():
            try:
                self._gpu_samples.append(
                    {
                        "t": time.perf_counter(),
                        "util_pct": int(torch.cuda.utilization()),
                        "mem_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
                    }
                )
            except Exception:  # pragma: no cover - sampler must never break the run.
                pass
            self._stop.wait(0.2)

    def stop_gpu_sampler(self) -> None:
        self._stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=1.0)
            self._sampler_thread = None

    # ------------------------------------------------------------------ per-step
    def record(self, **fields: Any) -> None:
        record = dict(fields)
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            record["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1024**3
            record["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        record["rss_gb"] = peak_rss_gb()
        self._records.append(record)
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ summary
    def summarize_and_write(
        self,
        *,
        batch_size: int,
        seq_len: int,
        optimizer_name: str = "",
        model_label: str = "",
    ) -> dict[str, Any]:
        records = self._records
        if not records:
            return {"num_steps": 0}

        step_times = [float(r["t_step"]) for r in records]
        tokens = [int(r.get("total_tokens", 0)) for r in records]
        sum_step = sum(step_times)
        sum_tokens = sum(tokens)
        mean_step = sum_step / len(step_times) if step_times else 0.0
        mean_tokens_per_step = sum_tokens / len(records) if records else 0.0

        def fraction(key: str) -> float:
            total = sum(float(r.get(key, 0.0)) for r in records)
            return total / sum_step if sum_step > 0 else 0.0

        tflops = None
        if self.total_params and mean_step > 0:
            # Rough SFT fwd+bwd matmul FLOPs: ~6 * params * tokens (lower bound).
            tflops = 6.0 * float(self.total_params) * mean_tokens_per_step / mean_step / 1e12

        gpu_util_mean = None
        if self._gpu_samples:
            gpu_util_mean = statistics.mean(s["util_pct"] for s in self._gpu_samples)

        peak_gpu_alloc = None
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            peak_gpu_alloc = torch.cuda.max_memory_allocated() / 1024**3

        summary: dict[str, Any] = {
            "model": model_label,
            "optimizer": optimizer_name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_steps": len(records),
            "total_params": self.total_params,
            "step_time_mean_s": mean_step,
            "step_time_p50_s": _percentile(sorted(step_times), 50),
            "step_time_p95_s": _percentile(sorted(step_times), 95),
            "tokens_per_sec": (sum_tokens / sum_step) if sum_step > 0 else 0.0,
            "samples_per_sec": (len(records) * batch_size / sum_step) if sum_step > 0 else 0.0,
            "est_sft_tflops": tflops,
            "time_fraction": {
                "data": fraction("t_data"),
                "forward": fraction("t_fwd"),
                "backward": fraction("t_bwd"),
                "optimizer": fraction("t_opt"),
                "zero_grad": fraction("t_zero"),
            },
            "peak_gpu_alloc_gb": peak_gpu_alloc,
            "peak_rss_gb": peak_rss_gb(),
            "gpu_util_mean_pct": gpu_util_mean,
            "gpu_sample_count": len(self._gpu_samples),
        }

        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if self._gpu_samples:
            with self.gpu_samples_path.open("w", encoding="utf-8") as handle:
                for sample in self._gpu_samples:
                    handle.write(json.dumps(sample) + "\n")
        return summary
