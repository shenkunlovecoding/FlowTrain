# FlowTrain Agent Guide

## Project Shape

FlowTrain is a narrow RWKV-7 training package, not a generic Transformer or
Hugging Face trainer. Keep the scope tight unless the user explicitly asks to
broaden it:

- Supported model family: RWKV-7 x070.
- Supported trainer mode: single CUDA GPU, bf16, `head_size=64`.
- Core design: CPU-resident master parameters and optimizer state, GPU layer
  streaming, replay backward, activation checkpointing/offload, and TileLang
  kernels where available.
- Public package: `flowtrain`. Older notes may mention `infinity/`; in this
  checkout the active implementation is under `flowtrain/`.

Prefer semantic navigation with `cx` before raw file reads:

```bash
cx overview . --full
cx symbols --kinds
cx definition --name FlowTrainTrainer --from flowtrain/trainer.py
cx references --name rwkv7_recurrence_tilelang --unique
```

Use `rg` for text/file searches when semantic navigation is not enough.

## Key Files

- `flowtrain/rwkv7.py`: RWKV-7 model, config, block structure, and official-style
  parameter grouping/init conventions.
- `flowtrain/trainer.py`: `FlowTrainTrainer`, CPU-master streaming schedule,
  CUDA stream orchestration, replay backward, gradient staging, and checkpoint
  loading/inference.
- `flowtrain/activation_store.py`: activation checkpoint packing, CPU offload,
  int8 activation packing, and async readiness handling.
- `flowtrain/optimizer.py`: CPU AdamW and CPU QR Muon implementations.
- `flowtrain/estimator.py`: RWKV-7 batch-size and memory estimator.
- `flowtrain/tilelang_recurrence.py`: fused RWKV-7 recurrence kernels and
  backward correctness-sensitive code.
- `flowtrain/tilelang_time_mix.py`, `flowtrain/tilelang_gemm.py`,
  `flowtrain/tilelang_block.py`: TileLang acceleration paths and fallbacks.
- `flowtrain/cli/train_rwkv7.py`: `flowtrain-train-rwkv7` smoke-training CLI.
- `flowtrain/cli/estimate_rwkv7_bs.py`: `flowtrain-estimate-rwkv7-bs` CLI.
- `examples/rwkv7/train.py`: source-tree RWKV-7 training smoke entrypoint.
- `scripts/measure_rwkv7_activation.py`: real activation-memory calibration.
- `scripts/benchmark_recurrence.py`: recurrence kernel timing sweep.
- `tests/`: CPU API tests plus CUDA/TileLang recurrence regression tests.

## Install And Commands

Editable install:

```bash
pip install -e .
```

Runtime dependencies are intentionally small and mirrored in `requirements.txt`:

```bash
pip install -r requirements.txt
```

CLI smoke train:

```bash
flowtrain-train-rwkv7 --backend tilelang --n-layer 2 --n-embd 64 --batch-size 4 --seq-len 32
```

Source-tree smoke train:

```bash
python examples/rwkv7/train.py --backend tilelang
```

Batch-size estimate:

```bash
flowtrain-estimate-rwkv7-bs --n-layer 24 --n-embd 2048 --seq-len 1024 --gpu-gb 16
```

Activation calibration:

```bash
python scripts/measure_rwkv7_activation.py --seq-len 256 --batch-sizes 1 2 4
```

Recurrence benchmark:

```bash
python scripts/benchmark_recurrence.py
```

## Testing

Run the CPU-safe API tests first:

```bash
pytest tests/test_flowtrain_api.py
```

Run recurrence tests when touching TileLang recurrence code:

```bash
pytest tests/test_recurrence_backward.py
```

The recurrence tests require CUDA and clear TileLang cache paths in subprocesses
for determinism. If CUDA or TileLang is unavailable, report that explicitly
rather than treating skipped or failed GPU tests as equivalent to a full pass.

Before publishing or handing off a code change, run the static compile check
from the README:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile $(find flowtrain examples scripts -name '*.py' -print)
```

## Implementation Constraints

- Do not assume Transformer-style naming or state flow. RWKV-7 uses
  `emb / blocks / ln_out / head` and carries cross-layer `v_first` state.
- `RWKV7Config` and `FlowTrainConfig` enforce bf16-oriented constraints;
  `head_size=64` is a hard FlowTrain v1 assumption.
- `FlowTrainTrainer` keeps CPU master weights and moves layer templates to GPU
  transiently. Avoid changes that accidentally make full model state persistent
  on GPU.
- Preserve overlap-first scheduling: compute, weight H2D, grad D2H, and
  activation movement use separate CUDA streams. Offload changes should improve
  or preserve overlap, not just move tensors synchronously.
- Activation quantization currently means `activation_quant="int8"` with
  `activation_offload="cpu"`. Keep validation behavior aligned across trainer,
  activation store, estimator, and CLIs.
- Batch-size estimation should stay tied to measured behavior. For RWKV-7
  activation claims, prefer `scripts/measure_rwkv7_activation.py` over theory
  alone.
- TileLang paths must keep explicit capability checks and safe torch-reference
  fallbacks. Guard kernels by dtype, CUDA placement, shape, `head_size`, and
  `chunk_len` divisibility.
- Avoid `.item()` / `.cpu()` in hot training paths unless synchronization is
  intentional and measured.

## Style Notes

- Keep changes small and local to the affected path.
- Use type hints consistently with the existing Python code.
- Prefer clear validation errors for unsupported configs.
- Add or update tests when changing config validation, estimator assumptions,
  activation storage, optimizer behavior, or TileLang kernels.
- Do not rewrite `tilelang-docs/` unless the task is documentation work.
