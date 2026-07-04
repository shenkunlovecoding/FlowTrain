# FlowTrain

FlowTrain is a hard-cut RWKV-7 training package. It keeps CPU-resident master
parameters, GPU layer streaming, explicit replay backward, and activation
offload, but removes the generic model stack. The only supported model family is
RWKV-7 x070 on a single bf16 CUDA GPU with `head_size=64`.

## Public API

```python
from flowtrain import (
    RWKV7Config,
    RWKV7,
    FlowTrainConfig,
    FlowTrainTrainer,
    make_optimizer,
)
```

The trainer accepts either an `RWKV7` instance or an RWKV-7 checkpoint path. It
does not discover arbitrary model structures and does not accept generic
attention masks, image tensors, or external model objects.

## Install

```bash
pip install -e .
```

Runtime dependencies are intentionally small:

- `torch`
- `tilelang`
- `numpy`
- `pyyaml`

No CUDA/C++ extension is built by `setup.py`.

## Train

```bash
flowtrain-train-rwkv7 \
  --backend tilelang \
  --n-layer 2 \
  --n-embd 64 \
  --batch-size 4 \
  --seq-len 32
```

The source-tree equivalent is:

```bash
python examples/rwkv7/train.py --backend tilelang
```

## Optimization Model

FlowTrain uses the original three-stage streaming schedule adapted to RWKV-7:

- Streaming forward loads one RWKV-7 block at a time from CPU master weights and
  keeps only checkpointed hidden states.
- Loss is computed with chunked vocabulary projection, then head and norm
  gradients are scheduled back to CPU.
- Replay backward reloads each checkpoint block, recomputes its activations, and
  schedules block gradients back to CPU as soon as they are produced.

The trainer maintains separate CUDA streams for compute, weight H2D prefetch,
and gradient D2H transfer. Two GPU layer buffers are alternated so the weight
stream can prefetch the next block while the compute stream runs the current
block. CPU gradients are staged through pinned slabs capped by
`num_grad_slabs`.

RWKV-7 layer execution uses stateless GPU templates specialized by block kind:
one template for layer 0, which owns `ln0` and creates `v_first`, and reusable
regular-block templates for all later layers. Loading a layer overwrites only
the template parameter slots, avoiding per-layer module construction during the
streaming schedule.

`make_optimizer` returns FlowTrain's CPU AdamW implementation by default.
Parameters and optimizer state stay on CPU; updates use PyTorch CPU vector ops.
Pass `optimizer="qr_muon"` to enable the RWKV-specialized QR Muon adapter:
`blocks.*` 2D matrices use streaming power-iteration Muon with shifted
Cholesky QR, while embeddings, the output head, normalization weights, and
vector/scalar parameters stay on AdamW.

```bash
python examples/rwkv7/train.py \
  --backend torch_ref \
  --optimizer qr_muon \
  --muon-beta 0.95
```

## Backends

- `backend="tilelang"` is the default training path. TimeMix and ChannelMix use
  TileLang GEMM helpers, and recurrence is routed through
  `rwkv7_recurrence_tilelang(r, raw_w, k, v, a, b, head_size, chunk_len)`.
- `backend="torch_ref"` is a correctness/debug path. It is not the default
  training backend.

### Recurrence backward

The recurrence forward has a custom TileLang kernel; its backward is now a
fused TileLang kernel too, instead of recomputing through the pure-PyTorch
reference. Given upstream grad `gout`, the kernel runs a checkpointed reverse
adjoint scan:

- Phase A re-runs the forward scan and dumps the pre-update state at every
  `chunk` boundary (default chunk 64, chosen to divide the sequence length)
  into a small workspace.
- Phase B walks chunks in reverse: per chunk it sub-recomputes the segment
  states from the boundary, then runs the adjoint step producing gradients for
  `r, decay, k, v, a, b` while carrying the adjoint state `H` across chunks.

Transient workspace is `O((T/chunk + chunk) * B * n_head * head_size^2 * 4)`
bytes (fp32), freed after the step — checkpoint-recompute rather than storing
all `T` forward states. `grad(decay)` is chained to `grad(raw_w)` in Python on
the host. When the TileLang capability gate is not met (non-bf16, non-CUDA,
`head_size != 64`, or `timesteps % chunk_len != 0`), the backward falls back to
the PyTorch reference recompute path.

Correctness is checked against finite differences in
`tests/test_recurrence_backward.py`. Each case runs in a fresh subprocess
because TileLang interns compiled kernels within a process, so compiling the
recurrence for more than one shape in a single process returns the wrong
binary; one shape per process (the normal training pattern) is unaffected.

## Activation Storage

`activation_offload="cpu"` enables RWKV-7 checkpoint storage that saves
`hidden_ref + v_first_handle`. `v_first` is generated once by layer 0 and stored
as a batch singleton, so later checkpoints do not repeatedly move the same
tensor between GPU and CPU.

Optional int8 activation compression is explicit:

```bash
python examples/rwkv7/train.py \
  --activation-offload cpu \
  --activation-quant int8
```

The int8 path uses per-token symmetric quantization for checkpointed hidden
states and the singleton `v_first`; tensors are dequantized back to bf16 during
replay.

`activation_strategy="store_layer_inputs"` stores every RWKV-7 layer input
boundary instead of only every `checkpoint_interval` layers:

```bash
python examples/rwkv7/train.py \
  --activation-offload cpu \
  --activation-quant int8 \
  --activation-strategy store_layer_inputs
```

This avoids cross-layer no-grad replay during backward. Each layer still runs
one local replay with gradients so TimeMix/ChannelMix internals do not need to
be stored. It is the middle ground between aggressive recompute and storing all
block-internal activations.

## Measurement

Estimate the largest batch size for the current single-GPU trainer:

```bash
python scripts/estimate_rwkv7_batch_size.py \
  --checkpoint rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --seq-len 8192 \
  --optimizer qr_muon \
  --activation-offload cpu \
  --activation-quant int8 \
  --activation-strategy store_layer_inputs
```

Without a checkpoint, pass the model dimensions directly:

```bash
flowtrain-estimate-rwkv7-bs \
  --n-layer 24 \
  --n-embd 2048 \
  --vocab-size 65536 \
  --seq-len 4096 \
  --gpu-gb 16
```

The estimator models the current trainer, including chunked logits by default.
Pass `--logit-chunk-size 0` only when comparing against full-sequence logits.

```bash
python scripts/measure_rwkv7_activation.py \
  --checkpoint rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --seq-len 256 \
  --batch-sizes 1 2 4
```

The script reports:

- `torch_ref`
- `tilelang`
- `tilelang+singleton`
- `tilelang+singleton+int8`
- `tilelang+store_layer_inputs+int8`

## Static Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile $(find flowtrain examples scripts -name '*.py' -print)
```

Run a forbidden-term scan over `flowtrain`, `examples`, `scripts`, `README.md`,
`requirements.txt`, and `setup.py` before publishing.
