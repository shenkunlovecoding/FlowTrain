# FlowTrain

FlowTrain is an RWKV-7-specialized training system derived from a fork of
MegaTrain and then thoroughly reworked around the RWKV-7 execution model. It is
not a generic Transformer trainer: the active path is a focused single-GPU
RWKV-7 stack with CPU master weights, GPU layer streaming, replay backward,
activation offload, and TileLang acceleration where available.

The goal is practical long-context RWKV-7 fine-tuning on constrained GPU memory
without making the whole model resident on the device.

## What It Is

| Area | FlowTrain choice |
| --- | --- |
| Model family | RWKV-7 x070 |
| Training target | Single CUDA GPU, bf16 |
| Memory model | CPU master parameters and optimizer state |
| GPU residency | One streamed RWKV-7 block at a time, plus transient buffers |
| Backends | `tilelang` with safe `torch_ref` fallbacks |
| Activations | Checkpointing, CPU offload, optional int8 packing |
| Optimizers | CPU AdamW, DeepSpeed CPUAdam, QR Muon, experimental 8-bit AdamW |

FlowTrain preserves the useful MegaTrain idea of streaming model state through a
small GPU working set, but the implementation has been rebuilt for RWKV-7
instead of carrying a generic model abstraction.

## Why FlowTrain

- RWKV-7-native module structure: `emb`, `blocks`, `ln_out`, `head`, and
  cross-layer `v_first` state are first-class concepts.
- CPU-first optimizer design: master parameters, gradients, and optimizer state
  remain on host memory.
- Overlapped execution: compute, weight H2D prefetch, gradient D2H transfer, and
  activation movement use separate CUDA streams.
- Replay backward: checkpoints keep the GPU footprint small while recomputing
  per-layer activations only when gradients are needed.
- TileLang acceleration: recurrence, time-mix, GEMM, and block-level paths are
  guarded by capability checks and fall back to PyTorch reference code.
- SFT-ready data path: JSONL supervised fine-tuning with prompt-token masking by
  default.

## Core Mechanisms

### 1. CPU-Master Model Residency

FlowTrain keeps the canonical RWKV-7 model on CPU:

```python
# flowtrain/trainer.py
self.model = model.cpu()
```

The GPU sees only the transient working set needed for the current step:
embedding/head/norm clones plus two reusable RWKV-7 layer template buffers. A
large bf16 checkpoint therefore does not need to live on GPU as a full model.

For RWKV-7 blocks, FlowTrain does not construct a fresh GPU module for every
layer. It keeps two GPU templates and overwrites their parameter storage as the
streaming schedule advances:

```python
# flowtrain/trainer.py
self.gpu_layer_templates: list[dict[str, nn.Module]] = [{}, {}]
self.gpu_flat_buffers = [
    torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device),
    torch.empty(self.max_layer_numel, dtype=config.dtype, device=self.device),
]
```

Layer 0 has a dedicated template kind because it owns `ln0` and creates the
first `v_first` state; later layers share a regular-block template kind.

### 2. Pinned Flat Layer Transfers

Each CPU layer is flattened into one contiguous pinned-memory tensor:

```python
# flowtrain/trainer.py
self.layer_pinned_flats = [_pin_empty(numel, config.dtype) for numel in self.layer_numels]
```

Before a layer runs, FlowTrain copies that flat tensor to a GPU flat buffer with
`non_blocking=True`, then scatters views into the template parameters with
`batched_copy_`:

```python
# flowtrain/trainer.py
gpu_flat.copy_(self.layer_pinned_flats[layer_idx], non_blocking=True)
batched_copy_(destinations, sources, non_blocking=True)
```

This turns many small parameter transfers into one large H2D copy per layer,
which is friendlier to PCIe and CUDA launch overhead.

### 3. Four CUDA Streams And Double Buffering

The trainer splits work across four CUDA streams:

| Stream | Responsibility |
| --- | --- |
| `compute_stream` | Forward, loss, replay backward kernels |
| `weight_stream` | Next-layer H2D weight prefetch |
| `grad_stream` | Async gradient D2H copies |
| `activation_stream` | Activation checkpoint offload and packing |

During forward and replay, layer `i + 1` can be prefetched while layer `i` is
computing:

```python
# flowtrain/trainer.py
self._prefetch_layer_async(next_idx, next_idx % 2)
```

The two GPU buffers are used in a ping-pong pattern:

```python
buffer_idx = layer_idx % 2
```

CUDA events track when a template buffer is ready or still busy, so the weight
stream can stay ahead without overwriting parameters that the compute stream is
still using.

### 4. Replay Backward With Activation Checkpoints

FlowTrain does not keep the full autograd tape for all RWKV-7 blocks. Forward
runs under `torch.no_grad()`, stores selected boundary activations, and backward
replays the required layers with gradients enabled.

Default `recompute` mode stores one checkpoint every `checkpoint_interval`
layers plus the final output. Backward walks those blocks in reverse and
reconstructs per-layer inputs before calling local autograd:

```python
# flowtrain/trainer.py
grads = torch.autograd.grad(
    (layer_output, state_output),
    (layer_input, state_input, *tuple(gpu_layer.parameters())),
    (grad_hidden, grad_v_first),
    allow_unused=True,
)
```

`activation_strategy="store_layer_inputs"` stores every layer boundary instead.
That uses more activation memory, but avoids cross-layer no-grad replay and only
recomputes the internals of the current layer during backward.

### 5. Activation CPU Offload And Int8 Packing

Activation checkpoints can stay on GPU, move to CPU, or move to CPU as int8
payloads:

```python
# flowtrain/activation_store.py
scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
quantized = torch.clamp(torch.round(values / scale), -127, 127).to(torch.int8)
```

The int8 path stores:

- int8 activation payload, one byte per hidden element.
- fp16 scale, one value per token.

Unpack copies payload and scale back to the target GPU asynchronously and
reconstructs the activation as `(q * scale).to(dtype)`.

### 6. RWKV-7 `v_first` Deduplication

RWKV-7 layer 0 creates `v_first`; later layers reuse it. FlowTrain stores that
tensor once per batch and lets checkpoints keep only a boolean handle:

```python
# flowtrain/activation_store.py
if self._v_first is None:
    self._v_first = self._pack(v_first)
```

For deep models, this avoids saving the same `v_first`-shaped tensor at every
checkpoint boundary.

### 7. K-Slab Gradient Staging

GPU gradients are not accumulated directly into CPU `param.grad` on the critical
path. FlowTrain copies gradients into a pool of pinned CPU slabs, records CUDA
events, and lets a background CPU worker accumulate ready slabs:

```python
# flowtrain/trainer.py
self.grad_slabs = [_pin_empty(grad_slab_numel, config.dtype) for _ in range(config.num_grad_slabs)]
self._drain_grad_tasks(wait_all=False, min_remaining=self.config.num_grad_slabs - 1)
```

The optional `flowtrain/csrc/cpu_accum.cpp` helper batches CPU-side accumulation
when it can be JIT-built; otherwise the Python foreach fallback keeps the
feature optional.

### 8. Chunked Logit Loss

Full logits can dominate memory at long sequence length and large vocabulary.
FlowTrain slices the time dimension and frees each chunk immediately:

```python
# flowtrain/trainer.py
for start_t in range(0, input_ids.shape[1] - 1, self.config.logit_chunk_size):
    logits = head_gpu(hidden_after_norm[:, start_t:end_t, :]).reshape(-1, self.vocab_size).float()
    total_loss = total_loss + F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    del logits
```

This keeps peak logit memory tied to `logit_chunk_size` rather than the full
sequence length.

### 9. TileLang Fused RWKV Kernels

The `tilelang` backend JIT-compiles CUDA kernels for RWKV-7 hot paths while
retaining PyTorch reference fallbacks. The recurrence forward kernel keeps each
head's `head_size x head_size` state matrix in a fragment:

```python
# flowtrain/tilelang_recurrence.py
state = T.alloc_fragment((head_size, head_size), T.float32)
```

The fused recurrence backward path uses a chunked checkpoint-recompute scheme:
Phase A scans forward and writes state at chunk boundaries; Phase B walks chunks
backward, sub-recomputes segment states, and performs the adjoint scan. Local
matrix workspace uses shared memory:

```python
# flowtrain/tilelang_recurrence.py
g_ws = T.alloc_shared((n, n), T.float32)
```

Kernel entry points are guarded by dtype, CUDA placement, shape, `head_size`,
and chunk divisibility checks. Unsupported cases fall back to the torch
reference implementation instead of silently changing semantics.

### 10. CPU Optimizers

CPU AdamW keeps master weights and optimizer states on host memory. QR Muon adds
an RWKV-aware path for eligible 2D block matrices, using shifted Cholesky QR
with a reduced-QR fallback:

```python
# flowtrain/optimizer.py
gram = a.T @ a
chol = torch.linalg.cholesky(gram + shift * eye)
q = torch.linalg.solve_triangular(chol, a.T, upper=False).T
```

The grouping logic keeps embeddings, output head, normalization parameters,
biases, scalar/vector RWKV parameters, and `att.r_k` on AdamW. Only suitable
`blocks.*` 2D matrices use Muon.

`adamw8bit` is the host-memory compression path for the optimizer itself. It
keeps the fp32 CPU master copy, but stores AdamW first and second moments as
block-wise int8 tensors with fp32 scales:

```python
# flowtrain/optimizer.py
"q_exp_avg": torch.zeros(n_blocks * self.block_size, dtype=torch.int8)
"exp_avg_scale": torch.zeros(n_blocks, dtype=torch.float32)
"q_exp_avg_sq": torch.zeros(n_blocks * self.block_size, dtype=torch.int8)
"exp_avg_sq_scale": torch.zeros(n_blocks, dtype=torch.float32)
```

Large tensors therefore reduce optimizer-state storage from roughly 8
bytes/parameter (`m` + `v` in fp32) to roughly 2 bytes/parameter plus scales.
Small tensors below `min_quantized_numel` stay on ordinary fp32 AdamW state, so
norms and RWKV scalar/vector parameters avoid unnecessary quantization noise.

### 11. Batch-Size Estimator

`flowtrain-estimate-rwkv7-bs` models the current trainer before a run starts. It
accounts for:

- GPU base memory: streamed layer buffers, transfer buffer, embedding clones,
  head, and norm.
- GPU per-sample memory: hidden copies, token IDs, and chunked logits.
- CPU base memory: fp32 master parameters, gradients, optimizer state, and
  pinned transfer buffers.
- CPU per-sample memory: activation offload, including the `int8` packing model.
- Optimizer differences: AdamW, DeepSpeed CPUAdam, QR Muon, and 8-bit AdamW.

## Install

```bash
pip install -e .
```

Runtime dependencies are intentionally small:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -e ".[deepspeed]"  # DeepSpeed CPUAdam
pip install -e ".[sft]"        # tokenizer path for supervised fine-tuning
```

`setup.py` does not build a mandatory CUDA/C++ extension. The optional CPU
gradient accumulation helper is JIT-built on first use when the local toolchain
supports it, with a pure-Python fallback.

## Quick Start

Run a small RWKV-7 training smoke test:

```bash
flowtrain-train-rwkv7 \
  --backend tilelang \
  --n-layer 2 \
  --n-embd 64 \
  --batch-size 4 \
  --seq-len 32
```

Source-tree equivalent:

```bash
python examples/rwkv7/train.py --backend tilelang
```

Use the PyTorch reference backend when validating correctness or running without
a supported TileLang kernel path:

```bash
flowtrain-train-rwkv7 --backend torch_ref
```

## Python API

```python
from flowtrain import (
    RWKV7,
    RWKV7Config,
    FlowTrainConfig,
    FlowTrainTrainer,
    make_optimizer,
)

model = RWKV7(RWKV7Config(
    vocab_size=65536,
    n_layer=2,
    n_embd=64,
    ctx_len=128,
    backend="tilelang",
))

trainer = FlowTrainTrainer(
    model,
    FlowTrainConfig(
        batch_size=4,
        seq_len=128,
        backend="tilelang",
        activation_offload="cpu",
        activation_strategy="store_layer_inputs",
    ),
)

optimizer = make_optimizer(trainer.optimizer_groups(), lr=1e-4)
```

The trainer accepts either an `RWKV7` instance or an RWKV-7 checkpoint path. It
does not discover arbitrary model structures, attention masks, image tensors, or
external model objects.

## Supervised Fine-Tuning

`flowtrain-train-sft` connects the same CPU-master trainer to a JSONL SFT
dataset. Supported record shapes include:

```jsonl
{"messages":[{"role":"user","content":"Say hi"},{"role":"assistant","content":"Hi."}]}
{"prompt":"Question: 2+2\nAnswer:","completion":" 4"}
{"instruction":"Reverse this string","input":"abc","output":"cba"}
{"text":"Plain next-token training text."}
```

Prompt/completion and instruction/output records mask prompt tokens by default,
so loss is computed on the assistant/completion side only. Use
`--full-sequence-loss` to train on every token.

```bash
flowtrain-train-sft \
  --dataset data/sft.jsonl \
  --tokenizer /path/to/tokenizer \
  --checkpoint rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --max-length 1024 \
  --batch-size 1 \
  --backend tilelang \
  --activation-offload cpu \
  --activation-quant int8 \
  --activation-strategy store_layer_inputs
```

Source-tree equivalent:

```bash
python examples/sft/train.py \
  --dataset data/sft.jsonl \
  --tokenizer /path/to/tokenizer
```

## Training Pipeline

At a high level, one training step is:

1. Copy input IDs and labels to GPU.
2. Clone the CPU embedding module to GPU and produce hidden states.
3. Stream RWKV-7 blocks through the two layer templates with async prefetch.
4. Store activation checkpoints according to `activation_strategy`.
5. Clone norm/head to GPU and compute chunked next-token loss.
6. Replay RWKV-7 blocks backward, copying gradients into CPU slabs.
7. Accumulate ready slabs on CPU and run the CPU optimizer step.

## Activation Modes

```bash
python examples/rwkv7/train.py \
  --activation-offload cpu \
  --activation-quant int8 \
  --activation-strategy store_layer_inputs
```

`activation_strategy="store_layer_inputs"` stores every RWKV-7 layer input
boundary instead of only every `checkpoint_interval` layers. That avoids
cross-layer no-grad replay during backward while still replaying each local
layer with gradients.

`activation_quant="int8"` currently requires `activation_offload="cpu"`.

## Optimizers

Default CPU AdamW:

```bash
python examples/rwkv7/train.py --optimizer adamw
```

DeepSpeed CPUAdam when the optional extra is installed:

```bash
python examples/rwkv7/train.py --optimizer deepspeed_cpu_adam
```

RWKV-specialized QR Muon:

```bash
python examples/rwkv7/train.py \
  --optimizer qr_muon \
  --muon-beta 0.95
```

Experimental block-wise 8-bit AdamW:

```bash
python examples/rwkv7/train.py --optimizer adamw8bit
```

QR Muon applies Muon updates to eligible `blocks.*` 2D matrices while
embeddings, output head, normalization weights, and vector/scalar parameters
stay on AdamW.

## Measurement

Estimate batch size from a checkpoint:

```bash
python scripts/estimate_rwkv7_batch_size.py \
  --checkpoint rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --seq-len 8192 \
  --optimizer qr_muon \
  --activation-offload cpu \
  --activation-quant int8 \
  --activation-strategy store_layer_inputs
```

Estimate from dimensions:

```bash
flowtrain-estimate-rwkv7-bs \
  --n-layer 24 \
  --n-embd 2048 \
  --vocab-size 65536 \
  --seq-len 4096 \
  --gpu-gb 16
```

Measure activation behavior on real RWKV-7 shapes:

```bash
python scripts/measure_rwkv7_activation.py \
  --checkpoint rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --seq-len 256 \
  --batch-sizes 1 2 4
```

Benchmark recurrence kernels:

```bash
python scripts/benchmark_recurrence.py
```

## Tests

CPU-safe API tests:

```bash
pytest tests/test_flowtrain_api.py
```

CUDA/TileLang recurrence regression tests:

```bash
pytest tests/test_recurrence_backward.py
```

Static compile check:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile $(find flowtrain examples scripts -name '*.py' -print)
```

The recurrence tests require CUDA and TileLang. If either is unavailable,
skips or failures should be reported separately from a full pass.

## Current Scope

FlowTrain is deliberately narrow:

- RWKV-7 x070 only.
- Single CUDA GPU only.
- bf16 training assumptions.
- `head_size=64`.
- Backends are `tilelang` and `torch_ref`.
- Generic Hugging Face model discovery is out of scope.

This focus is what lets FlowTrain keep the memory schedule explicit and
RWKV-aware instead of hiding it behind a broad trainer interface.
