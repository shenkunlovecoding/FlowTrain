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

## Backends

- `backend="tilelang"` is the default training path. TimeMix and ChannelMix use
  TileLang GEMM helpers, and recurrence is routed through
  `rwkv7_recurrence_tilelang(r, raw_w, k, v, a, b, head_size, chunk_len)`.
- `backend="torch_ref"` is a correctness/debug path. It is not the default
  training backend.

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

## Measurement

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

## Static Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile $(find flowtrain examples scripts -name '*.py' -print)
```

Run a forbidden-term scan over `flowtrain`, `examples`, `scripts`, `README.md`,
`requirements.txt`, and `setup.py` before publishing.
