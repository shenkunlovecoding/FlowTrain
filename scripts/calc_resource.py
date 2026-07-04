#!/usr/bin/env python3
"""
MegaTrain Resource Calculator

Estimates:
  1. Which models can be trained given your CPU memory
  2. Optimal batch size given GPU memory, model size, and context length

Usage:
  python scripts/calc_resource.py
"""

import math

# ============================================================
# Known model specifications
# ============================================================
KNOWN_MODELS = {
    # Qwen 2.5
    "Qwen2.5-0.5B":  {"params": 0.5,  "hidden": 896,  "layers": 24},
    "Qwen2.5-1.5B":  {"params": 1.5,  "hidden": 1536, "layers": 28},
    "Qwen2.5-3B":    {"params": 3,    "hidden": 2048, "layers": 36},
    "Qwen2.5-7B":    {"params": 7,    "hidden": 3584, "layers": 28},
    "Qwen2.5-14B":   {"params": 14,   "hidden": 5120, "layers": 48},
    "Qwen2.5-32B":   {"params": 32,   "hidden": 5120, "layers": 64},
    "Qwen2.5-72B":   {"params": 72,   "hidden": 8192, "layers": 80},
    # Qwen 3
    "Qwen3-0.6B":    {"params": 0.6,  "hidden": 1024, "layers": 28},
    "Qwen3-1.7B":    {"params": 1.7,  "hidden": 1536, "layers": 28},
    "Qwen3-4B":      {"params": 4,    "hidden": 2560, "layers": 36},
    "Qwen3-8B":      {"params": 8,    "hidden": 4096, "layers": 36},
    "Qwen3-14B":     {"params": 14,   "hidden": 5120, "layers": 48},
    "Qwen3-32B":     {"params": 32,   "hidden": 5120, "layers": 64},
    # Qwen 3.5 (hybrid linear + full attention)
    "Qwen3.5-0.8B":  {"params": 0.8,  "hidden": 1024, "layers": 28, "hybrid": True},
    "Qwen3.5-2B":    {"params": 2,    "hidden": 1536, "layers": 28, "hybrid": True},
    "Qwen3.5-4B":    {"params": 4,    "hidden": 2560, "layers": 36, "hybrid": True},
    "Qwen3.5-9B":    {"params": 9,    "hidden": 4096, "layers": 36, "hybrid": True},
    "Qwen3.5-27B":   {"params": 27,   "hidden": 5120, "layers": 64, "hybrid": True},
    # Llama 2
    "Llama2-7B":     {"params": 7,    "hidden": 4096, "layers": 32},
    "Llama2-13B":    {"params": 13,   "hidden": 5120, "layers": 40},
    "Llama2-70B":    {"params": 70,   "hidden": 8192, "layers": 80},
    # Llama 3 / 3.1
    "Llama3-1B":     {"params": 1,    "hidden": 2048, "layers": 16},
    "Llama3-3B":     {"params": 3,    "hidden": 3072, "layers": 28},
    "Llama3-8B":     {"params": 8,    "hidden": 4096, "layers": 32},
    "Llama3-70B":    {"params": 70,   "hidden": 8192, "layers": 80},
    # Mistral / Mixtral
    "Mistral-7B":    {"params": 7,    "hidden": 4096, "layers": 32},
    "Mixtral-8x7B":  {"params": 47,   "hidden": 4096, "layers": 32, "moe": True},
    "Mixtral-8x22B": {"params": 141,  "hidden": 6144, "layers": 56, "moe": True},
    # DeepSeek
    "DeepSeek-7B":   {"params": 7,    "hidden": 4096, "layers": 30},
    "DeepSeek-R1-8B":{"params": 8,    "hidden": 4096, "layers": 32},
    "DeepSeek-67B":  {"params": 67,   "hidden": 8192, "layers": 95},
    # Phi
    "Phi3-3.8B":     {"params": 3.8,  "hidden": 3072, "layers": 32},
    "Phi4-14B":      {"params": 14,   "hidden": 5120, "layers": 40},
    # Gemma
    "Gemma2-9B":     {"params": 9,    "hidden": 3584, "layers": 42},
    "Gemma3-27B":    {"params": 27,   "hidden": 5120, "layers": 62},
    # GLM
    "GLM4-9B":       {"params": 9,    "hidden": 4096, "layers": 40},
    "GLM4-32B":      {"params": 32,   "hidden": 5120, "layers": 60},
    # Others
    "InternLM2-7B":  {"params": 7,    "hidden": 4096, "layers": 32},
    "InternLM2-20B": {"params": 20,   "hidden": 5120, "layers": 48},
    "Yi1.5-6B":      {"params": 6,    "hidden": 4096, "layers": 32},
    "Yi1.5-9B":      {"params": 9,    "hidden": 4096, "layers": 48},
    "Yi1.5-34B":     {"params": 34,   "hidden": 7168, "layers": 60},
    "GPT-OSS-20B":   {"params": 20,   "hidden": 5120, "layers": 48},
    "GPT-OSS-120B":  {"params": 120,  "hidden": 10240,"layers": 96},
    # RWKV-7 x070 examples
    "RWKV7-0.1B":    {"params": 0.1,  "hidden": 768,  "layers": 12, "arch": "rwkv7", "head_size": 64},
    "RWKV7-0.4B":    {"params": 0.4,  "hidden": 1024, "layers": 24, "arch": "rwkv7", "head_size": 64},
    "RWKV7-1.5B":    {"params": 1.5,  "hidden": 2048, "layers": 24, "arch": "rwkv7", "head_size": 64},
    "RWKV7-3B":      {"params": 3,    "hidden": 2560, "layers": 32, "arch": "rwkv7", "head_size": 64},
    "RWKV7-7B":      {"params": 7,    "hidden": 4096, "layers": 32, "arch": "rwkv7", "head_size": 64},
}

# ============================================================
# Reference benchmark: Qwen2.5-7B on 80GB H100, seq=1024
# Calibrated activation cost per sample per checkpoint segment
# ============================================================
# From actual run: 80GB GPU, fixed ~16GB, available ~64GB, bs=148
# activation_per_sample = hidden * seq * ckpt_interval * ACTIVATION_BYTES_FACTOR
# 64GB / 148 / (3584 * 1024 * 4) = ~29 bytes per hidden*seq*ckpt element
ACTIVATION_BYTES_FACTOR = 29
CUDA_CONTEXT_GB = 2.0
DEFAULT_CHECKPOINT_INTERVAL = 4
DEFAULT_NUM_GRAD_SLABS = 12
HYBRID_OVERHEAD_FACTOR = 2.5  # hybrid attention uses ~2.5x more activation memory
RWKV7_SAFETY_FACTOR = 1.25
DEFAULT_RWKV7_HEAD_SIZE = 64
DEFAULT_RWKV7_CHUNK_LEN = 16


def model_tag(spec: dict) -> str:
    tags = []
    if spec.get("arch") == "rwkv7":
        tags.append("RWKV7")
    if spec.get("hybrid"):
        tags.append("hybrid")
    if spec.get("moe"):
        tags.append("MoE")
    return f" [{', '.join(tags)}]" if tags else ""


def calc_cpu_memory(params_billion: float) -> float:
    """Calculate required CPU memory in GB.

    MegaTrain stores on CPU:
      - FP32 master weights: 4 bytes/param
      - Adam optimizer states (m + v): 8 bytes/param
      - Total: 12 bytes/param
    """
    return params_billion * 12


def calc_gpu_fixed(params_billion: float, num_layers: int,
                   num_grad_slabs: int = DEFAULT_NUM_GRAD_SLABS) -> float:
    """Calculate fixed GPU memory usage in GB.

    Fixed components:
      - CUDA context: ~2 GB
      - Double-buffered layer templates: 2 * layer_size_bf16
      - Gradient slabs: num_grad_slabs * layer_size_fp32
      - Head + Embed gradient slabs: ~2 * vocab_layer_size
    """
    layer_params = params_billion * 1e9 / num_layers
    layer_bf16_gb = layer_params * 2 / 1e9
    layer_fp32_gb = layer_params * 4 / 1e9

    # Double buffer
    double_buf = 2 * layer_bf16_gb
    # Gradient slabs
    grad_slabs = num_grad_slabs * layer_fp32_gb
    # Head + Embed (rough estimate: ~2x largest layer each)
    head_embed = 2 * layer_fp32_gb

    return CUDA_CONTEXT_GB + double_buf + grad_slabs + head_embed


def calc_activation_per_sample_gb(
    hidden_dim: int,
    num_layers: int,
    seq_len: int,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    arch: str = "transformer",
    is_hybrid: bool = False,
    head_size: int = DEFAULT_RWKV7_HEAD_SIZE,
    chunk_len: int = DEFAULT_RWKV7_CHUNK_LEN,
    activation_offload: str = "none",
    dtype_bytes: int = 2,
) -> float:
    """Estimate batch-size-scaled GPU memory per sample in GB."""
    if arch == "rwkv7":
        # Current CPUMaster RWKV7 path keeps block checkpoints on GPU as
        # (hidden, v_first), then stores a block-local recompute cache with the
        # same pair. The fused recurrence also saves per-layer bf16 inputs plus
        # fp32 state/sa scratch for backward.
        checkpoints = math.ceil(num_layers / checkpoint_interval) + 1
        if activation_offload == "cpu":
            # CPU offload removes long-lived checkpoint residency, but the peak
            # still includes final/current materialized checkpoints and CUDA
            # allocator overlap during recompute. Empirically this retains about
            # 70% of the no-offload checkpoint term for the 0.1B RWKV7 probe.
            checkpoint_bytes = checkpoints * 2 * dtype_bytes * 0.70
        else:
            checkpoint_bytes = checkpoints * 2 * dtype_bytes
        recompute_bytes = checkpoint_interval * 2 * dtype_bytes
        recurrence_bytes = (
            6 * dtype_bytes
            + 4  # sa: [B,T,C] fp32
            + 4 * head_size / max(chunk_len, 1)  # state: [B,H,T/chunk,N,N] fp32
        )
        per_hidden_token = (
            checkpoint_bytes + recompute_bytes + recurrence_bytes
        ) * RWKV7_SAFETY_FACTOR
        return hidden_dim * seq_len * per_hidden_token / 1e9

    activation_per_sample = (
        hidden_dim * seq_len * checkpoint_interval * ACTIVATION_BYTES_FACTOR / 1e9
    )

    if is_hybrid:
        activation_per_sample *= HYBRID_OVERHEAD_FACTOR

    return activation_per_sample


def calc_batch_size(gpu_memory_gb: float, params_billion: float,
                    hidden_dim: int, num_layers: int, seq_len: int,
                    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
                    is_hybrid: bool = False,
                    arch: str = "transformer",
                    head_size: int = DEFAULT_RWKV7_HEAD_SIZE,
                    chunk_len: int = DEFAULT_RWKV7_CHUNK_LEN,
                    activation_offload: str = "none") -> int:
    """Calculate maximum batch size."""
    fixed = calc_gpu_fixed(params_billion, num_layers)
    available = gpu_memory_gb - fixed

    if available <= 0:
        return 0

    activation_per_sample = calc_activation_per_sample_gb(
        hidden_dim,
        num_layers,
        seq_len,
        checkpoint_interval=checkpoint_interval,
        arch=arch,
        is_hybrid=is_hybrid,
        head_size=head_size,
        chunk_len=chunk_len,
        activation_offload=activation_offload,
    )
    if activation_per_sample <= 0:
        return 0

    bs = int(available / activation_per_sample)
    return max(bs, 1)


def print_header(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def mode_cpu_memory():
    """Mode 1: Given CPU memory, show trainable models."""
    print_header("CPU Memory -> Trainable Models")
    print()

    while True:
        try:
            cpu_mem = float(input("  Enter your CPU memory (GB): "))
            break
        except ValueError:
            print("  Please enter a valid number.")

    print()
    print(f"  Your CPU memory: {cpu_mem:.0f} GB")
    print(f"  Rule: MegaTrain needs ~12x model BF16 size in CPU RAM")
    print(f"        (FP32 master weights + Adam optimizer states)")
    print()

    max_params = cpu_mem / 12
    print(f"  Maximum model size: ~{max_params:.1f}B parameters")
    print()

    # Group by trainable / not trainable
    trainable = []
    too_large = []

    for name, spec in sorted(KNOWN_MODELS.items(), key=lambda x: x[1]["params"]):
        needed = calc_cpu_memory(spec["params"])
        if needed <= cpu_mem:
            trainable.append((name, spec["params"], needed))
        else:
            too_large.append((name, spec["params"], needed))

    if trainable:
        print(f"  {'Model':<22} {'Params':>8} {'CPU Needed':>12} {'Status':>10}")
        print(f"  {'-'*22} {'-'*8} {'-'*12} {'-'*10}")
        for name, params, needed in trainable:
            print(f"  {name:<22} {params:>6.1f}B {needed:>9.0f} GB {'OK':>10}")

    if too_large:
        print()
        print(f"  --- Models that need more CPU memory ---")
        for name, params, needed in too_large[:5]:
            print(f"  {name:<22} {params:>6.1f}B {needed:>9.0f} GB {'NEED MORE':>10}")
        if len(too_large) > 5:
            print(f"  ... and {len(too_large) - 5} more")


def mode_batch_size():
    """Mode 2: Given GPU memory, model, and context length, calculate batch size."""
    print_header("GPU Memory + Model -> Batch Size")
    print()

    # GPU memory
    while True:
        try:
            gpu_mem = float(input("  Enter your GPU memory (GB, e.g. 40/80): "))
            break
        except ValueError:
            print("  Please enter a valid number.")

    # Context length
    while True:
        try:
            seq_len = int(input("  Enter context length (e.g. 1024/2048/4096): "))
            break
        except ValueError:
            print("  Please enter a valid number.")

    # Model selection
    print()
    print("  Available models:")
    model_list = sorted(KNOWN_MODELS.items(), key=lambda x: x[1]["params"])
    for i, (name, spec) in enumerate(model_list):
        print(f"    {i+1:>3}. {name} ({spec['params']}B){model_tag(spec)}")

    print(f"    {len(model_list)+1:>3}. Custom (enter params manually)")
    print()

    while True:
        try:
            choice = int(input("  Select model number: "))
            if 1 <= choice <= len(model_list) + 1:
                break
        except ValueError:
            pass
        print("  Invalid choice, try again.")

    if choice <= len(model_list):
        name, spec = model_list[choice - 1]
        params = spec["params"]
        hidden = spec["hidden"]
        layers = spec["layers"]
        arch = spec.get("arch", "transformer")
        head_size = spec.get("head_size", DEFAULT_RWKV7_HEAD_SIZE)
        chunk_len = spec.get("chunk_len", DEFAULT_RWKV7_CHUNK_LEN)
        is_hybrid = spec.get("hybrid", False)
        is_moe = spec.get("moe", False)
    else:
        name = "Custom"
        while True:
            try:
                params = float(input("  Enter model parameters (billions): "))
                hidden = int(input("  Enter hidden dimension: "))
                layers = int(input("  Enter number of layers: "))
                break
            except ValueError:
                print("  Please enter valid numbers.")
        while True:
            arch_choice = input("  Architecture (transformer/rwkv7, default transformer): ").strip().lower()
            if arch_choice in ("", "transformer", "rwkv7"):
                break
            print("  Please enter transformer or rwkv7.")
        arch = arch_choice or "transformer"
        if arch == "rwkv7":
            while True:
                try:
                    head_size = int(input(f"  Enter RWKV7 head_size (default {DEFAULT_RWKV7_HEAD_SIZE}): ") or DEFAULT_RWKV7_HEAD_SIZE)
                    chunk_len = int(input(f"  Enter RWKV7 CUDA chunk_len (default {DEFAULT_RWKV7_CHUNK_LEN}): ") or DEFAULT_RWKV7_CHUNK_LEN)
                    break
                except ValueError:
                    print("  Please enter valid integers.")
        else:
            head_size = DEFAULT_RWKV7_HEAD_SIZE
            chunk_len = DEFAULT_RWKV7_CHUNK_LEN
        is_hybrid = False
        is_moe = False

    if arch == "rwkv7":
        while True:
            activation_offload = input("  RWKV7 activation_offload (none/cpu, default cpu): ").strip().lower() or "cpu"
            if activation_offload in ("none", "cpu"):
                break
            print("  Please enter none or cpu.")
    else:
        activation_offload = "none"

    default_ckpt = 1 if arch == "rwkv7" else DEFAULT_CHECKPOINT_INTERVAL
    while True:
        try:
            checkpoint_interval = int(input(f"  Enter checkpoint_interval (default {default_ckpt}): ") or default_ckpt)
            if checkpoint_interval > 0:
                break
        except ValueError:
            pass
        print("  Please enter a positive integer.")

    # Calculate
    cpu_needed = calc_cpu_memory(params)
    gpu_fixed = calc_gpu_fixed(params, layers)
    activation_per_sample = calc_activation_per_sample_gb(
        hidden,
        layers,
        seq_len,
        checkpoint_interval=checkpoint_interval,
        arch=arch,
        is_hybrid=is_hybrid,
        head_size=head_size,
        chunk_len=chunk_len,
        activation_offload=activation_offload,
    )
    bs = calc_batch_size(gpu_mem, params, hidden, layers, seq_len,
                         checkpoint_interval=checkpoint_interval,
                         is_hybrid=is_hybrid,
                         arch=arch,
                         head_size=head_size,
                         chunk_len=chunk_len,
                         activation_offload=activation_offload)

    print()
    print(f"  {'='*50}")
    print(f"  Model:          {name} ({params}B params)")
    print(f"  Architecture:   {arch}")
    print(f"  GPU memory:     {gpu_mem:.0f} GB")
    print(f"  Context length: {seq_len}")
    print(f"  Checkpoint int: {checkpoint_interval}")
    if arch == "rwkv7":
        print(f"  Act offload:    {activation_offload}")
    print(f"  {'='*50}")
    print()
    print(f"  CPU memory needed:   {cpu_needed:>8.0f} GB")
    print(f"  GPU fixed overhead:  {gpu_fixed:>8.1f} GB")
    print(f"  GPU for activations: {max(gpu_mem - gpu_fixed, 0):>8.1f} GB")
    print(f"  Activation/sample:   {activation_per_sample:>8.3f} GB")
    print()

    if bs <= 0:
        print(f"  *** ERROR: GPU memory too small for this model! ***")
        print(f"  The model's layer buffers and gradient slabs alone")
        print(f"  require {gpu_fixed:.1f} GB, exceeding your {gpu_mem:.0f} GB GPU.")
    else:
        print(f"  >>> Recommended batch_size: {bs}")
        print()
        # Show a range
        safe_bs = max(int(bs * 0.8), 1)
        aggressive_bs = int(bs * 1.1)
        print(f"  Safe (80%):       batch_size = {safe_bs}")
        print(f"  Recommended:      batch_size = {bs}")
        print(f"  Aggressive (110%):batch_size = {aggressive_bs}")

        if is_hybrid:
            print()
            print(f"  NOTE: {name} uses hybrid attention (linear + full).")
            print(f"  Install flash-linear-attention + causal-conv1d for best")
            print(f"  performance. Without them, actual batch size may be lower.")

        if is_moe:
            print()
            print(f"  NOTE: {name} is a MoE model. Actual memory varies by layer")
            print(f"  (dense vs expert layers). Start with the safe batch size.")
        if arch == "rwkv7":
            print()
            print(f"  NOTE: RWKV7 estimate includes hidden + v_first checkpoints,")
            print(f"  block recompute cache, and fused recurrence backward scratch.")
            print(f"  It assumes bf16 recurrence with head_size={head_size}, chunk_len={chunk_len}.")
            print(f"  activation_offload={activation_offload}.")

    print()
    print(f"  Add to your YAML config:")
    print(f"    training:")
    print(f"      batch_size: {max(bs, 1)}")
    print(f"    memory:")
    print(f"      checkpoint_interval: {checkpoint_interval}")
    if arch == "rwkv7":
        print(f"      activation_offload: {activation_offload}")
    print()


def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║       MegaTrain Resource Calculator              ║")
    print("  ║  Estimate CPU/GPU requirements & batch size      ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print("  Select mode:")
    print("    1. CPU Memory -> What models can I train?")
    print("    2. GPU Memory + Model -> What batch size to use?")
    print()

    while True:
        try:
            mode = int(input("  Enter mode (1 or 2): "))
            if mode in (1, 2):
                break
        except ValueError:
            pass
        print("  Please enter 1 or 2.")

    if mode == 1:
        mode_cpu_memory()
    else:
        mode_batch_size()

    print()
    print("  Done! For more details see: examples/configs/README.md")
    print()


if __name__ == "__main__":
    main()
