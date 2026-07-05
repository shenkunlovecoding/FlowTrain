"""Correctness tests for the fused TileLang RWKV-7 recurrence backward kernel.

Finite-difference correctness cases run in fresh subprocesses and clear the
TileLang disk cache on entry for determinism. A separate cache-regression case
keeps multiple shapes in one process to catch frontend-cache key collisions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    # single head, single segment (chunk == T)
    "short": dict(batch=1, timesteps=16, channels=64, chunk_len=16, n_fd=384),
    # two heads, two segments (chunk=64): exercises cross-chunk adjoint carry
    "multiseg": dict(batch=1, timesteps=128, channels=128, chunk_len=16, n_fd=256),
    # _pick_recurrence_chunk falls back to chunk=16 (3 segments) for T=48
    "fallback_chunk": dict(batch=1, timesteps=48, channels=64, chunk_len=16, n_fd=256),
    # longer scan, 4 segments
    "long": dict(batch=2, timesteps=256, channels=64, chunk_len=16, n_fd=160),
}

WORKER = r'''
import shutil, os, sys
from pathlib import Path

# Clear TileLang disk cache so a stale binary from another shape cannot be served.
for p in (Path.home() / ".tilelang",):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

import torch
from flowtrain.tilelang_recurrence import rwkv7_recurrence_tilelang, _torch_ref_from_raw_w
from flowtrain.tilelang_recurrence import _pick_recurrence_chunk

HEAD = 64
NAMES = ("r", "raw_w", "k", "v", "a", "b")


def run(batch, timesteps, channels, chunk_len, seed, n_fd):
    assert torch.cuda.is_available(), "no cuda"
    torch.manual_seed(seed)
    scale = 0.3

    def rnd():
        return torch.randn(batch, timesteps, channels, device="cuda", dtype=torch.bfloat16) * scale

    inputs = tuple(rnd() for _ in NAMES)

    # fused forward + backward
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = rwkv7_recurrence_tilelang(*leaves, HEAD, chunk_len=chunk_len)
    assert out.shape == (batch, timesteps, channels)
    gout = torch.randn_like(out) * scale
    fused = torch.autograd.grad(out, tuple(leaves), gout, allow_unused=True)
    fused = [(g if g is not None else torch.zeros_like(t)) for g, t in zip(fused, inputs)]

    # fp32 reference forward (no autograd) is the ground-truth function for FD
    def ref_fp(args):
        with torch.no_grad():
            return _torch_ref_from_raw_w(*[a.float() for a in args], HEAD)

    fp32 = [t.float() for t in inputs]
    fp32_out = ref_fp(fp32)

    # fused forward must match the fp32 reference within bf16 precision
    fwd_rel = (out.float() - fp32_out).norm().item() / max(fp32_out.norm().item(), 1e-9)
    assert fwd_rel < 0.05, f"forward rel diff {fwd_rel:.4f}"

    # finite-difference ground truth, sampled per grad. FD is authoritative here:
    # the bf16 autograd reference itself drifts at long T (both are correct vs FD
    # but differ from each other), so we compare fused directly to FD.
    eps = 1e-2
    gout_f = gout.float()
    for i in range(6):
        gfd = torch.zeros_like(fp32[i])
        flat_numel = fp32[i].numel()
        idxs = torch.randperm(flat_numel)[: min(n_fd, flat_numel)]
        for idx in idxs.tolist():
            ap = [t.clone() for t in fp32]
            am = [t.clone() for t in fp32]
            ap[i].view(-1)[idx] += eps
            am[i].view(-1)[idx] -= eps
            lp = (gout_f * ref_fp(ap)).sum()
            lm = (gout_f * ref_fp(am)).sum()
            gfd.view(-1)[idx] = (lp - lm) / (2 * eps)
        mask = gfd.abs() > 1e-3
        assert mask.any(), f"grad {NAMES[i]} FD all below threshold"
        rel = (fused[i].float()[mask] - gfd[mask]).norm().item() / max(gfd[mask].norm().item(), 1e-9)
        assert rel < 0.06, f"FD grad {NAMES[i]} rel {rel:.4f}"


# chunk picker is pure python; check it here too
assert _pick_recurrence_chunk(16) == 16
assert _pick_recurrence_chunk(128) == 128
assert _pick_recurrence_chunk(128, target=64) == 64
assert _pick_recurrence_chunk(48) == 16
assert _pick_recurrence_chunk(8) == 8

tag, batch, timesteps, channels, chunk_len, seed, n_fd = sys.argv[1:8]
run(int(batch), int(timesteps), int(channels), int(chunk_len), int(seed), int(n_fd))
print(tag, "OK")
'''


CACHE_WORKER = r'''
import shutil
from pathlib import Path

for p in (Path.home() / ".tilelang",):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

import torch
from flowtrain.tilelang_recurrence import rwkv7_recurrence_tilelang

HEAD = 64
NAMES = ("r", "raw_w", "k", "v", "a", "b")


def run(batch, timesteps, channels, seed):
    assert torch.cuda.is_available(), "no cuda"
    torch.manual_seed(seed)
    inputs = [
        (torch.randn(batch, timesteps, channels, device="cuda", dtype=torch.bfloat16) * 0.1)
        .detach()
        .requires_grad_(True)
        for _ in NAMES
    ]
    out = rwkv7_recurrence_tilelang(*inputs, HEAD, chunk_len=16)
    assert out.shape == (batch, timesteps, channels)
    grads = torch.autograd.grad(out.float().sum(), tuple(inputs), allow_unused=True)
    for grad, tensor in zip(grads, inputs):
        assert grad is not None
        assert grad.shape == tensor.shape


run(1, 16, 64, 0)
run(1, 32, 256, 1)
run(1, 16, 64, 2)
print("cache-regression OK")
'''


def _run_case_subprocess(tag: str, case: dict) -> None:
    if not torch_cuda_available():
        return  # silently skip when there is no GPU
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c", WORKER, tag, str(case["batch"]), str(case["timesteps"]),
           str(case["channels"]), str(case["chunk_len"]), str(case.get("seed", 0)),
           str(case.get("n_fd", 256))]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"{tag} failed (rc={proc.returncode}):\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def _run_cache_regression_subprocess() -> None:
    if not torch_cuda_available():
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", CACHE_WORKER], env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"cache regression failed (rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def torch_cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def test_fused_backward_short():
    _run_case_subprocess("short", CASES["short"])


def test_fused_backward_multiseg():
    _run_case_subprocess("multiseg", CASES["multiseg"])


def test_fused_backward_fallback_chunk():
    _run_case_subprocess("fallback_chunk", CASES["fallback_chunk"])


def test_fused_backward_long():
    _run_case_subprocess("long", CASES["long"])


def test_pick_recurrence_chunk():
    from flowtrain.tilelang_recurrence import _pick_recurrence_chunk

    assert _pick_recurrence_chunk(16) == 16
    assert _pick_recurrence_chunk(128) == 128     # default target is one larger segment
    assert _pick_recurrence_chunk(128, target=64) == 64  # explicit legacy 2 segments of 64
    assert _pick_recurrence_chunk(48) == 16       # divisor fallback, 3 segments
    assert _pick_recurrence_chunk(8) == 8         # smaller than target -> single segment
    assert _pick_recurrence_chunk(64) == 64


def test_recurrence_cache_accepts_multiple_shapes_in_one_process():
    _run_cache_regression_subprocess()


if __name__ == "__main__":
    test_pick_recurrence_chunk()
    print("pick_recurrence_chunk ok")
    test_recurrence_cache_accepts_multiple_shapes_in_one_process()
    print("cache regression ok")
    for tag in ("short", "multiseg", "fallback_chunk", "long"):
        _run_case_subprocess(tag, CASES[tag])
        print(tag, "ok")
    print("recurrence backward smoke ok")
