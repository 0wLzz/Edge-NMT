"""Efficiency metrics: parameter count, on-disk size, inference latency.

Latency here is the PyTorch (development) measurement; the thesis-grade
on-device numbers come from the iOS app with the converted CoreML model.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        return round(Path(tmp.name).stat().st_size / 1e6, 2)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample_sources: list[torch.Tensor],
    bos_id: int,
    eos_id: int,
    max_length: int,
    warmup: int = 5,
    repeats: int = 30,
) -> dict:
    """Greedy-decode single sentences on CPU and report ms/sentence."""
    model = model.to("cpu").eval()
    sample_sources = [s.unsqueeze(0) for s in sample_sources]

    for source in sample_sources[:warmup]:
        model.greedy_decode(source, bos_id, eos_id, max_length)

    timings = []
    for i in range(repeats):
        source = sample_sources[i % len(sample_sources)]
        start = time.perf_counter()
        model.greedy_decode(source, bos_id, eos_id, max_length)
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "latency_ms_per_sentence": round(statistics.mean(timings), 1),
        "latency_ms_std": round(statistics.stdev(timings), 1) if len(timings) > 1 else 0.0,
        "latency_repeats": repeats,
    }
