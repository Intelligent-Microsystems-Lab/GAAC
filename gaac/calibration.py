"""Offline calibration of the per-(layer, kv head, channel) exponent bases.

Accumulates exponent histograms of the K/V a model produces during prefill and keeps, per channel, the start of the
densest 8-wide (3-bit) and 16-wide (4-bit) exponent window.  Calibrate on prompts as long as the deployment context:
RoPE makes the K exponent distribution drift with position.  The bases file is {layer: {k3, k4, v3, v4}} with uint8
[n_kv, 128] tables (4 KiB per layer).
"""
from __future__ import annotations
import torch
from .codec import D, bases_from_hist, exponent_hist


class BaseCollector:
    def __init__(self):
        self.hist = {}      # layer -> {'k': int64 [n_kv*D, 256], 'v': ...}

    def observe(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [N, n_kv, D] bf16 of one prefill step."""
        n_kv = k.shape[1]
        h = self.hist.setdefault(layer, {"k": 0, "v": 0, "n_kv": n_kv})
        h["k"] = h["k"] + exponent_hist(k, n_kv)
        h["v"] = h["v"] + exponent_hist(v, n_kv)

    def bases(self) -> dict:
        out = {}
        for layer, h in sorted(self.hist.items()):
            k3, k4 = bases_from_hist(h["k"], h["n_kv"]); v3, v4 = bases_from_hist(h["v"], h["n_kv"])
            out[layer] = dict(k3=k3.cpu(), k4=k4.cpu(), v3=v3.cpu(), v4=v4.cpu())
        return out

    def save(self, path: str) -> int:
        b = self.bases(); torch.save(b, path); return len(b)
