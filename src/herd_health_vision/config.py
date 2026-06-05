"""Configuration loading and device (Apple M4 / MPS) selection."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# MPS'te desteklenmeyen nadir operasyonlar CPU'ya düşsün
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device() -> str:
    """Apple M4 (MPS) > CUDA > CPU önceliğiyle cihaz seçer."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


CONFIG = load_config()
