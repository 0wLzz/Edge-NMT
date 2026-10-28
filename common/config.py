"""Central config loading and path resolution."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEFAULT_KD_FILE = "kd_train.tsv"


def getting_output_folder_name(args, target_sparsity: float) -> str:
    data_mode = args.data_mode
    # KD data from another teacher gets its own name (kd_train_nllb-1.3B.tsv ->
    # "kd-nllb-1.3B"), so its runs never resume or mix with the default kd_train.tsv runs.
    kd_file = getattr(args, "kd_file", DEFAULT_KD_FILE)
    if data_mode == "kd" and kd_file != DEFAULT_KD_FILE:
        data_mode += "-" + Path(kd_file).stem.removeprefix("kd_train").lstrip("_-")
    run_name = f"{args.arch}_{data_mode}" + ("_qat" if args.qat else "")
    if args.prune:
        run_name += f"_prune{int(round(target_sparsity * 100))}"

    return run_name