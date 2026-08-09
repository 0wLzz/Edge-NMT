"""Helpers for writing run outputs into results/ with a comparable history."""

import csv
import json
from datetime import datetime
from pathlib import Path

from common.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_COLUMNS = [
    "timestamp",
    "run_name",
    "arch",
    "data_mode",
    "qat",
    "quantized",
    "checkpoint",
    "bleu",
    "chrf",
    "meteor",
    "params_millions",
    "model_size_mb",
    "latency_ms_per_sentence",
    "notes",
]


def create_run_dir(stage: str, name: str) -> Path:
    """Create results/<stage>/<timestamp>_<name>/ and return it."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / stage / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def append_summary(row: dict) -> Path:
    """Append one evaluation row to results/summary.csv for cross-run comparison."""
    summary_path = RESULTS_DIR / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    return summary_path
