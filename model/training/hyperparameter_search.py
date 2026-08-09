"""Grid search over the architecture-specific hyperparameter grid.

Runs short trainings (trial_epochs) on a subset of the training data and picks
the configuration with the lowest validation loss. The best configuration is
saved to results/hparam_search/best_<arch>.json and consumed by train.py.

Usage:
    python -m model.training.hyperparameter_search --arch transformer
"""

from __future__ import annotations

import argparse
import itertools
import random

import torch
from torch.utils.data import DataLoader, Subset

from common.config import load_config, pick_device, resolve_path, set_seed
from common.results import RESULTS_DIR, create_run_dir, save_json
from model.architectures.factory import build_model, split_hparams
from model.training.dataset import TranslationDataset, load_tokenizer, make_collate_fn
from model.training.trainer import Trainer


def grid_combinations(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def make_subset_loaders(cfg, tokenizer, hparams, device):
    search_cfg = cfg["hyperparameter_search"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    dataset = TranslationDataset(
        processed_dir / "train.tsv", tokenizer, cfg["training"]["max_sequence_length"]
    )

    subset_size = int(len(dataset) * search_cfg["subset_fraction"])
    indices = random.sample(range(len(dataset)), subset_size)
    val_size = min(search_cfg["trial_validation_size"], subset_size // 10)
    train_subset = Subset(dataset, indices[val_size:])
    val_subset = Subset(dataset, indices[:val_size])

    collate = make_collate_fn(tokenizer.pad_id())
    loader_kwargs = {
        "batch_size": hparams["batch_size"],
        "collate_fn": collate,
        "num_workers": cfg["training"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_subset, shuffle=True, **loader_kwargs),
        DataLoader(val_subset, shuffle=False, **loader_kwargs),
    )


def run_trial(cfg, tokenizer, arch: str, hparams: dict, device) -> float:
    set_seed(cfg["seed"])  # identical init/data order across trials
    train_loader, val_loader = make_subset_loaders(cfg, tokenizer, hparams, device)
    model = build_model(arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams)
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        pad_id=tokenizer.pad_id(),
        device=device,
        learning_rate=hparams["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        label_smoothing=cfg["training"]["label_smoothing"],
        grad_clip=cfg["training"]["grad_clip"],
        epochs=cfg["hyperparameter_search"]["trial_epochs"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        quiet=True,
    )
    return trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["gru", "lstm", "transformer"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    tokenizer = load_tokenizer(cfg)
    grid = cfg["hyperparameter_search"]["grids"][args.arch]
    combinations = grid_combinations(grid)
    print(f"Grid search for {args.arch}: {len(combinations)} combinations on {device}")

    run_dir = create_run_dir("hparam_search", args.arch)
    trials = []
    for i, hparams in enumerate(combinations, 1):
        val_loss = run_trial(cfg, tokenizer, args.arch, hparams, device)
        trials.append({"hparams": hparams, "val_loss": val_loss})
        print(f"[{i}/{len(combinations)}] val_loss={val_loss:.4f} {hparams}")
        save_json(run_dir / "trials.json", {"arch": args.arch, "trials": trials})

    best = min(trials, key=lambda t: t["val_loss"])
    best_record = {"arch": args.arch, **best}
    save_json(run_dir / "best.json", best_record)
    save_json(RESULTS_DIR / "hparam_search" / f"best_{args.arch}.json", best_record)
    print(f"Best {args.arch}: val_loss={best['val_loss']:.4f} {best['hparams']}")
    print(f"Saved to {RESULTS_DIR / 'hparam_search' / f'best_{args.arch}.json'}")


if __name__ == "__main__":
    main()
