"""Optuna hyperparameter search over the architecture-specific search space.

Replaces the old exhaustive grid search. An Optuna study per architecture uses a
TPE sampler to propose configurations and a Hyperband pruner to stop unpromising
trials early, so limited compute is spent where it matters. The objective is the
best validation cross-entropy loss of a short training run.

Every trial is held identical except the sampled hyperparameters:
  * same seed (model init + data order),
  * same training slice: the top `subset_size` rows of train.tsv -- CCMatrix is
    LASER-margin-sorted, so these are the highest-quality pairs, and the fixed
    slice makes trials directly comparable,
  * same validation slice: the first `trial_validation_size` rows of valid.tsv,
  * same epochs (`trial_epochs`) and objective (validation loss).

The study is stored in a resumable SQLite file
(results/hparam_search/<arch>.db). `n_trials` is a *total* budget: rerunning the
command continues the same study until that many trials have finished, so Colab
session drops don't lose progress. The winning configuration is written to
results/hparam_search/best_<arch>.json (same schema as before) and consumed by
train.py.

Per trial, a torchinfo layer/parameter summary is written to
results/hparam_search/<arch>/trial_<n>_summary.txt and a per-module parameter
breakdown is attached to the trial's user_attrs. Optionally a graphviz diagram of
the model is rendered (config `hyperparameter_search.save_diagram`: best_only |
all | none; needs torchview).

Usage:
    python -m model.training.hyperparameter_search --arch transformer
    python -m model.training.hyperparameter_search --arch gru --n-trials 5
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from common.config import load_config, pick_device, resolve_path, set_seed
from common.results import RESULTS_DIR, save_json
from model.architectures.factory import build_model
from model.training.dataset import TranslationDataset, load_tokenizer, make_collate_fn
from model.training.trainer import Trainer


# --------------------------------------------------------------------------- #
# Search-space parsing
# --------------------------------------------------------------------------- #
def _is_spec(value, key: str) -> bool:
    return isinstance(value, dict) and key in value


def sample_hparams(trial: optuna.Trial, space: dict) -> dict:
    """Turn a config search_space into concrete hyperparameters via Optuna.

    Spec forms (see config.yaml):
      [a, b, ...]              -> categorical choice
      {int: [lo, hi]}          -> integer range (inclusive)
      {loguniform: [lo, hi]}   -> float, log-uniform
      {uniform: [lo, hi]}      -> float, uniform
    """
    hparams = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            hparams[name] = trial.suggest_categorical(name, spec)
        elif _is_spec(spec, "int"):
            lo, hi = spec["int"]
            hparams[name] = trial.suggest_int(name, lo, hi)
        elif _is_spec(spec, "loguniform"):
            lo, hi = spec["loguniform"]
            hparams[name] = trial.suggest_float(name, lo, hi, log=True)
        elif _is_spec(spec, "uniform"):
            lo, hi = spec["uniform"]
            hparams[name] = trial.suggest_float(name, lo, hi)
        else:
            raise ValueError(f"Unrecognised search-space spec for {name!r}: {spec!r}")
    return hparams


def default_hparams(space: dict) -> dict:
    """Deterministic default config from a search space (first/low value of each
    spec). Used as the fallback in train.py when no search results exist yet."""
    defaults = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            defaults[name] = spec[0]
        elif _is_spec(spec, "int"):
            defaults[name] = spec["int"][0]
        elif _is_spec(spec, "loguniform"):
            defaults[name] = spec["loguniform"][0]
        elif _is_spec(spec, "uniform"):
            defaults[name] = spec["uniform"][0]
        else:
            raise ValueError(f"Unrecognised search-space spec for {name!r}: {spec!r}")
    return defaults


# --------------------------------------------------------------------------- #
# Data (identical slice for every trial)
# --------------------------------------------------------------------------- #
def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_subset_loaders(cfg, tokenizer, batch_size: int, device, seed: int):
    """Top-N train slice + fixed val slice, deterministic across trials."""
    search_cfg = cfg["hyperparameter_search"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    max_length = cfg["training"]["max_sequence_length"]

    train_dataset = TranslationDataset(
        processed_dir / "train.tsv", tokenizer, max_length,
        limit=search_cfg["subset_size"],
    )
    val_dataset = TranslationDataset(
        processed_dir / "valid.tsv", tokenizer, max_length,
        limit=search_cfg["trial_validation_size"],
    )

    collate = make_collate_fn(tokenizer.pad_id())
    generator = torch.Generator().manual_seed(seed)  # identical shuffle each trial
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": cfg["training"]["num_workers"],
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# Model info logging (the "architecture diagram" info, per trial)
# --------------------------------------------------------------------------- #
def module_param_breakdown(model: torch.nn.Module) -> dict:
    """Parameters per top-level submodule (embedding / encoder / decoder / ...),
    plus the total. This is the breakdown REVISI §5.2 needs before choosing
    pruning ratios, captured for free during the search."""
    breakdown = {
        name: sum(p.numel() for p in child.parameters())
        for name, child in model.named_children()
    }
    breakdown["total"] = sum(p.numel() for p in model.parameters())
    return breakdown


def log_model_info(model, arch: str, hparams: dict, trial, artifacts_dir, save_diagram: str):
    """Write a torchinfo summary + attach the param breakdown to the trial. Any
    failure here is non-fatal -- it must never sink a training trial."""
    breakdown = module_param_breakdown(model)
    trial.set_user_attr("param_breakdown", breakdown)
    trial.set_user_attr("total_params", breakdown["total"])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts_dir / f"trial_{trial.number}_summary.txt"
    try:
        from torchinfo import summary

        src = torch.randint(4, 100, (2, 16), dtype=torch.long)
        tgt = torch.randint(4, 100, (2, 15), dtype=torch.long)
        info = summary(
            model, input_data=[src, tgt], device="cpu", verbose=0,
            col_names=("input_size", "output_size", "num_params"),
        )
        summary_path.write_text(
            f"arch={arch}\nhparams={hparams}\n\n{info}\n\n"
            f"param_breakdown={breakdown}\n"
        )
    except Exception as exc:  # torchinfo missing, or a model that resists tracing
        summary_path.write_text(
            f"arch={arch}\nhparams={hparams}\n\n"
            f"[torchinfo summary unavailable: {exc}]\n\n"
            f"param_breakdown={breakdown}\n"
        )

    if save_diagram == "all":
        _render_diagram(model, arch, artifacts_dir / f"trial_{trial.number}_model")


def _render_diagram(model, arch: str, out_stem) -> None:
    """Optional graphviz diagram (needs torchview + graphviz)."""
    try:
        from torchview import draw_graph

        src = torch.randint(4, 100, (1, 16), dtype=torch.long)
        tgt = torch.randint(4, 100, (1, 15), dtype=torch.long)
        g = draw_graph(model, input_data=[src, tgt], device="cpu", expand_nested=True)
        g.visual_graph.render(str(out_stem), format="png", cleanup=True)
    except Exception as exc:
        print(f"[diagram] skipped ({exc}); pip install torchview + graphviz to enable")


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def make_objective(cfg, tokenizer, arch: str, device, artifacts_dir, save_diagram: str):
    search_cfg = cfg["hyperparameter_search"]
    space = search_cfg["search_space"][arch]
    seed = cfg["seed"]

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed)  # identical init + data order; only hparams vary
        hparams = sample_hparams(trial, space)

        train_loader, val_loader = make_subset_loaders(
            cfg, tokenizer, hparams["batch_size"], device, seed
        )
        model = build_model(arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams)
        log_model_info(model, arch, hparams, trial, artifacts_dir, save_diagram)

        def epoch_callback(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

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
            epochs=search_cfg["trial_epochs"],
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
            quiet=True,
            scheduler_cfg=cfg["training"].get("scheduler"),
            epoch_callback=epoch_callback,
        )
        return trainer.train()

    return objective


# --------------------------------------------------------------------------- #
# Study construction
# --------------------------------------------------------------------------- #
def build_study(arch: str, cfg: dict) -> optuna.Study:
    search_cfg = cfg["hyperparameter_search"]
    seed = cfg["seed"]

    sampler_name = search_cfg.get("sampler", "tpe")
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler {sampler_name!r} (tpe | random)")

    pruner_name = search_cfg.get("pruner", "hyperband")
    if pruner_name == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=search_cfg["trial_epochs"], reduction_factor=3
        )
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    elif pruner_name == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner {pruner_name!r} (hyperband | median | none)")

    db_path = RESULTS_DIR / "hparam_search" / f"{arch}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=arch,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )


def _finished(study: optuna.Study) -> int:
    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(1 for t in study.trials if t.state in done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["gru", "lstm", "transformer"])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override the total-trial budget from config (useful for smoke tests)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    tokenizer = load_tokenizer(cfg)
    search_cfg = cfg["hyperparameter_search"]

    target_total = args.n_trials if args.n_trials is not None else search_cfg["n_trials"]
    artifacts_dir = RESULTS_DIR / "hparam_search" / args.arch
    save_diagram = search_cfg.get("save_diagram", "best_only")

    study = build_study(args.arch, cfg)
    remaining = max(0, target_total - _finished(study))
    print(
        f"Optuna search for {args.arch} on {device}: "
        f"{_finished(study)}/{target_total} trials done, running {remaining} more"
    )

    if remaining > 0:
        objective = make_objective(
            cfg, tokenizer, args.arch, device, artifacts_dir, save_diagram
        )
        study.optimize(objective, n_trials=remaining)

    best = study.best_trial
    best_record = {
        "arch": args.arch,
        "hparams": best.params,
        "val_loss": best.value,
        "trial_number": best.number,
        "total_params": best.user_attrs.get("total_params"),
        "param_breakdown": best.user_attrs.get("param_breakdown"),
        "n_finished": _finished(study),
    }
    save_json(RESULTS_DIR / "hparam_search" / f"best_{args.arch}.json", best_record)

    if save_diagram in ("best_only", "all"):
        set_seed(cfg["seed"])
        best_model = build_model(
            args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), best.params
        )
        _render_diagram(best_model, args.arch, artifacts_dir / f"best_{args.arch}_model")

    print(f"Best {args.arch}: val_loss={best.value:.4f} {best.params}")
    print(f"Saved to {RESULTS_DIR / 'hparam_search' / f'best_{args.arch}.json'}")


if __name__ == "__main__":
    main()
