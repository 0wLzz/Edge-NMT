"""Full training entry point for the student models.

Data modes:
  baseline - train on the cleaned reference pairs (train.tsv)
  kd       - sequence-level knowledge distillation: train on the teacher's
             translations (kd_train.tsv, from generate_kd_dataset.py)

QAT can be toggled on with --qat (asymmetric uint8 fake quantization on all
Linear layers, see model/qat.py). Pruning can be toggled on with --prune
(gradual magnitude pruning of the same Linear layers, see model/pruning.py).
Combining --data-mode kd + --qat + --prune is the one-shot KD + quantize +
prune setup: all three compress the student in a single training run.

Resumable: every epoch writes last.pt (weights + optimizer + early-stopping
state). `--resume` with no value finds the latest run matching the same
arch/data-mode/qat/prune and continues it (starting fresh if none exists), so
the same command is safe to rerun daily on Colab. `--resume <run_dir>` targets
a specific run.

Usage:
    python -m model.training.train --arch transformer --data-mode kd --qat
    python -m model.training.train --arch transformer --data-mode kd --qat --prune
    python -m model.training.train --arch transformer --data-mode kd --qat --prune \
        --target-sparsity 0.75 --resume
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from common.config import load_config, pick_device, resolve_path, set_seed, getting_output_folder_name
from common.results import RESULTS_DIR, create_run_dir, load_json, save_json
from model.architectures.factory import build_model
from model.pruning import MagnitudePruner
from model.training.dataset import TranslationDataset, load_tokenizer, make_collate_fn
from model.training.hyperparameter_search import default_hparams
from model.training.trainer import Trainer


def find_resume_run_dir(resume: str | None, run_name: str):
    """Resolve the --resume argument to an existing run dir (or None).

    "latest" picks the newest results/runs/<timestamp>_<run_name> that has a
    resume state; timestamps prefix the names, so lexicographic order is
    chronological.
    """
    if resume is None:
        return None
    
    if resume != "latest":
        run_dir = resolve_path(resume)
        if not (run_dir / "run_config.json").exists():
            raise SystemExit(f"No run_config.json in {run_dir}; is this a run dir?")
        return run_dir
    
    candidates = sorted(
        d
        for d in (RESULTS_DIR / "runs").glob(f"*_{run_name}")
        if (d / "last.pt").exists()
    )
    return candidates[-1] if candidates else None


def load_best_hparams(arch: str, cfg: dict, override_path: str | None) -> dict:
    """Load hyperparameters from the search results (or an explicit JSON file)."""
    if override_path:
        return load_json(resolve_path(override_path))["hparams"]
    best_path = RESULTS_DIR / "hparam_search" / f"best_{arch}.json"
    if best_path.exists():
        return load_json(best_path)["hparams"]
    # Fall back to a deterministic default drawn from the search space.
    space = cfg["hyperparameter_search"]["search_space"][arch]
    print(f"[train] No search results at {best_path}; using search-space defaults.")
    return default_hparams(space)


def make_loaders(cfg: dict, tokenizer, data_mode: str, batch_size: int, device):
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    train_file = "kd_train.tsv" if data_mode == "kd" else "train.tsv"
    max_length = cfg["training"]["max_sequence_length"]

    train_dataset = TranslationDataset(processed_dir / train_file, tokenizer, max_length)
    val_dataset = TranslationDataset(processed_dir / "valid.tsv", tokenizer, max_length)

    collate = make_collate_fn(tokenizer.pad_id())
    loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate,
        "num_workers": cfg["training"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["gru", "lstm", "transformer"])
    parser.add_argument("--data-mode", default="baseline", choices=["baseline", "kd"])
    parser.add_argument("--qat", action="store_true", help="Enable quantization-aware training")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Enable gradual magnitude pruning of nn.Linear weights during training",
    )
    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=None,
        help="Final sparsity for --prune (overrides pruning.target_sparsity in config)",
    )
    parser.add_argument("--hparams", default=None, help="Path to a hparams JSON (overrides search results)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a run dir, or pass bare --resume to continue the "
        "latest run matching this arch/data-mode/qat (fresh start if none)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device()

    # Loading SentencePiece Tokenizer
    tokenizer = load_tokenizer(cfg)

    prune_cfg = cfg.get("pruning", {})

    # Setting Target Sparsity
    target_sparsity = (
        args.target_sparsity
        if args.target_sparsity is not None
        else prune_cfg.get("target_sparsity", 0.5)
    )

    # Getting Run Name
    run_name = getting_output_folder_name(args, target_sparsity)

    run_dir = find_resume_run_dir(args.resume, run_name)
    # Getting Configurations and Hyperparameters 
    if run_dir is not None:
        meta = load_json(run_dir / "run_config.json")
        for key, value in (
            ("arch", args.arch),
            ("data_mode", args.data_mode),
            ("qat", args.qat),
            ("prune", args.prune),
            ("target_sparsity", target_sparsity),
        ):
            if meta.get(key, False) != value:
                raise SystemExit(
                    f"Cannot resume {run_dir}: its {key}={meta.get(key, False)!r} does not "
                    f"match the requested {value!r}"
                )
        hparams = meta["hparams"]
        print(f"Resuming run: {run_dir}")
    else:
        hparams = load_best_hparams(args.arch, cfg, args.hparams)
        run_dir = create_run_dir("runs", run_name)
    print(f"Run dir: {run_dir} | device: {device}")
    print(f"Hyperparameters: {hparams}")

    train_loader, val_loader = make_loaders(
        cfg, tokenizer, args.data_mode, hparams["batch_size"], device
    )
    model = build_model(
        args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams, qat=args.qat
    )

    checkpoint_meta = {
        "arch": args.arch,
        "hparams": hparams,
        "vocab_size": tokenizer.vocab_size(),
        "pad_id": tokenizer.pad_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
        "data_mode": args.data_mode,
        "qat": args.qat,
        "prune": args.prune,
        "target_sparsity": target_sparsity if args.prune else 0.0,
    }
    save_json(run_dir / "run_config.json", checkpoint_meta)

    # One-shot compression: build the pruner (if requested) so it ramps sparsity
    # in the same run as KD (--data-mode kd) and QAT (--qat). Schedule steps are
    # derived from the planned run length; begin/end fractions come from config.
    pruner = None
    if args.prune:
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * cfg["training"]["epochs"]
        begin_step = int(prune_cfg.get("begin_step_fraction", 0.1) * total_steps)
        end_step = int(prune_cfg.get("end_step_fraction", 0.7) * total_steps)
        end_step = max(end_step, begin_step + 1)
        pruner = MagnitudePruner(
            model,
            target_sparsity=target_sparsity,
            begin_step=begin_step,
            end_step=end_step,
            update_frequency=prune_cfg.get("update_frequency", 100),
        )
        n_layers = pruner.prepare()
        print(
            f"[prune] Gradual magnitude pruning to {target_sparsity:.0%} sparsity "
            f"on {n_layers} Linear layers "
            f"(ramp steps {begin_step}->{end_step} of {total_steps})"
        )

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
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        run_dir=run_dir,
        checkpoint_meta=checkpoint_meta,
        pruner=pruner,
        scheduler_cfg=cfg["training"].get("scheduler"),
    )
    last_state = run_dir / "last.pt"
    if args.resume and last_state.exists():
        resumed_epoch = trainer.load_state(last_state)
        print(f"Resumed training at epoch {resumed_epoch}")
    best_val_loss = trainer.train()

    save_json(run_dir / "history.json", {"best_val_loss": best_val_loss, "epochs": trainer.history})
    print(f"Best val_loss={best_val_loss:.4f} | checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
