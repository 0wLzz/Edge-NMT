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

On time-boxed GPUs (Kaggle 12h), `--time-budget 11.5` stops the session after
the last epoch that fits, and `--session-epochs N` after N epochs, instead of
being killed mid-epoch and losing it. Both only end the session (no final.pt);
rerun with --resume to continue.

Usage:
    python -m model.training.train --arch transformer --data-mode kd --qat
    python -m model.training.train --arch transformer --data-mode kd --qat --prune
    python -m model.training.train --arch transformer --data-mode kd --qat --prune \
        --target-sparsity 0.75 --resume
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

from common.config import (
    DEFAULT_KD_FILE, getting_output_folder_name, load_config, pick_device, resolve_path, set_seed,
)
from common.results import RESULTS_DIR, create_run_dir, load_json, save_json
from model.architectures.factory import build_model
from model.pruning import MagnitudePruner
from model.training.dataset import TranslationDataset, load_tokenizer, make_collate_fn
from model.training.hyperparameter_search import default_hparams
from model.training.trainer import Trainer


def find_resume_run_dir(resume: str | None, run_name: str):
    """Resolve the --resume argument to an existing run dir (or None).

    "latest" picks the newest results/runs/<timestamp>_<run_name> that has a resume state;
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


def make_loaders(cfg: dict, tokenizer, data_mode: str, batch_size: int, device, dataset_dir=None,
                 kd_file: str = DEFAULT_KD_FILE):
    processed_dir = Path(dataset_dir) if dataset_dir else resolve_path(cfg["data"]["processed_dir"])
    train_file = kd_file if data_mode == "kd" else "train.tsv"
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

def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=["gru", "lstm", "transformer"])
    parser.add_argument("--data-mode", default="baseline", choices=["baseline", "kd"])
    parser.add_argument("--kd-file", default=DEFAULT_KD_FILE,
                        help="KD training file in the dataset dir for --data-mode kd (default: the "
                        "mBART-50 kd_train.tsv; e.g. kd_train_nllb-1.3B.tsv). A non-default file "
                        "adds its teacher to the run name, e.g. gru_kd-nllb-1.3B")
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
    parser.add_argument("--dataset-dir", default=None,
                        help="dir with train.tsv/kd_train.tsv/valid.tsv (e.g. a Kaggle input); defaults to data.processed_dir")
    parser.add_argument("--tokenizer-model", default=None,
                        help="SentencePiece .model path; defaults to tokenizer.model_prefix")
    parser.add_argument("--config", default=None)

    parser.add_argument("--last-model", default=None,
                        help="Path to a last.pt checkpoint to resume from (overrides --resume)")
    parser.add_argument("--session-epochs", type=int, default=None,
                        help="Stop this session cleanly after N epochs (last.pt saved); continue "
                        "with --resume. Does not change training.epochs or the LR/pruning schedule")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="Session time budget in hours (e.g. 11.5 on Kaggle's 12h limit): stop "
                        "after the last epoch that fits instead of being killed mid-epoch")
    return parser

def prep_prune(model, prune_cfg : dict, cfg : dict, train_loader: DataLoader, target_sparsity: float) -> MagnitudePruner:
    """Build the CoreMLMagnitudePruner and prepare it for gradual pruning."""

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
        f"[Prune] Gradual magnitude pruning to {target_sparsity:.0%} sparsity "
        f"on {n_layers} Linear layers "
        f"(ramp steps {begin_step}->{end_step} of {total_steps})"
    )

    return pruner

def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    # Measured from process start so data loading/tokenization counts too.
    deadline = time.monotonic() + args.time_budget * 3600 if args.time_budget else None

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = pick_device()

    # Loading SentencePiece Tokenizer
    tokenizer_path = Path(args.tokenizer_model).expanduser().resolve() if args.tokenizer_model else None
    tokenizer = load_tokenizer(cfg, tokenizer_path)

    prune_cfg = cfg.get("pruning", {})

    # Setting Target Sparsity
    target_sparsity = (
        args.target_sparsity
        if args.target_sparsity is not None
        else prune_cfg.get("target_sparsity", 0.75)
    )

    # Getting Run Name
    run_name = getting_output_folder_name(args, target_sparsity)

    kd_file = args.kd_file if args.data_mode == "kd" else None

    run_dir = find_resume_run_dir(args.resume, run_name)
    # Getting Configurations and Hyperparameters
    if run_dir is not None:
        meta = load_json(run_dir / "run_config.json")
        # Runs from before --kd-file have no kd_file key; their KD data was kd_train.tsv.
        meta.setdefault("kd_file", DEFAULT_KD_FILE if meta.get("data_mode") == "kd" else None)
        for key, value in (
            ("arch", args.arch),
            ("data_mode", args.data_mode),
            ("kd_file", kd_file),
            ("qat", args.qat),
            ("prune", args.prune),
            # Saved as 0.0 for runs without --prune (see checkpoint_meta below).
            ("target_sparsity", target_sparsity if args.prune else 0.0),
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
    print(f"Scheduler config: {cfg['training'].get('scheduler')}")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    train_loader, val_loader = make_loaders(
        cfg, tokenizer, args.data_mode, hparams["batch_size"], device, dataset_dir, args.kd_file
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
        "kd_file": kd_file,
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
        pruner = prep_prune(model, prune_cfg, cfg, train_loader, target_sparsity)

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
        session_epochs=args.session_epochs,
        deadline=deadline,
    )

    last_state = Path(args.last_model).expanduser().resolve() if args.last_model else run_dir / "last.pt"
    if args.resume and last_state.exists():
        resumed_epoch = trainer.load_state(last_state)
        print(f"Resumed training at epoch {resumed_epoch}")

    best_val_loss = trainer.train()

    save_json(run_dir / "history.json", {"best_val_loss": best_val_loss, "epochs": trainer.history})
    if trainer.session_interrupted:
        print(f"Session ended early. Best val_loss so far={best_val_loss:.4f} | "
              f"resume state: {run_dir / 'last.pt'}")
    else:
        print(f"Best val_loss={best_val_loss:.4f} | best: {run_dir / 'best.pt'} | final: {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
