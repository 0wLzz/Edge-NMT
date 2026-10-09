"""Post-training vs training-time pruning mini-experiment (pruning analogue of
model/experiments/quant_ptq_vs_qat.py).

Compares dense fp32 against magnitude pruning introduced two ways, for ONE
architecture, on the SAME 200k / 10k search-regime slice with the SAME
optimizer, scheduler, seed and epochs as the Optuna search, so the only thing
that differs between the pruned arms is *when* sparsity is introduced:

  baseline     full-precision, dense fp32 reference (the quality bar and the
               source model for post-training pruning)
  prune-train  coremltools training-time gradual magnitude pruning
               (ct.optimize.torch MagnitudePruner, cubic sparsity ramp): the
               network fine-tunes as the zeros come in and recovers accuracy
  prune-post   coremltools one-shot magnitude pruning of the trained baseline
               (same algorithm, ConstantSparsityScheduler at step 0) with NO
               weight updates -- the deploy-time equivalent of
               ct.optimize.coreml.prune_weights

Both pruned arms use the SAME coremltools magnitude algorithm (unstructured,
per-tensor, every nn.Linear); embeddings and recurrent cells stay dense. This
is the evidence needed to decide whether pruning earns its accuracy cost and,
if so, which route degrades less and at what sparsity -- so a sweep over
several target sparsities is run for both arms.

Metrics per arm:
  * validation cross-entropy on the full 10k val slice (same criterion as the
    search, directly comparable to best_<arch>.json),
  * BLEU (+ optional Indonesian-aware METEOR) on a val subsample via greedy
    decoding,
  * measured sparsity and a dense-vs-sparse size estimate.

Usage (locally):
    python -m model.experiments.prune_post_vs_train --arch gru \
        --baseline --prune-train --prune-post --sparsities 0.3 0.5 0.75
Usage (Kaggle, mirroring the search runner):
    python -m model.experiments.prune_post_vs_train --arch gru --config configs/config.yaml \
        --baseline --prune-train --prune-post \
        --dataset-dir /kaggle/input/skripsi-edgenmten-id \
        --tokenizer-model /kaggle/input/.../spm_en_id.model --bleu-samples 1000
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from common.config import load_config, pick_device, resolve_path, set_seed
from common.results import RESULTS_DIR, save_json
from model.architectures.factory import build_model
from model.pruning import sparsity_report
from model.pruning_coreml import CoreMLMagnitudePruner
from model.training.dataset import load_tokenizer, read_pairs
from model.training.hyperparameter_search import default_hparams, make_subset_loaders
from model.training.trainer import Trainer
# Reuse the exact evaluation helpers the QAT experiment uses, so val_loss and
# BLEU are computed identically and the two experiments are comparable.
from model.experiments.quant_ptq_vs_qat import eval_val_loss, generation_metrics


def load_hparams(arch: str, config: dict, override: str | None) -> dict:
    """Best config from the search (or an explicit JSON), else search defaults."""
    if override:
        return json.loads(Path(override).read_text())["hparams"]
    best = RESULTS_DIR / "hparam_search" / f"best_{arch}.json"
    if best.exists():
        print(f"[hparams] using search winner {best.name}")
        return json.loads(best.read_text())["hparams"]
    print(f"[hparams] {best} missing; falling back to search-space defaults")
    return default_hparams(config["hyperparameter_search"]["search_space"][arch])


def prune_size_report(model: nn.Module) -> dict:
    """Dense-vs-sparse size estimate from the zero pattern of nn.Linear weights.

    fp32 dense = all Linear weights stored as float32. The sparse estimate keeps
    only the surviving (non-zero) weights as float32; the index overhead of a
    real sparse format is ignored, so this is an optimistic upper bound on the
    saving, reported for the thesis size table alongside measured sparsity.
    """
    rep = sparsity_report(model)
    total = rep["linear_weight_params"]
    nonzero = total - rep["zeroed_params"]
    return {
        "sparsity": rep["sparsity"],
        "linear_weight_params": total,
        "nonzero_linear_params": nonzero,
        "dense_linear_mb": round(total * 4 / 1e6, 2),
        "sparse_linear_mb": round(nonzero * 4 / 1e6, 2),
    }

def validate_arms(args: argparse.Namespace) -> list[str]:
    """Validate the requested arms. prune-post needs a trained baseline."""
    arms = []
    if args.baseline:
        arms.append("baseline")
    if args.prune_train:
        arms.append("prune-train")
    if args.prune_post:
        if not args.baseline:
            raise ValueError("--prune-post requires --baseline (it prunes the trained baseline)")
        arms.append("prune-post")
    if not arms:
        raise ValueError("At least one of --baseline, --prune-train, or --prune-post must be specified")
    return arms

def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--arch", default="gru", choices=["gru", "lstm", "transformer"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset-dir", default=None, help="dir with train.tsv + valid.tsv (Kaggle input)")
    ap.add_argument("--tokenizer-model", default=None, help="SentencePiece .model (Kaggle input)")
    ap.add_argument("--hparams", default=None, help="hparams JSON override (default: best_<arch>.json)")
    ap.add_argument("--epochs", type=int, default=None, help="default: hyperparameter_search.trial_epochs")
    ap.add_argument("--bleu-samples", type=int, default=1000, help="val pairs to greedy-decode; 0 to skip generation metrics")
    ap.add_argument("--meteor", action="store_true", help="also compute Indonesian-aware METEOR (needs nltk data + Sastrawi)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline", default=False, action="store_true", help="Train + evaluate the dense fp32 baseline arm")
    ap.add_argument("--prune-train", dest="prune_train", default=False, action="store_true", help="Training-time gradual coremltools pruning arm")
    ap.add_argument("--prune-post", dest="prune_post", default=False, action="store_true", help="Post-training one-shot coremltools pruning arm (needs --baseline)")
    ap.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.75], help="target sparsities to sweep for the pruned arms")
    return ap


def _optional_path(path: str | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None

@dataclass
class PruneExperimentArgs:
    """Every resolved setting of one experiment run, from CLI flags + config + hparams.

    Built once by `from_cli`; the rest of the script reads only from here, so a
    setting has a single name no matter where it came from.
    """
    arch: str
    arms: list[str]
    sparsities: list[float]
    hparams: dict
    seed: int
    device: torch.device
    epochs: int
    batch_size: int
    subset_size: int
    val_size: int
    bleu_samples: int
    meteor: bool
    begin_frac: float
    end_frac: float
    update_frequency: int
    dataset_dir: Path | None
    tokenizer_model: Path | None
    out_dir: Path

    @classmethod
    def from_cli(cls, args: argparse.Namespace, config: dict) -> PruneExperimentArgs:
        arms = validate_arms(args)  # fail fast, before loading anything
        search_config = config["hyperparameter_search"]
        prune_config = config.get("pruning", {})
        hparams = load_hparams(args.arch, config, args.hparams)
        return cls(
            arch=args.arch,
            arms=arms,
            sparsities=args.sparsities,
            hparams=hparams,
            seed=config["seed"],
            device=pick_device(),
            epochs=args.epochs if args.epochs is not None else search_config["trial_epochs"],
            batch_size=hparams["batch_size"],
            subset_size=search_config["subset_size"],
            val_size=search_config["trial_validation_size"],
            bleu_samples=args.bleu_samples,
            meteor=args.meteor,
            begin_frac=prune_config.get("begin_step_fraction", 0.1),
            end_frac=prune_config.get("end_step_fraction", 0.7),
            update_frequency=prune_config.get("update_frequency", 100),
            dataset_dir=_optional_path(args.dataset_dir),
            tokenizer_model=_optional_path(args.tokenizer_model),
            out_dir=_optional_path(args.out_dir) or RESULTS_DIR / "prune_experiment",
        )

    @property
    def do_gen(self) -> bool:
        return self.bleu_samples > 0

    def prune_steps(self, steps_per_epoch: int) -> tuple[int, int]:
        """(begin_step, end_step) of the gradual ramp over the planned run length."""
        total_steps = max(1, steps_per_epoch * self.epochs)
        begin_step = int(self.begin_frac * total_steps)
        return begin_step, max(begin_step + 1, int(self.end_frac * total_steps))

    def print_experiment_args(self) -> None:
        print(f"Architecture={self.arch} Device={self.device} Epochs={self.epochs} "
              f"Subset={self.subset_size} Validation={self.val_size} Arms={self.arms} "
              f"Batch_Size={self.batch_size} Sparsities={self.sparsities} Hyperparameter={self.hparams}")

def main() -> None:
    args = argument_parser().parse_args()
    config = load_config(args.config)

    ex_ags = PruneExperimentArgs.from_cli(args, config)
    ex_ags.print_experiment_args()

    tokenizer = load_tokenizer(config, ex_ags.tokenizer_model)

    # identical 200k/10k slice + deterministic shuffle as the search
    train_loader, val_loader = make_subset_loaders(
        config, tokenizer, ex_ags.batch_size, ex_ags.device, ex_ags.seed, ex_ags.dataset_dir
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id(), label_smoothing=config["training"]["label_smoothing"]
    )

    begin_step, end_step = ex_ags.prune_steps(len(train_loader))

    sources = references = None
    if ex_ags.do_gen:
        valid_path = (ex_ags.dataset_dir or resolve_path(config["data"]["processed_dir"])) / "valid.tsv"
        pairs = read_pairs(valid_path)[: ex_ags.bleu_samples]
        sources = [s for s, _ in pairs]
        references = [t for _, t in pairs]
        print(f"[GEN] BLEU/METEOR on {len(pairs)} val pairs (meteor={ex_ags.meteor})")

    def new_model() -> nn.Module:
        set_seed(ex_ags.seed)
        model = build_model(ex_ags.arch, tokenizer.vocab_size(), tokenizer.pad_id(), ex_ags.hparams, qat=False)
        return model.to(ex_ags.device)

    def make_trainer(model, pruner=None):
        return Trainer(
            model, train_loader, val_loader,
            pad_id=tokenizer.pad_id(), device=ex_ags.device,
            learning_rate=0.0005,
            weight_decay=config["training"]["weight_decay"],
            label_smoothing=config["training"]["label_smoothing"],
            grad_clip=config["training"]["grad_clip"],
            epochs=ex_ags.epochs,
            early_stopping_patience=config["training"]["early_stopping_patience"],
            quiet=False,

            # Same schedule as the Optuna search this experiment mirrors.
            scheduler_cfg=config["hyperparameter_search"].get(
                "scheduler", config["training"].get("scheduler")
            ),
            pruner=pruner,
        )

    def evaluate_variant(name: str, model) -> None:
        row = {"val_loss": round(eval_val_loss(model, val_loader, criterion, ex_ags.device), 4)}
        
        if ex_ags.do_gen:
            row.update(generation_metrics(model, sources, references, tokenizer, config, ex_ags.device, ex_ags.meteor))
        
        row.update(prune_size_report(model))
        results[name] = row
        print(f"[{name}] {row}")

    results: dict[str, dict] = {}

    # 1) dense fp32 baseline (also the source model for post-training pruning)
    baseline = None
    if "baseline" in ex_ags.arms:
        print("\n===== Baseline (Dense FP32) =====")
        baseline = new_model()
        baseline_trainer = make_trainer(baseline)
        baseline_trainer.train()

        # Evaluate the baseline model
        evaluate_variant("baseline", baseline)

        # Save the trained baseline
        baseline_path = ex_ags.out_dir / f"{ex_ags.arch}_baseline.pt"
        baseline_trainer.save_state(baseline_path)
        print(f"[Baseline] Saved trained baseline to {baseline_path}")

    # 2) training-time gradual pruning, one fresh run per target sparsity
    if "prune-train" in ex_ags.arms:
        for sparsity in ex_ags.sparsities:
            print(f"\n===== Prune-Train (coremltools, gradual) sparsity={sparsity:.0%} =====")
            model = new_model()

            # Setting Gradiual Pruner
            pruner = CoreMLMagnitudePruner(
                model, target_sparsity=sparsity,
                begin_step=begin_step, end_step=end_step,
                update_frequency=ex_ags.update_frequency, one_shot=False,
            )

            n = pruner.prepare()
            print(f"[Prune-Train] Gradual pruning to {sparsity:.0%} over steps {begin_step}..{end_step} "
                  f"({n} Linear layers)")
            
            qat_trainer = make_trainer(model, pruner=pruner)
            qat_trainer.train()
            pruner.finalize()
            evaluate_variant(f"prune-train@{sparsity:g}", model)

            # Save the trained pruned model
            save_path = ex_ags.out_dir / f"{ex_ags.arch}_prune_train_{sparsity:g}.pt"
            qat_trainer.save_state(save_path)
            print(f"[Prune-Train] Saved trained pruned model to {save_path}")

    # 3) post-training one-shot pruning of the trained baseline, per target sparsity
    if "prune-post" in ex_ags.arms:
        for sparsity in ex_ags.sparsities:
            print(f"\n===== Prune-Post (coremltools, one-shot) sparsity={sparsity:.0%} =====")
            model = copy.deepcopy(baseline)
            model.to(ex_ags.device)

            pruner = CoreMLMagnitudePruner(model, target_sparsity=sparsity, one_shot=True)
            pruner.prune_now()
            pruner.finalize()
            evaluate_variant(f"prune-post@{sparsity:g}", model)

            # Save the trained pruned model
            save_path = ex_ags.out_dir / f"{ex_ags.arch}_prune_post_{sparsity:g}.pt"
            post_trainer = make_trainer(model)
            post_trainer.save_state(save_path)
            print(f"[Prune-Post] Saved trained pruned model to {save_path}")

    # ---- deltas vs the dense baseline ---------------------------------------
    if "baseline" in ex_ags.arms:
        base = results["baseline"]
        for name, r in results.items():
            if name == "baseline":
                continue
            r["val_loss_delta_vs_dense"] = round(r["val_loss"] - base["val_loss"], 4)
            if ex_ags.do_gen and "bleu" in r and "bleu" in base:
                r["bleu_delta_vs_dense"] = round(r["bleu"] - base["bleu"], 2)

    ex_ags.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": ex_ags.arch,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "epochs": ex_ags.epochs, "seed": ex_ags.seed, "batch_size": ex_ags.batch_size,
            "subset_size": ex_ags.subset_size,
            "val_size": ex_ags.val_size,
            "bleu_samples": ex_ags.bleu_samples, "meteor": ex_ags.meteor,
            "sparsities": ex_ags.sparsities,
            "begin_step": begin_step, "end_step": end_step,
            "update_frequency": ex_ags.update_frequency,
            "hparams": ex_ags.hparams, "device": str(ex_ags.device),
        },
        "results": results,
    }
    out_path = ex_ags.out_dir / f"{ex_ags.arch}_prune_post_vs_train.json"
    save_json(out_path, payload)

    # printed comparison table
    cols = ["val_loss"] + (["bleu"] if ex_ags.do_gen else []) + (["meteor"] if (ex_ags.do_gen and ex_ags.meteor) else []) \
        + ["sparsity", "sparse_linear_mb", "dense_linear_mb"]
    print("\n================ post vs train pruning ({}) ================".format(ex_ags.arch))
    header = "variant".ljust(18) + "".join(c.rjust(14) for c in cols)
    print(header); print("-" * len(header))
    for name in results:
        r = results[name]
        line = name.ljust(18) + "".join(str(r.get(c, "")).rjust(14) for c in cols)
        print(line)

    print("\nDeltas vs dense fp32 (positive val_loss / negative BLEU = worse):")
    for name, r in results.items():
        if name == "baseline":
            continue
        print(f"  {name}: val_loss {r.get('val_loss_delta_vs_dense', float('nan')):+.4f}"
                + (f"   BLEU {r.get('bleu_delta_vs_dense', float('nan')):+.2f}" if ex_ags.do_gen else ""))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
