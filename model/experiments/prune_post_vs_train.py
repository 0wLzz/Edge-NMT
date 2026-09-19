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

def build_arms(args) -> list[str]:
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


def main() -> None:
    args = argument_parser().parse_args()
    arms = build_arms(args)
    config = load_config(args.config)

    # Setting up Tokenizer and Dataset
    tok_path = Path(args.tokenizer_model).expanduser().resolve() if args.tokenizer_model else None
    tokenizer = load_tokenizer(config, tok_path)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None

    seed = config["seed"]
    device = pick_device()
    hparams = load_hparams(args.arch, config, args.hparams)
    epochs = args.epochs if args.epochs is not None else config["hyperparameter_search"]["trial_epochs"]
    batch_size = hparams["batch_size"]
    search_config = config["hyperparameter_search"]
    prune_config = config.get("pruning", {})
    begin_frac = prune_config.get("begin_step_fraction", 0.1)
    end_frac = prune_config.get("end_step_fraction", 0.7)
    update_frequency = prune_config.get("update_frequency", 100)
    print(f"Architecture={args.arch} Device={device} Epochs={epochs} "
          f"Subset={search_config['subset_size']} Validation={search_config['trial_validation_size']} "
          f"Batch_Size={batch_size} Sparsities={args.sparsities} Hyperparameter={hparams}")

    # identical 200k/10k slice + deterministic shuffle as the search
    train_loader, val_loader = make_subset_loaders(config, tokenizer, batch_size, device, seed, dataset_dir)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id(), label_smoothing=config["training"]["label_smoothing"]
    )
    total_steps = max(1, len(train_loader) * epochs)
    begin_step = int(begin_frac * total_steps)
    end_step = max(begin_step + 1, int(end_frac * total_steps))

    do_gen = args.bleu_samples > 0
    sources = references = None
    if do_gen:
        valid_path = (dataset_dir or resolve_path(config["data"]["processed_dir"])) / "valid.tsv"
        pairs = read_pairs(valid_path)[: args.bleu_samples]
        sources = [s for s, _ in pairs]
        references = [t for _, t in pairs]
        print(f"[gen] BLEU/METEOR on {len(pairs)} val pairs (meteor={args.meteor})")

    def make_trainer(model, pruner=None):
        return Trainer(
            model, train_loader, val_loader,
            pad_id=tokenizer.pad_id(), device=device,
            learning_rate=0.0005,
            weight_decay=config["training"]["weight_decay"],
            label_smoothing=config["training"]["label_smoothing"],
            grad_clip=config["training"]["grad_clip"],
            epochs=epochs,
            early_stopping_patience=config["training"]["early_stopping_patience"],
            quiet=False,
            scheduler_cfg=config["training"].get("scheduler"),
            pruner=pruner,
        )

    results: dict[str, dict] = {}

    def evaluate_variant(name: str, model) -> None:
        row = {"val_loss": round(eval_val_loss(model, val_loader, criterion, device), 4)}
        if do_gen:
            row.update(generation_metrics(model, sources, references, tokenizer, config, device, args.meteor))
        row.update(prune_size_report(model))
        results[name] = row
        print(f"[{name}] {row}")

    # 1) dense fp32 baseline (also the source model for post-training pruning)
    baseline = None
    if args.baseline:
        print("\n===== baseline (dense fp32) =====")
        set_seed(seed)
        baseline = build_model(args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams, qat=False)
        make_trainer(baseline).train()
        evaluate_variant("baseline", baseline)

    # 2) training-time gradual pruning, one fresh run per target sparsity
    if args.prune_train:
        for s in args.sparsities:
            print(f"\n===== prune-train (coremltools, gradual) sparsity={s:.0%} =====")
            set_seed(seed)
            model = build_model(args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams, qat=False)
            model.to(device)
            pruner = CoreMLMagnitudePruner(
                model, target_sparsity=s,
                begin_step=begin_step, end_step=end_step,
                update_frequency=update_frequency, one_shot=False,
            )
            n = pruner.prepare()
            print(f"[prune-train] gradual to {s:.0%} over steps {begin_step}..{end_step} "
                  f"({n} Linear layers)")
            make_trainer(model, pruner=pruner).train()
            pruner.finalize()
            evaluate_variant(f"prune-train@{s:g}", model)

    # 3) post-training one-shot pruning of the trained baseline, per target sparsity
    if args.prune_post:
        for s in args.sparsities:
            print(f"\n===== prune-post (coremltools, one-shot) sparsity={s:.0%} =====")
            model = copy.deepcopy(baseline)
            model.to(device)
            pruner = CoreMLMagnitudePruner(model, target_sparsity=s, one_shot=True)
            pruner.prune_now()
            pruner.finalize()
            evaluate_variant(f"prune-post@{s:g}", model)

    # ---- deltas vs the dense baseline ---------------------------------------
    if args.baseline:
        base = results["baseline"]
        for name, r in results.items():
            if name == "baseline":
                continue
            r["val_loss_delta_vs_dense"] = round(r["val_loss"] - base["val_loss"], 4)
            if do_gen and "bleu" in r and "bleu" in base:
                r["bleu_delta_vs_dense"] = round(r["bleu"] - base["bleu"], 2)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else RESULTS_DIR / "prune_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": args.arch,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "epochs": epochs, "seed": seed, "batch_size": batch_size,
            "subset_size": search_config["subset_size"],
            "val_size": search_config["trial_validation_size"],
            "bleu_samples": args.bleu_samples, "meteor": args.meteor,
            "sparsities": args.sparsities,
            "begin_step": begin_step, "end_step": end_step,
            "update_frequency": update_frequency,
            "hparams": hparams, "device": str(device),
        },
        "results": results,
    }
    out_path = out_dir / f"{args.arch}_prune_post_vs_train.json"
    save_json(out_path, payload)

    # printed comparison table
    cols = ["val_loss"] + (["bleu"] if do_gen else []) + (["meteor"] if (do_gen and args.meteor) else []) \
        + ["sparsity", "sparse_linear_mb", "dense_linear_mb"]
    print("\n================ post vs train pruning ({}) ================".format(args.arch))
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
                + (f"   BLEU {r.get('bleu_delta_vs_dense', float('nan')):+.2f}" if do_gen else ""))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
