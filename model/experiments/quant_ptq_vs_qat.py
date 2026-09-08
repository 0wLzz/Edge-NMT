"""QAT vs PTQ accuracy mini-experiment (executes D13 step 2).

Compares three variants of ONE architecture, trained and evaluated on the SAME
200k / 10k search-regime slice, with the SAME optimizer, scheduler, seed and
epochs as the Optuna hyperparameter search, so the *only* thing that differs
between the arms is how int8 (weights + Linear activations) is introduced:

  baseline  full-precision fp32 reference (the quality bar)
  qat       custom quantization-aware training (model/qat.py) trained from
            scratch with fake-quant active
  ptq       the SAME fake-quant applied post-hoc to the trained baseline, with
            a short calibration pass and NO weight updates

All arms are evaluated with the fake-quant *active* (weights + Linear-input
activations quantized at inference — W8A8), so the numbers show the real
accuracy cost of int8, and QAT vs PTQ differ only in *when* quantization is
introduced (during training vs after). This is the evidence D13 needs to decide
whether quantization earns its accuracy cost and, if so, which route degrades
less — before committing to the D12 deployment stack.

Metrics per arm:
  * validation cross-entropy on the full 10k val slice (same criterion as the
    search, so it is directly comparable to best_<arch>.json),
  * BLEU (+ optional Indonesian-aware METEOR) on a val subsample via greedy
    decoding,
  * fp32-vs-int8 size estimate (weights of the quantized Linear layers).

Usage (locally):
    python -m model.experiments.quant_ptq_vs_qat --arch gru
Usage (Kaggle, mirroring the search runner):
    python -m model.experiments.quant_ptq_vs_qat --arch gru --config configs/config.yaml \
        --dataset-dir /kaggle/input/skripsi-edgenmten-id \
        --tokenizer-model /kaggle/input/.../spm_en_id.model \
        --bleu-samples 1000
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
from model.qat import FakeQuantize, apply_qat, int8_size_report, set_qat_enabled
from model.training.dataset import load_tokenizer, read_pairs
from model.training.hyperparameter_search import default_hparams, make_subset_loaders
from model.training.trainer import Trainer
from model.evaluation.evaluate import encode_sources, translate_corpus
from model.evaluation.metrics import compute_all


def load_hparams(arch: str, cfg: dict, override: str | None) -> dict:
    """Best config from the search (or an explicit JSON), else search defaults."""
    if override:
        return json.loads(Path(override).read_text())["hparams"]
    best = RESULTS_DIR / "hparam_search" / f"best_{arch}.json"
    if best.exists():
        print(f"[hparams] using search winner {best.name}")
        return json.loads(best.read_text())["hparams"]
    print(f"[hparams] {best} missing; falling back to search-space defaults")
    return default_hparams(cfg["hyperparameter_search"]["search_space"][arch])


@torch.no_grad()
def eval_val_loss(model, loader, criterion, device) -> float:
    """Mean validation cross-entropy — identical computation to Trainer.validate."""
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        source, target_input, target_labels = (t.to(device) for t in batch)
        logits = model(source, target_input)
        total += criterion(logits.reshape(-1, logits.shape[-1]), target_labels.reshape(-1)).item()
        batches += 1
    return total / max(batches, 1)


def calibrate_ptq(model, loader, n_batches: int, device) -> None:
    """PTQ calibration: let the fake-quant observers see representative data and
    collect ranges, WITHOUT any weight updates. Dropout stays off (model.eval());
    only the FakeQuantize modules are put in observe mode, then frozen."""
    model.eval()
    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.train()  # FakeQuantize.forward observes only while training=True

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            source, target_input, _ = (t.to(device) for t in batch)
            model(source, target_input)

    for m in model.modules():
        if isinstance(m, FakeQuantize):
            m.eval()  # freeze the observed ranges for inference


def generation_metrics(model, sources, references, tokenizer, cfg, device, meteor: bool) -> dict:
    """BLEU (+ optional METEOR) on the val subsample via greedy decoding."""
    meta = {
        "pad_id": tokenizer.pad_id(),
        "bos_id": tokenizer.bos_id(),
        "eos_id": tokenizer.eos_id(),
    }
    max_len = cfg["evaluation"]["decode_max_length"]
    batch_size = cfg["evaluation"]["decode_batch_size"]
    encoded = encode_sources(sources, tokenizer, max_len)
    hyps = translate_corpus(model, encoded, tokenizer, meta, device, batch_size, max_len)
    try:
        return compute_all(hyps, references, chrf=False, meteor=meteor)
    except Exception as exc:  # missing nltk data / Sastrawi -> keep BLEU, skip METEOR
        print(f"[metrics] METEOR unavailable ({exc}); reporting BLEU only")
        return compute_all(hyps, references, chrf=False, meteor=False)


def main() -> None:
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
    ap.add_argument("--calib-batches", type=int, default=50, help="PTQ calibration batches")
    ap.add_argument("--meteor", action="store_true", help="also compute Indonesian-aware METEOR (needs nltk data + Sastrawi)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--qat", default=False, type=bool, help="Enable quantization-aware training (QAT) for the qat arm")
    ap.add_argument("--ptq", default=False, type=bool, help="Enable post-training quantization (PTQ) for the ptq arm")
    ap.add_argument("--baseline", defaul=False, type=bool, help="Enable baseline (fp32) for the baseline arm")

    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    device = pick_device()
    tok_path = Path(args.tokenizer_model).expanduser().resolve() if args.tokenizer_model else None
    tokenizer = load_tokenizer(cfg, tok_path)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None

    hparams = load_hparams(args.arch, cfg, args.hparams)
    epochs = args.epochs if args.epochs is not None else cfg["hyperparameter_search"]["trial_epochs"]
    batch_size = hparams["batch_size"]
    search_cfg = cfg["hyperparameter_search"]
    print(f"arch={args.arch} device={device} epochs={epochs} "
          f"subset={search_cfg['subset_size']} val={search_cfg['trial_validation_size']} "
          f"batch_size={batch_size} hparams={hparams}")

    # identical 200k/10k slice + deterministic shuffle as the search
    train_loader, val_loader = make_subset_loaders(cfg, tokenizer, batch_size, device, seed, dataset_dir)
    # validation loss criterion identical to the search (label smoothing + pad ignore)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id(), label_smoothing=cfg["training"]["label_smoothing"]
    )

    do_gen = args.bleu_samples > 0
    sources = references = None
    if do_gen:
        valid_path = (dataset_dir or resolve_path(cfg["data"]["processed_dir"])) / "valid.tsv"
        pairs = read_pairs(valid_path)[: args.bleu_samples]
        sources = [s for s, _ in pairs]
        references = [t for _, t in pairs]
        print(f"[gen] BLEU/METEOR on {len(pairs)} val pairs (meteor={args.meteor})")

    def make_trainer(model):
        return Trainer(
            model, train_loader, val_loader,
            pad_id=tokenizer.pad_id(), device=device,
            learning_rate=0.0005,
            weight_decay=cfg["training"]["weight_decay"],
            label_smoothing=cfg["training"]["label_smoothing"],
            grad_clip=cfg["training"]["grad_clip"],
            epochs=epochs,
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
            quiet=False,
            scheduler_cfg=cfg["training"].get("scheduler"),
        )

    results: dict[str, dict] = {}

    def evaluate_variant(name: str, model) -> None:
        set_qat_enabled(model, True)  # keep fake-quant ON at inference (no-op for fp32)
        row = {"val_loss": round(eval_val_loss(model, val_loader, criterion, device), 4)}
        if do_gen:
            row.update(generation_metrics(model, sources, references, tokenizer, cfg, device, args.meteor))
        row.update(int8_size_report(model))
        results[name] = row
        print(f"[{name}] {row}")

    # 1) fp32 baseline (also the source model for PTQ)
    if args.baseline:
        print("\n===== baseline (fp32) =====")
        set_seed(seed)
        baseline = build_model(args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams, qat=False)
        make_trainer(baseline).train()
        evaluate_variant("baseline", baseline)

        # 3) PTQ: same fake-quant applied to the *trained* baseline + calibration only
        if args.ptq:
            print("\n===== PTQ (post-training, calibrated) =====")
            ptq_model = copy.deepcopy(baseline)
            apply_qat(ptq_model)          # wrap Linears with fresh observers
            ptq_model.to(device)          # move the new FakeQuantize buffers onto the device
            calibrate_ptq(ptq_model, train_loader, args.calib_batches, device)
            evaluate_variant("ptq", ptq_model)

    if args.qat: 
        # 2) QAT: fresh model, fake-quant active during training (same seed init)
        print("\n===== QAT (custom, train-time) =====")
        set_seed(seed)
        qat_model = build_model(args.arch, tokenizer.vocab_size(), tokenizer.pad_id(), hparams, qat=True)
        make_trainer(qat_model).train()
        evaluate_variant("qat", qat_model)


    # ---- summary ------------------------------------------------------------
    base = results["baseline"]
    for name in ("qat", "ptq"):
        r = results[name]
        r["val_loss_delta_vs_fp32"] = round(r["val_loss"] - base["val_loss"], 4)
        if do_gen and "bleu" in r and "bleu" in base:
            r["bleu_delta_vs_fp32"] = round(r["bleu"] - base["bleu"], 2)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else RESULTS_DIR / "quant_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": args.arch,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "epochs": epochs, "seed": seed, "batch_size": batch_size,
            "subset_size": search_cfg["subset_size"],
            "val_size": search_cfg["trial_validation_size"],
            "bleu_samples": args.bleu_samples, "calib_batches": args.calib_batches,
            "meteor": args.meteor, "hparams": hparams, "device": str(device),
        },
        "results": results,
    }
    out_path = out_dir / f"{args.arch}_qat_vs_ptq.json"
    save_json(out_path, payload)

    # printed comparison table
    cols = ["val_loss"] + (["bleu"] if do_gen else []) + (["meteor"] if (do_gen and args.meteor) else []) + ["int8_size_mb", "fp32_size_mb"]
    print("\n================ QAT vs PTQ ({}) ================".format(args.arch))
    header = "variant".ljust(10) + "".join(c.rjust(14) for c in cols)
    print(header); print("-" * len(header))
    for name in ("baseline", "qat", "ptq"):
        r = results[name]
        line = name.ljust(10) + "".join(str(r.get(c, "")).rjust(14) for c in cols)
        print(line)
    print("\nDeltas vs fp32 (negative BLEU / positive val_loss = worse):")
    for name in ("qat", "ptq"):
        r = results[name]
        print(f"  {name}: val_loss {r['val_loss_delta_vs_fp32']:+.4f}"
              + (f"   BLEU {r.get('bleu_delta_vs_fp32', float('nan')):+.2f}" if do_gen else ""))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
