"""Evaluate a trained checkpoint on FLORES-200 and record it in results/summary.csv.

Reports BLEU, chrF++/METEOR (per config), parameter count, model size, and
PyTorch CPU latency. Writes next to the checkpoint:
  metrics.json      - all scores + checkpoint meta
  translations.csv  - id, source, reference, hypothesis, sentence_bleu per line
  hypotheses.txt    - raw hypotheses, one per line

The checkpoint does not record which SentencePiece model it was trained with,
so pass --tokenizer when it differs from config `tokenizer.model_prefix`. A
mismatched tokenizer silently produces garbage (every token id maps to the
wrong subword), so check translations.csv whenever scores look implausible.

Usage:
    python -m model.evaluation.evaluate --checkpoint results/runs/<run>/best.pt \
        [--tokenizer data/processed/2M/spm_en_id.model] [--notes "Some notes about this evaluation"]
"""

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from common.config import load_config, pick_device, resolve_path
from common.results import append_summary, save_json
from model.architectures.factory import load_checkpoint
from model.evaluation.efficiency import count_parameters, measure_latency, model_size_mb
from model.evaluation.metrics import compute_all, sentence_bleu_scores
from model.training.dataset import load_tokenizer

# FLORES CSV column names for each side. Columns are looked up by name, never by
# position: a swapped order would feed Indonesian into an en->id model and score
# it against English (BLEU ~0) without raising any error.
SOURCE_COLUMNS = ("source", "eng")
TARGET_COLUMNS = ("target", "ind")


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...], path: Path) -> str:
    for name in candidates:
        if name in fieldnames:
            return name
    raise SystemExit(
        f"{path} has columns {fieldnames}; expected one of {list(candidates)}"
    )


def read_flores(path: Path) -> list[dict]:
    """Read the FLORES CSV (header row required) into [{id, source, reference}]."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        source_col = _pick_column(fieldnames, SOURCE_COLUMNS, path)
        target_col = _pick_column(fieldnames, TARGET_COLUMNS, path)
        return [
            {
                "id": row.get("id", str(i)),
                "source": row[source_col],
                "reference": row[target_col],
            }
            for i, row in enumerate(reader)
        ]


def export_translations(path: Path, examples: list[dict], hypotheses: list[str]) -> None:
    """Write source/reference/hypothesis side by side for manual inspection."""
    references = [ex["reference"] for ex in examples]
    scores = sentence_bleu_scores(hypotheses, references)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "reference", "hypothesis", "sentence_bleu"])
        for ex, hyp, score in zip(examples, hypotheses, scores):
            writer.writerow([ex["id"], ex["source"], ex["reference"], hyp, score])


def encode_sources(sources: list[str], tokenizer, max_length: int) -> list[torch.Tensor]:
    eos = tokenizer.eos_id()
    return [
        torch.tensor(tokenizer.encode(text)[: max_length - 1] + [eos]) for text in sources
    ]


@torch.no_grad()
def translate_corpus(
    model, encoded_sources, tokenizer, meta, device, batch_size: int, max_length: int
) -> list[str]:
    pad_id = meta["pad_id"]
    hypotheses = []
    for start in tqdm(range(0, len(encoded_sources), batch_size), desc="Decoding"):
        batch = encoded_sources[start : start + batch_size]
        source = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=pad_id
        ).to(device) 
        token_lists = model.greedy_decode(source, meta["bos_id"], meta["eos_id"], max_length)
        hypotheses.extend(tokenizer.decode(ids) for ids in token_lists)
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sentences")
    parser.add_argument("--notes", default="")
    parser.add_argument("--quantized", action="store_true", help="Mark this row as a folded/quantized model")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="SentencePiece .model the checkpoint was trained with "
        "(default: config tokenizer.model_prefix)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    checkpoint_path = resolve_path(args.checkpoint)
    model, meta = load_checkpoint(checkpoint_path, device)
    tokenizer_path = resolve_path(args.tokenizer) if args.tokenizer else None
    tokenizer = load_tokenizer(cfg, tokenizer_path)
    if tokenizer.vocab_size() != meta["vocab_size"]:
        raise SystemExit(
            f"Tokenizer vocab {tokenizer.vocab_size()} != checkpoint vocab "
            f"{meta['vocab_size']}; pass the tokenizer the model was trained with"
        )

    eval_cfg = cfg["evaluation"]
    flores_path = (
        resolve_path(cfg["data"]["raw_dir"]) / f"flores_{eval_cfg['flores_split']}.tsv"
    )

    examples = read_flores(flores_path)
    if args.limit:
        examples = examples[: args.limit]
    sources = [ex["source"] for ex in examples]
    references = [ex["reference"] for ex in examples]

    max_length = eval_cfg["decode_max_length"]
    encoded = encode_sources(sources, tokenizer, max_length)
    hypotheses = translate_corpus(
        model, encoded, tokenizer, meta, device, eval_cfg["decode_batch_size"], max_length
    )

    metrics = compute_all(
        hypotheses,
        references,
        chrf=eval_cfg.get("compute_chrf", True),
        meteor=eval_cfg.get("compute_meteor", False),
    )

    metrics.update(
        {
            "params_millions": round(count_parameters(model) / 1e6, 4),
            "model_size_mb": model_size_mb(model),
        }
    )
    
    metrics.update(
        measure_latency(
            model,
            encoded[:10],
            meta["bos_id"],
            meta["eos_id"],
            max_length,
            warmup=eval_cfg["latency_warmup"],
            repeats=eval_cfg["latency_repeats"],
        )
    )

    run_dir = checkpoint_path.parent
    save_json(
        run_dir / "metrics.json",
        {
            **meta,
            **metrics,
            "flores_split": eval_cfg["flores_split"],
            "num_sentences": len(examples),
            "tokenizer": str(tokenizer_path or "config default"),
        },
    )

    export_translations(run_dir / "translations.csv", examples, hypotheses)

    append_summary(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": run_dir.name,
            "arch": meta["arch"],
            "data_mode": meta.get("data_mode", ""),
            "qat": meta.get("qat", False),
            "quantized": args.quantized,
            "checkpoint": str(checkpoint_path),
            "bleu": metrics["bleu"],
            "chrf": metrics.get("chrf", ""),
            "meteor": metrics.get("meteor", ""),
            "params_millions": metrics["params_millions"],
            "model_size_mb": metrics["model_size_mb"],
            "latency_ms_per_sentence": metrics["latency_ms_per_sentence"],
            "notes": args.notes,
        }
    )

    score_line = f"BLEU={metrics['bleu']}"
    if "chrf" in metrics:
        score_line += f" chrF++={metrics['chrf']}"
    if "meteor" in metrics:
        score_line += f" METEOR={metrics['meteor']}"
    print(score_line)
    print(f"Params={metrics['params_millions']}M size={metrics['model_size_mb']}MB "
          f"latency={metrics['latency_ms_per_sentence']}ms/sentence")
    print(f"Saved: {run_dir / 'metrics.json'}, {run_dir / 'translations.csv'}; "
          f"appended to results/summary.csv")


if __name__ == "__main__":
    main()
