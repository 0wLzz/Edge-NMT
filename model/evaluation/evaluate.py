"""Evaluate a trained checkpoint on FLORES-200 and record it in results/summary.csv.

Reports BLEU, chrF++, parameter count, model size, and PyTorch CPU latency.
Writes metrics.json plus the decoded hypotheses next to the checkpoint.

Usage:
    python -m model.evaluation.evaluate --checkpoint results/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from common.config import load_config, pick_device, resolve_path
from common.results import append_summary, save_json
from model.architectures.factory import load_checkpoint
from model.evaluation.efficiency import count_parameters, measure_latency, model_size_mb
from model.evaluation.metrics import compute_all
from model.training.dataset import load_tokenizer, read_pairs


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    checkpoint_path = resolve_path(args.checkpoint)
    model, meta = load_checkpoint(checkpoint_path, device)
    tokenizer = load_tokenizer(cfg)

    eval_cfg = cfg["evaluation"]
    flores_path = (
        resolve_path(cfg["data"]["raw_dir"]) / f"flores_{eval_cfg['flores_split']}.tsv"
    )
    pairs = read_pairs(flores_path)
    if args.limit:
        pairs = pairs[: args.limit]
    sources = [source for source, _ in pairs]
    references = [target for _, target in pairs]

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
            "params_millions": round(count_parameters(model) / 1e6, 2),
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
    save_json(run_dir / "metrics.json", {**meta, **metrics, "flores_split": eval_cfg["flores_split"]})
    with open(run_dir / "hypotheses.txt", "w") as f:
        f.write("\n".join(hypotheses))

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
    print(f"Saved: {run_dir / 'metrics.json'} and appended to results/summary.csv")


if __name__ == "__main__":
    main()
