"""Sequence-level knowledge distillation data generation.

The teacher (mBART-50) translates every source sentence of the cleaned
training split once. Students then train on (source, teacher_translation)
pairs with plain cross-entropy — this is sequence-level KD (Kim & Rush, 2016)
and avoids the vocabulary mismatch between mBART's tokenizer and the students'
SentencePiece vocabulary.

Resumable: progress is recorded in kd_train.progress.json after every batch,
so an interrupted run (Colab timeout, crash) restarts from the last completed
batch instead of the beginning. Just rerun the same command.

Usage:
    python -m model.training.generate_kd_dataset [--config ...] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import math

import torch
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from common.config import load_config, pick_device, resolve_path
from common.results import load_json, save_json
from model.training.dataset import read_pairs


def load_teacher(cfg: dict, device: torch.device):
    teacher_cfg = cfg["teacher"]
    tokenizer = MBart50TokenizerFast.from_pretrained(
        teacher_cfg["model_name"], src_lang=teacher_cfg["source_lang_code"]
    )
    model = MBartForConditionalGeneration.from_pretrained(teacher_cfg["model_name"])
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def translate_batch(sources, tokenizer, model, teacher_cfg, device) -> list[str]:
    inputs = tokenizer(
        sources, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[teacher_cfg["target_lang_code"]],
        num_beams=teacher_cfg["num_beams"],
        max_new_tokens=teacher_cfg["max_new_tokens"],
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Translate only the first N pairs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    output_path = processed_dir / "kd_train.tsv"
    progress_path = processed_dir / "kd_train.progress.json"

    pairs = read_pairs(processed_dir / "train.tsv")
    if args.limit:
        pairs = pairs[: args.limit]
    sources = [source for source, _ in pairs]

    # Resume from the last completed batch if a previous run was interrupted.
    done = 0
    if progress_path.exists() and output_path.exists():
        done = min(load_json(progress_path)["processed_sources"], len(sources))
        print(f"Resuming: {done}/{len(sources)} sources already translated")
    if done >= len(sources):
        print(f"Already complete: {output_path}")
        return

    tokenizer, model = load_teacher(cfg, device)
    teacher_cfg = cfg["teacher"]
    batch_size = teacher_cfg["batch_size"]

    with open(output_path, "a" if done else "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        progress = tqdm(
            range(done, len(sources), batch_size),
            initial=done // batch_size,
            total=math.ceil(len(sources) / batch_size),
            desc="Teacher translating",
        )
        for start in progress:
            batch = sources[start : start + batch_size]
            translations = translate_batch(batch, tokenizer, model, teacher_cfg, device)
            for source, translation in zip(batch, translations):
                translation = translation.replace("\t", " ").replace("\n", " ").strip()
                if translation:
                    writer.writerow([source, translation])
            f.flush()
            save_json(
                progress_path,
                {"processed_sources": start + len(batch), "total_sources": len(sources)},
            )

    print(f"KD training data -> {output_path}")


if __name__ == "__main__":
    main()
