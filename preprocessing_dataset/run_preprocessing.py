"""Clean the raw CCMatrix TSV and produce train/valid splits.

Pipeline: unicode normalization -> length, ratio & long-word filters ->
deduplication -> language-ID filtering (if lid.176.bin is present) ->
reservoir-sampled validation split, streaming the rest to train.tsv.

Memory-safe: kept pairs are written straight to disk instead of held in a
list, and the validation set is drawn with reservoir sampling (O(valid_size)
memory), so the whole 70M-pair corpus can be processed without loading it into
RAM. The only structure that grows with corpus size is the dedup hash set
(a few GB at 70M). train.tsv keeps the input's margin-descending order, so a
training-time --limit still selects the highest-quality pairs; shuffle in the
DataLoader during training.

Usage:
    python -m preprocessing_dataset.run_preprocessing [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from tqdm import tqdm

from common.config import load_config, resolve_path, set_seed
from common.results import RESULTS_DIR, save_json
from preprocessing_dataset.cleaning import LanguageFilter, PairCleaner


def read_pairs_tsv(path: Path, limit: int | None = None):
    """Yield (source, target) pairs from a tab-separated corpus.

    Handles both the 2-column format written by download_data (source, target)
    and the raw CCMatrix 3-column format (margin_score, source, target); the
    leading margin score is ignored. A plain split is used rather than csv so
    that backslashes in the text don't confuse the parser.
    """
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = line.rstrip("\n").split("\t")
            if len(row) == 3:
                source, target = row[1], row[2]
            elif len(row) == 2:
                
                source, target = row[0], row[1]
            else:
                continue
            yield source, target
            count += 1
            if limit is not None and count >= limit:
                break


def write_pairs_tsv(path: Path, pairs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max input pairs to read. CCMatrix is sorted by margin score, so "
        "the first N are the highest quality. Omit to process the whole file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    raw_path = resolve_path(cfg["data"]["raw_dir"]) / "ccmatrix.tsv"
    processed_dir = resolve_path(cfg["data"]["processed_dir"])

    lang_filter = LanguageFilter(
        resolve_path(cfg["preprocessing"]["fasttext_lid_model"]),
        cfg["preprocessing"]["language_confidence_threshold"],
    )
    cleaner = PairCleaner(cfg, lang_filter)

    validation_size = cfg["data"]["validation_size"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.tsv"
    valid_path = processed_dir / "valid.tsv"

    # Reservoir sampling: `reservoir` holds a uniform random validation_size
    # sample of the kept pairs; everything else is streamed straight to train.
    reservoir: list = []
    kept = 0
    with open(train_path, "w", newline="") as train_f:
        writer = csv.writer(
            train_f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        for source, target in tqdm(read_pairs_tsv(raw_path, args.limit), desc="Cleaning"):
            pair = cleaner.clean(source, target)
            if pair is None:
                continue
            if kept < validation_size:
                reservoir.append(pair)
            else:
                j = random.randint(0, kept)
                if j < validation_size:
                    writer.writerow(reservoir[j])  # evict to train
                    reservoir[j] = pair
                else:
                    writer.writerow(pair)
            kept += 1

    write_pairs_tsv(valid_path, reservoir)
    train_size = kept - len(reservoir)

    stats = cleaner.stats.as_dict()
    stats["train_size"] = train_size
    stats["valid_size"] = len(reservoir)
    stats_path = RESULTS_DIR / "preprocessing" / "cleaning_stats.json"
    save_json(stats_path, stats)

    samples_path = RESULTS_DIR / "preprocessing" / "dropped_samples.json"
    save_json(samples_path, cleaner.drop_samples)

    print(f"Train: {train_size} pairs -> {train_path}")
    print(f"Valid: {len(reservoir)} pairs -> {valid_path}")
    print(f"Stats: {stats_path}")
    print(f"Dropped samples: {samples_path}")


if __name__ == "__main__":
    main()
