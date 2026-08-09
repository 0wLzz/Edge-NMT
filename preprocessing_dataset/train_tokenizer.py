"""Train the shared English-Indonesian SentencePiece BPE tokenizer.

Trained on both sides of the cleaned training split so one vocabulary serves
encoder and decoder (required for weight sharing and simpler CoreML export).

Usage:
    python -m preprocessing_dataset.train_tokenizer [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import sentencepiece as spm

from common.config import load_config, resolve_path


def dump_sentences(train_tsv: Path, out_file) -> int:
    """Write every source and target sentence on its own line."""
    count = 0
    with open(train_tsv, newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            if len(row) != 2:
                continue
            out_file.write(row[0] + "\n")
            out_file.write(row[1] + "\n")
            count += 2
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer_cfg = cfg["tokenizer"]
    train_tsv = resolve_path(cfg["data"]["processed_dir"]) / "train.tsv"
    model_prefix = resolve_path(tokenizer_cfg["model_prefix"])
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        n = dump_sentences(train_tsv, tmp)
        corpus_path = tmp.name
    print(f"Tokenizer corpus: {n} sentences")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=str(model_prefix),
        vocab_size=tokenizer_cfg["vocab_size"],
        model_type=tokenizer_cfg["model_type"],
        character_coverage=tokenizer_cfg["character_coverage"],
        split_digits=tokenizer_cfg["split_digits"],
        byte_fallback=tokenizer_cfg["byte_fallback"],
        allow_whitespace_only_pieces=True,
        unk_id=tokenizer_cfg["unk_id"],
        bos_id=tokenizer_cfg["bos_id"],
        eos_id=tokenizer_cfg["eos_id"],
        pad_id=tokenizer_cfg["pad_id"],
        input_sentence_size=tokenizer_cfg["input_sentence_size"],
        shuffle_input_sentence=True,
    )
    print(f"Tokenizer saved: {model_prefix}.model / {model_prefix}.vocab")


if __name__ == "__main__":
    main()
