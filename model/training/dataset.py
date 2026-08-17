"""Tokenized parallel dataset and padding collation."""

from __future__ import annotations

import csv
import functools
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from common.config import PROJECT_ROOT, resolve_path

import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = PROJECT_ROOT / "data/processed"

def load_tokenizer(
    cfg: dict, model_path: str | Path | None = None
) -> spm.SentencePieceProcessor:
    """Load the configured tokenizer, or a model supplied by a caller."""
    if model_path is None:
        model_path = resolve_path(cfg["tokenizer"]["model_prefix"]).with_suffix(".model")
    else:
        model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"SentencePiece tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.Load(str(model_path)):
        raise RuntimeError(f"Could not load SentencePiece tokenizer: {model_path}")
    return tokenizer
 

def read_pairs(tsv_path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """Read (source, target) pairs. With `limit`, stop after that many valid
    pairs -- train.tsv preserves CCMatrix's LASER-margin order, so the first N
    rows are the highest-quality pairs and reading only those avoids loading the
    full (multi-GB) file into memory."""
    pairs = []
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
                if limit is not None and len(pairs) >= limit:
                    break
    return pairs


class TranslationDataset(Dataset):
    """Encodes (source, target) pairs on the fly.

    Source: ids + EOS. Target: BOS + ids + EOS (the trainer shifts it into
    decoder input / label).
    """

    def __init__(
        self,
        tsv_path: Path,
        tokenizer: spm.SentencePieceProcessor,
        max_length: int,
        limit: int | None = None,
    ):
        self.pairs = read_pairs(tsv_path, limit=limit)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_text, target_text = self.pairs[index]
        source = self.tokenizer.encode(source_text)[: self.max_length - 1] + [self.eos_id]
        target = (
            [self.bos_id]
            + self.tokenizer.encode(target_text)[: self.max_length - 2]
            + [self.eos_id]
        )
        return torch.tensor(source), torch.tensor(target)


def _collate(batch, pad_id: int):
    sources, targets = zip(*batch)
    source = torch.nn.utils.rnn.pad_sequence(
        sources, batch_first=True, padding_value=pad_id
    )
    target = torch.nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=pad_id
    )
    # Teacher forcing: input drops the last token, labels drop BOS.
    return source, target[:, :-1], target[:, 1:]


def make_collate_fn(pad_id: int):
    # A module-level function bound with partial (not a local closure) so the
    # collate fn is picklable -- required for DataLoader num_workers > 0 under the
    # macOS/Windows 'spawn' start method.
    return functools.partial(_collate, pad_id=pad_id)
