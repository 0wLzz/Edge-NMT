"""Tokenized parallel dataset and padding collation."""

from __future__ import annotations

import csv
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from common.config import resolve_path


def load_tokenizer(cfg: dict) -> spm.SentencePieceProcessor:
    model_path = resolve_path(cfg["tokenizer"]["model_prefix"]).with_suffix(".model")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_path))
    return tokenizer


def read_pairs(tsv_path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            if len(row) == 2:
                pairs.append((row[0], row[1]))
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
    ):
        self.pairs = read_pairs(tsv_path)
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


def make_collate_fn(pad_id: int):
    def collate(batch):
        sources, targets = zip(*batch)
        source = torch.nn.utils.rnn.pad_sequence(
            sources, batch_first=True, padding_value=pad_id
        )
        target = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=pad_id
        )
        # Teacher forcing: input drops the last token, labels drop BOS.
        return source, target[:, :-1], target[:, 1:]

    return collate
