"""Build models from hyperparameter dicts and rebuild them from checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from model.architectures.gru import GRUSeq2Seq
from model.architectures.lstm import LSTMSeq2Seq
from model.architectures.transformer import TransformerSeq2Seq
from model.qat import apply_qat

ARCHITECTURES = {
    "gru": GRUSeq2Seq,
    "lstm": LSTMSeq2Seq,
    "transformer": TransformerSeq2Seq,
}

# Hyperparameters accepted by each architecture's constructor; anything else
# in a search grid (learning_rate, batch_size) belongs to the trainer.
_MODEL_HPARAMS = {
    "gru": ("embedding_dim", "hidden_dim", "num_layers", "dropout"),
    "lstm": ("embedding_dim", "hidden_dim", "num_layers", "dropout"),
    "transformer": ("d_model", "num_heads", "num_layers", "ffn_dim", "dropout"),
}


def split_hparams(arch: str, hparams: dict) -> tuple[dict, dict]:
    """Split a flat hyperparameter dict into (model kwargs, trainer kwargs)."""
    model_keys = _MODEL_HPARAMS[arch]
    model_hparams = {k: v for k, v in hparams.items() if k in model_keys}
    trainer_hparams = {k: v for k, v in hparams.items() if k not in model_keys}
    return model_hparams, trainer_hparams


def build_model(
    arch: str, vocab_size: int, pad_id: int, hparams: dict, qat: bool = False
) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(ARCHITECTURES)}")
    model_hparams, _ = split_hparams(arch, hparams)
    model = ARCHITECTURES[arch](vocab_size=vocab_size, pad_id=pad_id, **model_hparams)
    if qat:
        replaced = apply_qat(model)
        print(f"[qat] Fake quantization applied to {replaced} Linear layers")
    return model


def save_checkpoint(path: Path, model: nn.Module, meta: dict) -> None:
    """Save weights plus everything needed to rebuild the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild a model (including its QAT wrappers) from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model = build_model(
        arch=meta["arch"],
        vocab_size=meta["vocab_size"],
        pad_id=meta["pad_id"],
        hparams=meta["hparams"],
        qat=meta.get("qat", False),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), meta
