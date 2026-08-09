"""Trace-friendly wrappers that split each seq2seq model into an encoder
module and a single-step decoder module for CoreML export.

The iOS app runs the greedy loop: encode once, then call the decoder
repeatedly, feeding back the previous token (and, for GRU/LSTM, the hidden
state; for the transformer, the growing prefix padded to a fixed length).
Wrappers assume a single unpadded sentence (batch size 1), which is exactly
the on-device inference scenario.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentEncoderWrapper(nn.Module):
    """source [1, S] -> encoder_states [1, S, 2H], initial decoder state(s)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_lstm = model.CELL == "lstm"

    def forward(self, source):
        encoder_states, decoder_state, _ = self.model.encode(source)
        if self.is_lstm:
            hidden, cell = decoder_state
            return encoder_states, hidden, cell
        return encoder_states, decoder_state


class RecurrentDecoderWrapper(nn.Module):
    """One greedy step: (token, encoder_states, state) -> (logits, new state)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_lstm = model.CELL == "lstm"

    def forward(self, token, encoder_states, hidden, cell=None):
        state = (hidden, cell) if self.is_lstm else hidden
        logits, new_state = self.model._decode_sequence(
            token, state, encoder_states, source_mask=None
        )
        logits = logits.squeeze(1)  # [1, vocab]
        if self.is_lstm:
            return logits, new_state[0], new_state[1]
        return logits, new_state


class TransformerEncoderWrapper(nn.Module):
    """source [1, S] -> memory [1, S, D] (no padding mask: single sentence)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, source):
        return self.model.transformer.encoder(self.model._embed(source))


class TransformerDecoderWrapper(nn.Module):
    """Fixed-length decoding step for traceability.

    The prefix is padded with pad_id up to max_length. No padding mask is
    needed: pads sit strictly after the last real token, and the fixed causal
    mask already stops every earlier position from attending to them. The
    client reads logits at index (current_prefix_length - 1).
    """

    def __init__(self, model, max_length: int):
        super().__init__()
        self.model = model
        causal_mask = nn.Transformer.generate_square_subsequent_mask(max_length)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, prefix, memory):
        decoded = self.model.transformer.decoder(
            self.model._embed(prefix), memory, tgt_mask=self.causal_mask
        )
        return self.model.output_projection(decoded)  # [1, max_length, vocab]


def build_wrappers(model, arch: str, max_target_length: int):
    """Return (encoder_wrapper, decoder_wrapper) in eval mode."""
    model = model.eval()
    if arch in ("gru", "lstm"):
        return RecurrentEncoderWrapper(model), RecurrentDecoderWrapper(model)
    if arch == "transformer":
        return TransformerEncoderWrapper(model), TransformerDecoderWrapper(
            model, max_target_length
        )
    raise ValueError(f"Unknown architecture '{arch}'")
