"""Shared encoder-decoder base for the GRU and LSTM students.

Bidirectional recurrent encoder + unidirectional recurrent decoder with
Luong (general) attention computed over the full decoded sequence, so
training is fully teacher-forced and vectorized; decoding reuses the same
modules one step at a time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_RNN_CLASSES = {"gru": nn.GRU, "lstm": nn.LSTM}


class LuongAttention(nn.Module):
    """score(h_dec, h_enc) = h_dec . W h_enc  (general form)."""

    def __init__(self, decoder_dim: int, encoder_dim: int):
        super().__init__()
        self.project = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(self, decoder_states, encoder_states, source_mask=None):
        # decoder_states: [B, T, D]; encoder_states: [B, S, E]; source_mask: [B, S]
        # source_mask=None means "no padding" (single-sentence CoreML export).
        scores = decoder_states @ self.project(encoder_states).transpose(1, 2)  # [B, T, S]
        if source_mask is not None:
            scores = scores.masked_fill(~source_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ encoder_states  # [B, T, E]


class RecurrentSeq2Seq(nn.Module):
    CELL: str = "gru"  # overridden by subclasses

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = pad_id
        rnn_class = _RNN_CLASSES[self.CELL]
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.encoder = rnn_class(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=rnn_dropout,
        )
        self.decoder = rnn_class(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=rnn_dropout,
        )
        # Merge forward/backward final encoder states into decoder initial states.
        self.bridge_hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        if self.CELL == "lstm":
            self.bridge_cell = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention = LuongAttention(hidden_dim, 2 * hidden_dim)
        self.combine = nn.Linear(hidden_dim + 2 * hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    # ---- encoding -------------------------------------------------------

    def _bridge(self, state, projection):
        # state: [num_layers * 2, B, H] -> [num_layers, B, H]
        num_directions = 2
        layers, batch, hidden = state.shape[0] // num_directions, state.shape[1], state.shape[2]
        state = state.view(layers, num_directions, batch, hidden)
        merged = torch.cat([state[:, 0], state[:, 1]], dim=-1)
        return torch.tanh(projection(merged))

    def encode(self, source):
        source_mask = source != self.pad_id
        encoder_states, final_state = self.encoder(self.dropout(self.embedding(source)))
        if self.CELL == "lstm":
            hidden, cell = final_state
            decoder_state = (
                self._bridge(hidden, self.bridge_hidden),
                self._bridge(cell, self.bridge_cell),
            )
        else:
            decoder_state = self._bridge(final_state, self.bridge_hidden)
        return encoder_states, decoder_state, source_mask

    # ---- training forward (teacher forcing) -----------------------------

    def forward(self, source, target_input):
        encoder_states, decoder_state, source_mask = self.encode(source)
        return self._decode_sequence(target_input, decoder_state, encoder_states, source_mask)[0]

    def _decode_sequence(self, target_input, decoder_state, encoder_states, source_mask):
        decoder_states, new_state = self.decoder(
            self.dropout(self.embedding(target_input)), decoder_state
        )
        context = self.attention(decoder_states, encoder_states, source_mask)
        combined = torch.tanh(self.combine(torch.cat([decoder_states, context], dim=-1)))
        return self.output_projection(self.dropout(combined)), new_state

    # ---- inference -------------------------------------------------------

    def decode_step(self, token, decoder_state, encoder_states, source_mask):
        """One greedy-decode step. token: [B, 1] -> logits [B, vocab]."""
        logits, new_state = self._decode_sequence(
            token, decoder_state, encoder_states, source_mask
        )
        return logits.squeeze(1), new_state

    @torch.no_grad()
    def greedy_decode(self, source, bos_id: int, eos_id: int, max_length: int):
        """Batched greedy decoding. Returns list of token-id lists (no BOS/EOS)."""
        self.eval()
        batch_size = source.shape[0]
        encoder_states, decoder_state, source_mask = self.encode(source)

        tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        outputs = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            logits, decoder_state = self.decode_step(
                tokens, decoder_state, encoder_states, source_mask
            )
            next_tokens = logits.argmax(dim=-1)
            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_tokens[i].item()
                    if token_id == eos_id:
                        finished[i] = True
                    else:
                        outputs[i].append(token_id)
            if bool(finished.all()):
                break
            tokens = next_tokens.unsqueeze(1)
        return outputs
