"""Transformer student model (standard encoder-decoder, batch-first)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.shape[1]]


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def _embed(self, tokens):
        return self.dropout(
            self.positional_encoding(self.embedding(tokens) * math.sqrt(self.d_model))
        )

    def _causal_mask(self, length: int, device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length, device=device)

    def encode(self, source):
        source_padding_mask = source == self.pad_id
        memory = self.transformer.encoder(
            self._embed(source), src_key_padding_mask=source_padding_mask
        )
        return memory, source_padding_mask

    def decode(self, target_input, memory, source_padding_mask=None):
        """Decode a (growing) target prefix against encoder memory. Returns logits [B, T, V]."""
        target_mask = self._causal_mask(target_input.shape[1], target_input.device)
        decoded = self.transformer.decoder(
            self._embed(target_input),
            memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_input == self.pad_id,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output_projection(decoded)

    def forward(self, source, target_input):
        memory, source_padding_mask = self.encode(source)
        return self.decode(target_input, memory, source_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, source, bos_id: int, eos_id: int, max_length: int):
        """Batched greedy decoding. Returns list of token-id lists (no BOS/EOS)."""
        self.eval()
        batch_size = source.shape[0]
        memory, source_padding_mask = self.encode(source)

        prefix = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
        outputs = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            logits = self.decode(prefix, memory, source_padding_mask)
            next_tokens = logits[:, -1].argmax(dim=-1)
            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_tokens[i].item()
                    if token_id == eos_id:
                        finished[i] = True
                    else:
                        outputs[i].append(token_id)
            if bool(finished.all()):
                break
            # Finished rows keep appending EOS so their logits stay harmless.
            next_tokens = next_tokens.masked_fill(finished, eos_id)
            prefix = torch.cat([prefix, next_tokens.unsqueeze(1)], dim=1)
        return outputs
