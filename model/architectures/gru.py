"""GRU student model (encoder-decoder with Luong attention)."""

from model.architectures.recurrent import RecurrentSeq2Seq


class GRUSeq2Seq(RecurrentSeq2Seq):
    CELL = "gru"
