"""LSTM student model (encoder-decoder with Luong attention)."""

from model.architectures.recurrent import RecurrentSeq2Seq


class LSTMSeq2Seq(RecurrentSeq2Seq):
    CELL = "lstm"
