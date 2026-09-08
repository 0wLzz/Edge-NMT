#!/usr/bin/env python3
"""Render Keras-`plot_model`-style architecture diagrams for the three students.

Produces one block diagram per architecture (GRU, LSTM, Transformer) in the
"black header + input/output shape" style, driven by the winning hyperparameters
from the Optuna search (results/hparam_search/best_<arch>.json). Intended for
Bab 3 of the thesis.

Requirements: Graphviz `dot` binary on PATH + the `graphviz` Python package
    (pip install graphviz ; and `brew install graphviz` / `apt-get install graphviz`).

Usage (from the CODE/ directory):
    python -m visualization.plot_architectures
    python -m visualization.plot_architectures --out-dir ../Visualization/architecture_diagrams
    python -m visualization.plot_architectures --formats png pdf svg
    python -m visualization.plot_architectures --seq-source 76 --seq-target 80   # concrete lengths

The sequence lengths default to the symbols S (source) and T (target) so the
diagram reads as variable-length, matching the model (batch = None).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    import graphviz
except ImportError as e:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install graphviz") from e

# ------------------------------------------------------------------ defaults
VOCAB = 32000                      # SentencePiece vocab (configs/config.yaml -> tokenizer.vocab_size)
HERE = Path(__file__).resolve().parent
BEST_DIR = HERE.parent / "results" / "hparam_search"

FALLBACK = {  # used if a best_<arch>.json is missing
    "gru":         dict(embedding_dim=512, hidden_dim=512,  num_layers=2, dropout=0.25),
    "lstm":        dict(embedding_dim=256, hidden_dim=1024, num_layers=3, dropout=0.17),
    "transformer": dict(d_model=512, num_heads=8, num_layers=3, ffn_dim=3072, dropout=0.12),
}

# Keras-plot_model palette
HEADER_BG = "#111111"
NODE_FONT = "Helvetica"

def load_hparams(arch: str) -> dict:
    f = BEST_DIR / f"best_{arch}.json"
    if f.exists():
        hp = json.loads(f.read_text()).get("hparams", {})
        return {**FALLBACK[arch], **hp}
    print(f"  ! {f.name} not found, using fallback dims for {arch}")
    return FALLBACK[arch]


def _label(title: str, in_shape: str | None, out_shape: str) -> str:
    """HTML-like record label: black title bar + shape row(s)."""
    if in_shape is None:  # input layer -> single output cell
        body = (f'<TR><TD ALIGN="LEFT" CELLPADDING="6">'
                f'Output shape: <B>{out_shape}</B></TD></TR>')
        cols = 1
    else:
        body = (f'<TR>'
                f'<TD ALIGN="LEFT" CELLPADDING="6">Input shape: <B>{in_shape}</B></TD>'
                f'<TD ALIGN="LEFT" CELLPADDING="6">Output shape: <B>{out_shape}</B></TD>'
                f'</TR>')
        cols = 2
    return (
        f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
        f'<TR><TD COLSPAN="{cols}" BGCOLOR="{HEADER_BG}" CELLPADDING="7">'
        f'<FONT COLOR="white" FACE="{NODE_FONT}" POINT-SIZE="13"><B>{title}</B></FONT>'
        f'</TD></TR>{body}</TABLE>>'
    )


def _new_graph(name: str) -> graphviz.Digraph:
    g = graphviz.Digraph(name, format="png")
    g.attr(rankdir="TB", splines="ortho", nodesep="0.35", ranksep="0.5", bgcolor="white")
    g.attr("node", shape="plaintext", fontname=NODE_FONT, fontsize="11", margin="0")
    g.attr("edge", color="#333333", arrowsize="0.8")
    return g


def node(g, nid, title, in_shape, out_shape):
    g.node(nid, _label(title, in_shape, out_shape))


# ------------------------------------------------------------------ recurrent
def build_recurrent(arch: str, hp: dict) -> graphviz.Digraph:
    E, H, L = hp["embedding_dim"], hp["hidden_dim"], hp["num_layers"]
    cell = arch.upper()  # GRU / LSTM
    S, T, V = "S", "T", VOCAB
    d2, d3 = 2 * H, 3 * H
    g = _new_graph(arch)

    # encoder branch
    node(g, "src_in",  "InputLayer",            None,            f"(None, {S})")
    node(g, "src_emb", "Embedding",             f"(None, {S})",  f"(None, {S}, {E})")
    node(g, "enc_drop","Dropout",               f"(None, {S}, {E})", f"(None, {S}, {E})")
    node(g, "encoder", f"Bidirectional {cell}  (layers={L})",
         f"(None, {S}, {E})", f"(None, {S}, {d2})  + final states")
    node(g, "bridge_h","Bridge — Dense + tanh", f"(None, {d2})", f"(None, {H})")

    # decoder branch
    node(g, "tgt_in",  "InputLayer",            None,            f"(None, {T})")
    node(g, "tgt_emb", "Embedding  (shared)",   f"(None, {T})",  f"(None, {T}, {E})")
    node(g, "dec_drop","Dropout",               f"(None, {T}, {E})", f"(None, {T}, {E})")
    node(g, "decoder", f"{cell}  (layers={L})",
         f"(None, {T}, {E})  + init state", f"(None, {T}, {H})")
    node(g, "attn",    "Luong Attention",
         f"[(None, {T}, {H}), (None, {S}, {d2})]", f"(None, {T}, {d2})")
    node(g, "combine", "Combine — Dense + tanh",
         f"(None, {T}, {d3})", f"(None, {T}, {H})")
    node(g, "out_drop","Dropout",               f"(None, {T}, {H})", f"(None, {T}, {H})")
    node(g, "outproj", "Dense  (output projection)",
         f"(None, {T}, {H})", f"(None, {T}, {V})")

    if cell == "LSTM":
        node(g, "bridge_c", "Bridge (cell) — Dense + tanh", f"(None, {d2})", f"(None, {H})")

    g.edge("src_in", "src_emb"); g.edge("src_emb", "enc_drop"); g.edge("enc_drop", "encoder")
    g.edge("encoder", "bridge_h")
    g.edge("bridge_h", "decoder")
    if cell == "LSTM":
        g.edge("encoder", "bridge_c"); g.edge("bridge_c", "decoder")
    g.edge("tgt_in", "tgt_emb"); g.edge("tgt_emb", "dec_drop"); g.edge("dec_drop", "decoder")
    g.edge("decoder", "attn"); g.edge("encoder", "attn")
    g.edge("decoder", "combine"); g.edge("attn", "combine")
    g.edge("combine", "out_drop"); g.edge("out_drop", "outproj")
    return g


# ------------------------------------------------------------------ transformer
def build_transformer(hp: dict) -> graphviz.Digraph:
    d, heads, L, ffn = hp["d_model"], hp["num_heads"], hp["num_layers"], hp["ffn_dim"]
    S, T, V = "S", "T", VOCAB
    g = _new_graph("transformer")

    node(g, "src_in",  "InputLayer",              None,            f"(None, {S})")
    node(g, "src_emb", "Embedding  (× √d_model)", f"(None, {S})",  f"(None, {S}, {d})")
    node(g, "src_pos", "Positional Encoding",     f"(None, {S}, {d})", f"(None, {S}, {d})")
    node(g, "src_drop","Dropout",                 f"(None, {S}, {d})", f"(None, {S}, {d})")
    node(g, "enc", f"Transformer Encoder  (layers={L}, heads={heads}, ffn={ffn})",
         f"(None, {S}, {d})", f"(None, {S}, {d})")

    node(g, "tgt_in",  "InputLayer",              None,            f"(None, {T})")
    node(g, "tgt_emb", "Embedding  (× √d_model)", f"(None, {T})",  f"(None, {T}, {d})")
    node(g, "tgt_pos", "Positional Encoding",     f"(None, {T}, {d})", f"(None, {T}, {d})")
    node(g, "tgt_drop","Dropout",                 f"(None, {T}, {d})", f"(None, {T}, {d})")
    node(g, "dec", f"Transformer Decoder  (layers={L}, heads={heads}, ffn={ffn}, causal)",
         f"[(None, {T}, {d}), memory (None, {S}, {d})]", f"(None, {T}, {d})")
    node(g, "outproj", "Dense  (output projection)",
         f"(None, {T}, {d})", f"(None, {T}, {V})")

    g.edge("src_in", "src_emb"); g.edge("src_emb", "src_pos"); g.edge("src_pos", "src_drop")
    g.edge("src_drop", "enc")
    g.edge("tgt_in", "tgt_emb"); g.edge("tgt_emb", "tgt_pos"); g.edge("tgt_pos", "tgt_drop")
    g.edge("tgt_drop", "dec")
    g.edge("enc", "dec")            # encoder memory -> decoder cross-attention
    g.edge("dec", "outproj")
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "architecture_diagrams"))
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], help="png pdf svg ...")
    ap.add_argument("--archs", nargs="+", default=["gru", "lstm", "transformer"])
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if shutil.which("dot") is None:
        raise SystemExit("Graphviz 'dot' not found on PATH. Install it: "
                         "brew install graphviz  (macOS)  |  apt-get install graphviz  (Linux)")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for arch in args.archs:
        hp = load_hparams(arch)
        g = build_transformer(hp) if arch == "transformer" else build_recurrent(arch, hp)
        g.attr(dpi=str(args.dpi))
        for fmt in args.formats:
            g.format = fmt
            path = g.render(filename=f"{arch}_architecture", directory=str(out), cleanup=True)
            print(f"  saved {path}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
