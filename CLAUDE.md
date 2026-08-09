# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code for an undergraduate thesis (skripsi) on **knowledge distillation for on-device (Edge AI) English→Indonesian machine translation**. Students (GRU, LSTM, Transformer) are distilled from mBART-50, optionally trained with QAT, and converted to CoreML for an iOS app. The full methodology is in `../Proposal PreThesis_KUMPUL.docx`; the README documents the pipeline stage by stage.

Methodology decisions that differ from the original proposal (agreed during review):
- **Sequence-level KD** (Kim & Rush 2016), not token-level KL distillation — mBART-50's 250k vocab and the students' 32k SentencePiece vocab make KL divergence undefined. The teacher pre-translates the training sources (`data/processed/kd_train.tsv`).
- **BLEU + Indonesian-aware METEOR** are the reported metrics. BLEU = surface precision (sacreBLEU); METEOR (`model/evaluation/meteor.py`, gated by `evaluation.compute_meteor`) = recall + morphology + synonymy. Off-the-shelf METEOR degrades to exact-match on Indonesian; this version injects a Sastrawi stemmer and Wordnet Bahasa (OMW `lang="ind"`) into NLTK's `meteor_score` via its `stemmer=`/`wordnet=` params, reusing NLTK's alignment/penalty machinery. One-time data: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`.
- **chrF++** (sacreBLEU) is implemented but OFF by default (`evaluation.compute_chrf: false`): it is another surface n-gram-overlap metric, and METEOR's Sastrawi stemming already recovers the Indonesian morphological credit chrF++ provided, so it is redundant. Flip the flag to report it.
- Preprocessing keeps casing and punctuation so scores on raw FLORES-200 references are comparable with published numbers.
- QAT is custom (asymmetric per-tensor uint8 fake quantization in `model/qat.py`), deliberately **not** `coremltools.optimize`. `fold_qat` bakes weights onto the uint8 grid before conversion.
- Pruning is custom too (gradual, unstructured, per-tensor magnitude pruning in `model/pruning.py`), enabled with `--prune` and deliberately **not** `coremltools.optimize.torch.MagnitudePruner`, to compose with the custom QAT in one training run and reuse the existing export path. It targets the same `nn.Linear` layers as QAT; when `--qat` is on it prunes the `nn.Linear` inside each `QATLinear`, so the two stack with no special-casing. Zeroed weights persist in the (dense) checkpoint with no new state_dict keys, so `evaluate`/`quantize`/`convert` are unchanged; a sparse CoreML representation can be generated later via `ct.optimize.coreml.prune_weights`. **One-shot KD + quantize + prune = `--data-mode kd --qat --prune`.**

## Commands

Environment: `.venv` in this directory (Python 3.9 — hence `from __future__ import annotations` in every module using `X | Y` type syntax). Activate with `source .venv/bin/activate`; run everything from this directory as modules.

Pipeline order (details and flags in README.md):

```bash
python -m preprocessing_dataset.download_data        # CCMatrix + FLORES-200 -> data/raw/
python -m preprocessing_dataset.run_preprocessing    # clean/dedup/split -> data/processed/
python -m preprocessing_dataset.train_tokenizer      # SentencePiece BPE 32k
python -m model.training.generate_kd_dataset         # teacher translations (GPU recommended)
python -m model.training.hyperparameter_search --arch <gru|lstm|transformer>
python -m model.training.train --arch <arch> --data-mode <baseline|kd> [--qat] [--prune] [--target-sparsity 0.5] [--resume]
python -m model.evaluation.evaluate --checkpoint results/runs/<run>/best.pt
python -m quantization_coreml.quantize --checkpoint results/runs/<run>/best.pt
python -m quantization_coreml.convert_coreml --checkpoint .../quantized.pt --name <name>
python -m quantization_coreml.verify --checkpoint .../quantized.pt --coreml-dir results/coreml/<name>
```

Smoke-testing: most CLIs accept `--limit N`; shrink `data.ccmatrix.max_pairs` and `training.epochs` in `configs/config.yaml` for quick runs.

GPU stages are Colab-resumable (`colab/skripsi_colab.ipynb`, project on Google Drive): `generate_kd_dataset` checkpoints per batch via `kd_train.progress.json`; `train` writes `last.pt` (weights+optimizer+early-stopping) every epoch, and bare `--resume` continues the newest run matching arch/data-mode/qat. CoreML stages stay on macOS.

## Architecture

- `configs/config.yaml` is the single source of truth for all stage parameters; scripts take `--config` but default to it. Paths in it are project-root-relative (resolved by `common/config.py:resolve_path`).
- Checkpoints are self-describing: `factory.save_checkpoint` stores `{meta, state_dict}` where meta has arch/hparams/vocab/special-ids/qat flag, and `factory.load_checkpoint` rebuilds the model (including QAT wrappers) from it. Downstream stages (evaluate, quantize, convert) never need the original hparams JSON.
- Every evaluation appends a row to `results/summary.csv` — that file is the cross-experiment comparison table for the thesis. Run artifacts live in `results/runs/<timestamp>_<name>/`.
- QAT gotchas encoded in `model/qat.py`: never descend into `QATLinear` when wrapping (infinite recursion), skip `nn.MultiheadAttention` (reads `out_proj.weight` directly), and the transformer fused fast path is disabled so eval doesn't silently bypass fake quantization.
- Pruning gotchas in `model/pruning.py`: masks are applied by zeroing `weight.data` in-place after each optimizer step (hard masking), *not* via forward hooks — a hook on the inner `nn.Linear` would never fire because `QATLinear.forward` calls `F.linear` on `self.linear.weight` directly rather than `self.linear(x)`. The pruner registers no buffers (masks live in the pruner object), so it adds no state_dict keys; `--resume` rebuilds the mask from the zero pattern of the loaded weights via `restore_from_model`. The sparsity ramp is anchored to `training.epochs`, so early stopping can cut it short before target sparsity — raise `early_stopping_patience`/`epochs` for aggressive sparsity.
- CoreML export splits each model into Encoder + single-step Decoder `.mlpackage` (wrappers in `quantization_coreml/export_wrappers.py`); the iOS app owns the greedy loop. The transformer decoder is traced at fixed `max_target_length` with no padding mask — pads sit after the prefix so the causal mask already excludes them; read logits at `prefix_length - 1`. Source-length inputs use `EnumeratedShapes` (1..128) — `ct.RangeDim` hangs coremltools indefinitely on these graphs; the app must feed exact tokenized lengths, never pad the source. Traces use `check_trace=False` (encoder/decoder wrappers share modules, which mangles retrace class names); correctness is checked by `quantization_coreml/verify.py` instead.
- Special token ids are fixed at tokenizer training time (unk=0, bos=1, eos=2, pad=3) and flow through checkpoints and the CoreML bundle's `meta.json`.

## Conventions

- Thesis documents, reference papers, and diagrams live in the parent directory (`../`); keep this directory code-only.
- Data files are TSV (tab-separated, `csv.QUOTE_NONE` with `\\` escape) throughout.
