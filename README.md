# Edge Machine Translation with Knowledge Distillation

Thesis code: English→Indonesian neural machine translation students (GRU, LSTM,
Transformer) trained with **sequence-level knowledge distillation** from
mBART-50, optionally with **quantization-aware training (QAT)**, converted to
CoreML for on-device inference on iOS.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. For the optional language-ID filter, download
[lid.176.bin](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)
to `data/lid.176.bin` (preprocessing runs without it, skipping that filter).

All commands run from this directory as modules (`python -m ...`). Every stage
reads `configs/config.yaml`; tweak dataset size, search space, and training settings
there rather than in code.

## Pipeline

Run the stages in order:

```bash
# 1. Download CCMatrix (en-id) and FLORES-200 into data/raw/
python -m preprocessing_dataset.download_data

# 2. Clean + dedup + split -> data/processed/{train,valid}.tsv
python -m preprocessing_dataset.run_preprocessing

# 3. Train the shared SentencePiece BPE tokenizer (32k vocab)
python -m preprocessing_dataset.train_tokenizer

# 4. Sequence-level KD data: mBART-50 translates the training sources
#    -> data/processed/kd_train.tsv  (GPU strongly recommended)
#    Resumable: progress saved per batch; rerun the same command to continue.
python -m model.training.generate_kd_dataset

# 5. Optuna hyperparameter search per architecture (TPE + Hyperband pruner,
#    top-300k train slice, resumable SQLite study at results/hparam_search/<arch>.db)
#    -> results/hparam_search/best_<arch>.json  (+ per-trial torchinfo summaries)
#    n_trials in config is a TOTAL budget: rerun to continue after a Colab drop.
python -m model.training.hyperparameter_search --arch gru
python -m model.training.hyperparameter_search --arch lstm
python -m model.training.hyperparameter_search --arch transformer

# 6. Full training (picks up the best hparams automatically)
#    -> results/runs/<timestamp>_<name>/best.pt
#    Resumable: last.pt written every epoch; add bare --resume to continue the
#    latest matching run (fresh start if none exists).
python -m model.training.train --arch transformer --data-mode baseline
python -m model.training.train --arch transformer --data-mode kd
python -m model.training.train --arch transformer --data-mode kd --qat --resume

# 7. Evaluate on FLORES-200 devtest (BLEU, METEOR, size, params, latency)
#    chrF++ is available but off by default (evaluation.compute_chrf).
#    METEOR needs one-time data: python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
#    -> metrics.json in the run dir + a row in results/summary.csv
python -m model.evaluation.evaluate --checkpoint results/runs/<run>/best.pt

# 8. Fold QAT fake-quantization into the weights -> quantized.pt
python -m quantization_coreml.quantize --checkpoint results/runs/<run>/best.pt

# 9. Convert to CoreML -> results/coreml/<name>/{Encoder,Decoder}.mlpackage
python -m quantization_coreml.convert_coreml \
    --checkpoint results/runs/<run>/quantized.pt --name transformer_kd_int8

# 10. Verify CoreML output matches PyTorch (macOS only)
python -m quantization_coreml.verify \
    --checkpoint results/runs/<run>/quantized.pt \
    --coreml-dir results/coreml/transformer_kd_int8
```

For quick end-to-end smoke tests, most scripts accept `--limit N`, and you can
shrink `data.ccmatrix.max_pairs` and `training.epochs` in the config.

## Running on Google Colab (free GPU, daily sessions)

The GPU-heavy stages (KD generation, training) are designed to survive Colab
timeouts: KD generation checkpoints per batch, training per epoch, and both
resume automatically. Workflow:

1. Upload this `CODE` folder to Google Drive at `MyDrive/Skripsi/CODE`
   (exclude `.venv`).
2. Open `colab/skripsi_colab.ipynb` in Colab, select a GPU runtime.
3. Each day: `Runtime > Run all`. Completed stages skip themselves; the
   current stage continues where the last session died. All data and results
   live on Drive, so nothing is lost between sessions.
4. Stages 8-10 (QAT folding, CoreML conversion, verification) run on the Mac —
   CoreML needs macOS. Drive syncs `results/` back to you.

## Folder structure

```
configs/                  Central config (config.yaml)
common/                   Config loading, seeding, results helpers
preprocessing_dataset/    Download, cleaning, split, tokenizer training
model/
  architectures/          GRU / LSTM / Transformer + factory
  qat.py                  Asymmetric uint8 fake quantization (toggle on/off)
  training/               Dataset, trainer, KD data generation, Optuna search, train CLI
  evaluation/             BLEU + Indonesian METEOR (chrF++ optional), efficiency metrics, evaluate CLI
quantization_coreml/      QAT folding, CoreML conversion, PyTorch-vs-CoreML verification
results/                  Every run's outputs land here (see below)
data/                     Raw + processed datasets (gitignored, recreatable)
```

## Results layout

- `results/summary.csv` — one row per evaluation; the cross-experiment
  comparison table (arch, data mode, QAT, quantized, BLEU, METEOR, size,
  latency; the chrF++ column stays present but empty unless enabled).
- `results/runs/<timestamp>_<name>/` — checkpoint, run config, training
  history, metrics, decoded hypotheses.
- `results/hparam_search/` — all trials + `best_<arch>.json` per architecture.
- `results/coreml/<name>/` — `Encoder.mlpackage`, `Decoder.mlpackage`,
  `tokenizer.model`, `meta.json`, `verification.json`. Drop this bundle into
  the iOS app.

## Method notes

- **Sequence-level KD** (Kim & Rush, 2016): the teacher translates the training
  sources once; students train on those outputs with plain cross-entropy. This
  avoids the vocabulary mismatch between mBART-50 and the students' 32k
  SentencePiece vocab that token-level KL distillation would hit.
- **QAT scope**: asymmetric per-tensor uint8 fake quantization on all
  `nn.Linear` layers (weights + input activations). Embeddings and recurrent
  cells stay float. `fold_qat` bakes the weight grid into the checkpoint so
  conversion needs no custom ops; the `.mlpackage` itself stores weights in
  float (true int8 storage would need `coremltools.optimize`, deliberately not
  used here).
- **CoreML export** splits each model into an encoder and a single-step
  decoder; the iOS app owns the greedy loop. The transformer decoder is traced
  at fixed `max_target_length` (pads sit after the prefix, so the causal mask
  already ignores them; read logits at `prefix_length - 1`). Source-length
  inputs use `EnumeratedShapes` over every length 1..128 (`RangeDim` hangs
  coremltools on these graphs), so the app always feeds the exact tokenized
  length — never pad the source.
- Preprocessing keeps casing and punctuation so sacreBLEU scores on raw
  FLORES-200 references stay comparable with published results.
