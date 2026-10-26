"""Score candidate KD teachers on FLORES devtest (is the teacher better than the students?).

Sequence-level KD caps a student near its teacher's quality, so a teacher is only
worth distilling from if it beats the baselines trained on the human data. Each
teacher translates FLORES with the same settings as generate_kd_dataset.py
(teacher.num_beams / teacher.max_new_tokens, fp16 on CUDA) and is scored with the
same BLEU + Indonesian METEOR as evaluate.py. Generation time is reported too,
since it decides how long regenerating kd_train.tsv with that teacher would take.

Writes results/teacher_eval/<teacher>.json (metrics + timing) and
<teacher>_translations.csv per teacher.

Usage:
    python -m model.evaluation.evaluate_teacher --teachers mbart-50 nllb-1.3B
    python -m model.evaluation.evaluate_teacher --teachers nllb-3.3B --flores /kaggle/input/.../flores_devtest.tsv
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common.config import load_config, pick_device, resolve_path
from common.results import RESULTS_DIR, save_json
from model.evaluation.evaluate import read_flores
from model.evaluation.metrics import compute_all

# key -> (Hugging Face model, source language code, target language code)
TEACHERS = {
    "mbart-50": ("facebook/mbart-large-50-many-to-many-mmt", "en_XX", "id_ID"),
    "nllb-600M": ("facebook/nllb-200-distilled-600M", "eng_Latn", "ind_Latn"),
    "nllb-1.3B-distilled": ("facebook/nllb-200-distilled-1.3B", "eng_Latn", "ind_Latn"),
    "nllb-1.3B": ("facebook/nllb-200-1.3B", "eng_Latn", "ind_Latn"),
    "nllb-3.3B": ("facebook/nllb-200-3.3B", "eng_Latn", "ind_Latn"),
}


def translate_all(sources, tokenizer, model, forced_bos, beams, max_new, batch_size, device):
    hypotheses = []
    with torch.no_grad():
        for start in range(0, len(sources), batch_size):
            batch = tokenizer(
                sources[start : start + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=256,
            ).to(device)
            generated = model.generate(
                **batch, forced_bos_token_id=forced_bos,
                num_beams=beams, max_new_tokens=max_new,
            )
            hypotheses += tokenizer.batch_decode(generated, skip_special_tokens=True)
    return hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers", nargs="+", default=["mbart-50"], choices=list(TEACHERS))
    parser.add_argument("--flores", default=None, help="FLORES csv (default: data.raw_dir/flores_<split>.tsv)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="only the first N sentences (smoke test)")
    parser.add_argument("--out-dir", default=None, help="default: results/teacher_eval")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    split = cfg["evaluation"]["flores_split"]
    flores_path = (
        Path(args.flores).expanduser().resolve() if args.flores
        else resolve_path(cfg["data"]["raw_dir"]) / f"flores_{split}.tsv"
    )
    examples = read_flores(flores_path)[: args.limit]
    sources = [ex["source"] for ex in examples]
    references = [ex["reference"] for ex in examples]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else RESULTS_DIR / "teacher_eval"
    beams, max_new = cfg["teacher"]["num_beams"], cfg["teacher"]["max_new_tokens"]
    # Same precision generate_kd_dataset.load_teacher uses.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"FLORES {split}: {len(sources)} sentences | device={device} dtype={dtype} beams={beams}")

    for key in args.teachers:
        name, src_lang, tgt_lang = TEACHERS[key]
        print(f"\n===== {key} ({name}) =====", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(name, src_lang=src_lang)
        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

        start = time.time()
        hypotheses = translate_all(
            sources, tokenizer, model, forced_bos, beams, max_new, args.batch_size, device
        )
        seconds = time.time() - start

        row = {
            "teacher": key,
            "model": name,
            "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
            "device": str(device),
            "dtype": str(dtype),
            "flores_split": split,
            "num_sentences": len(sources),
            "seconds": round(seconds, 4),
            "sec_per_sentence": round(seconds / max(len(sources), 1), 4),
            **compute_all(hypotheses, references, chrf=False, meteor=True),
        }
        save_json(out_dir / f"{key}.json", row)
        with open(out_dir / f"{key}_translations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "source", "reference", "hypothesis"])
            for ex, hyp in zip(examples, hypotheses):
                writer.writerow([ex["id"], ex["source"], ex["reference"], hyp])
        print(f"[{key}] BLEU={row['bleu']:.4f} METEOR={row.get('meteor', float('nan')):.4f} "
              f"| {row['sec_per_sentence']:.4f} s/sentence | -> {out_dir / (key + '.json')}", flush=True)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
