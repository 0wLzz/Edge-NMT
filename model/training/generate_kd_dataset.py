"""Sequence-level knowledge distillation data generation.

The teacher (configs teacher.model_name: NLLB-200 1.3B, previously mBART-50)
translates every source sentence of the cleaned training split once. Students
then train on (source, teacher_translation) pairs with plain cross-entropy --
this is sequence-level KD (Kim & Rush, 2016) and avoids the vocabulary mismatch
between the teacher's tokenizer and the students' SentencePiece vocabulary.

Output: data/processed/<teacher.output_file>, rows in the same order as the
source train.tsv. Each teacher gets its own file name, so a new teacher never
overwrites an earlier teacher's data.

Speed:
  * length-sorted batching -- every `teacher.chunk_size` sources are sorted by
    length before batching (far less padding per batch), then written back in
    the original order;
  * sharding -- `--num-shards 2 --shard-id {0,1}` splits the sources into
    contiguous halves, one per GPU (Kaggle has 2x T4); `--merge 2` joins the
    finished shards into the output file;
  * a batch that runs out of GPU memory is retried in halves instead of crashing.

Resumable: after every chunk the output byte size and source count go to
<output>.progress.json; a rerun truncates the output to the last recorded size
(dropping a half-written chunk) and continues from there. `--time-budget` ends
the session cleanly after the last chunk that fits.

Usage:
    python -m model.training.generate_kd_dataset [--limit N] [--time-budget 11.5]
    CUDA_VISIBLE_DEVICES=0 python -m model.training.generate_kd_dataset --num-shards 2 --shard-id 0 &
    CUDA_VISIBLE_DEVICES=1 python -m model.training.generate_kd_dataset --num-shards 2 --shard-id 1
    python -m model.training.generate_kd_dataset --merge 2
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from common.config import load_config, pick_device, resolve_path
from common.results import load_json, save_json
from model.training.dataset import read_pairs


def load_teacher(cfg: dict, device: torch.device):
    teacher_cfg = cfg["teacher"]

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_cfg["model_name"], src_lang=teacher_cfg["source_lang_code"]
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(teacher_cfg["model_name"], torch_dtype=dtype)
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def translate_batch(sources, tokenizer, model, teacher_cfg, device) -> list[str]:
    inputs = tokenizer(
        sources, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(teacher_cfg["target_lang_code"]),
        num_beams=teacher_cfg["num_beams"],
        max_new_tokens=teacher_cfg["max_new_tokens"],
    )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_safely(sources, tokenizer, model, teacher_cfg, device) -> list[str]:
    """translate_batch, retried in halves if the batch runs out of GPU memory
    (length-sorted batches put all the longest sentences together)."""
    try:
        return translate_batch(sources, tokenizer, model, teacher_cfg, device)
    except torch.cuda.OutOfMemoryError:
        if len(sources) == 1:
            raise
        torch.cuda.empty_cache()
        mid = len(sources) // 2
        return (translate_safely(sources[:mid], tokenizer, model, teacher_cfg, device)
                + translate_safely(sources[mid:], tokenizer, model, teacher_cfg, device))


def output_paths(cfg: dict, num_shards: int = 1, shard_id: int = 0) -> tuple[Path, Path]:
    """(tsv, progress.json) for the whole output or for one shard of it."""
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    stem = Path(cfg["teacher"].get("output_file", "kd_train.tsv")).stem
    if num_shards > 1:
        stem += f".shard{shard_id}of{num_shards}"
    return processed_dir / f"{stem}.tsv", processed_dir / f"{stem}.progress.json"


def merge_shards(cfg: dict, num_shards: int) -> None:
    """Concatenate finished shards (in order) into the output file."""
    output_path, progress_path = output_paths(cfg)
    shards = [output_paths(cfg, num_shards, k) for k in range(num_shards)]
    total = 0
    for tsv, progress in shards:
        state = load_json(progress) if progress.exists() else None
        if not state or state["processed_sources"] < state["total_sources"]:
            raise SystemExit(f"{tsv.name} is not finished yet ({progress.name}: {state})")
        total += state["total_sources"]
    with open(output_path, "wb") as out:
        for tsv, _ in shards:
            with open(tsv, "rb") as f:
                shutil.copyfileobj(f, out)
    save_json(progress_path, {"processed_sources": total, "total_sources": total,
                              "output_bytes": output_path.stat().st_size,
                              "teacher": cfg["teacher"]["model_name"], "merged_shards": num_shards})
    print(f"Merged {num_shards} shards ({total} sources) -> {output_path}")


def main() -> None:
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Translate only the first N pairs")
    parser.add_argument("--num-shards", type=int, default=1, help="split the sources into N contiguous shards")
    parser.add_argument("--shard-id", type=int, default=0, help="which shard this process translates (0-based)")
    parser.add_argument("--merge", type=int, default=None, metavar="N",
                        help="join the N finished shards into the output file and exit")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="hours: stop cleanly after the last chunk that fits (e.g. 11.5 on Kaggle)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.merge:
        merge_shards(cfg, args.merge)
        return

    teacher_cfg = cfg["teacher"]
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    output_path, progress_path = output_paths(cfg, args.num_shards, args.shard_id)

    pairs = read_pairs(processed_dir / "2M" / "train.tsv", limit=args.limit)
    begin = len(pairs) * args.shard_id // args.num_shards
    end = len(pairs) * (args.shard_id + 1) // args.num_shards
    sources = [source for source, _ in pairs[begin:end]]
    del pairs
    print(f"Teacher {teacher_cfg['model_name']} | shard {args.shard_id + 1}/{args.num_shards}: "
          f"sources {begin}..{end} ({len(sources)}) -> {output_path.name}")

    # Resume from the last completed chunk; drop anything written after it.
    done, output_bytes = 0, 0
    if progress_path.exists() and output_path.exists():
        state = load_json(progress_path)
        done = min(state["processed_sources"], len(sources))
        output_bytes = state.get("output_bytes", output_path.stat().st_size)
        with open(output_path, "r+b") as f:
            f.truncate(output_bytes)
        print(f"Resuming: {done}/{len(sources)} sources already translated")

    if done >= len(sources):
        print(f"Already complete: {output_path}")
        return

    device = pick_device()
    tokenizer, model = load_teacher(cfg, device)
    batch_size = teacher_cfg["batch_size"]
    chunk_size = teacher_cfg.get("chunk_size", 4096)
    deadline = start_time + args.time_budget * 3600 if args.time_budget else None
    slowest_chunk, dropped = 0.0, 0

    with open(output_path, "a" if done else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        progress = tqdm(total=len(sources), initial=done, desc="Teacher translating", unit="sent")

        for chunk_start in range(done, len(sources), chunk_size):
            chunk_began = time.monotonic()
            chunk = sources[chunk_start : chunk_start + chunk_size]
            # Sort by length so each batch holds similar lengths (little padding),
            # then put the translations back in the original order.
            order = sorted(range(len(chunk)), key=lambda i: len(chunk[i]))
            translations = [""] * len(chunk)
            for b in range(0, len(order), batch_size):
                index = order[b : b + batch_size]
                outputs = translate_safely([chunk[i] for i in index], tokenizer, model, teacher_cfg, device)
                for i, translation in zip(index, outputs):
                    translations[i] = translation

            for source, translation in zip(chunk, translations):
                translation = translation.replace("\t", " ").replace("\n", " ").strip()
                if translation:
                    writer.writerow([source, translation])
                else:
                    dropped += 1
            f.flush()
            save_json(progress_path, {
                "processed_sources": chunk_start + len(chunk),
                "total_sources": len(sources),
                "output_bytes": f.tell(),
                "teacher": teacher_cfg["model_name"],
                "shard": [args.shard_id, args.num_shards],
                "source_range": [begin, end],
            })
            progress.update(len(chunk))

            slowest_chunk = max(slowest_chunk, time.monotonic() - chunk_began)
            if deadline is not None and time.monotonic() + slowest_chunk > deadline:
                if chunk_start + len(chunk) < len(sources):
                    print(f"\nTime budget reached after {chunk_start + len(chunk)}/{len(sources)} "
                          f"sources; rerun to continue.")
                    break
        progress.close()

    if dropped:
        print(f"Dropped {dropped} empty teacher translations this session")
    print(f"KD training data -> {output_path}")


if __name__ == "__main__":
    main()
