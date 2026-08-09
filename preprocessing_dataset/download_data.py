"""Download CCMatrix (en-id) and FLORES-200 (en-id) into data/raw/ as TSV files.

Usage:
    python -m preprocessing_dataset.download_data [--config configs/config.yaml]
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common.config import load_config, resolve_path


def write_pairs_tsv(path: Path, pairs) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for source, target in pairs:
            writer.writerow([source, target])
            count += 1
    return count


def download_ccmatrix(cfg: dict, out_path: Path) -> int:
    ccmatrix_cfg = cfg["data"]["ccmatrix"]
    source_lang = cfg["data"]["source_lang"]
    target_lang = cfg["data"]["target_lang"]
    max_pairs = ccmatrix_cfg["max_pairs"]

    # Streaming avoids downloading the full 70M-pair corpus. CCMatrix is
    # sorted by LASER margin score, so the first N pairs are the cleanest.
    dataset = load_dataset(
        ccmatrix_cfg["hf_dataset"],
        ccmatrix_cfg["hf_config"],
        split="train",
        streaming=True,
    )

    def pairs():
        for row in tqdm(dataset.take(max_pairs), total=max_pairs, desc="CCMatrix"):
            translation = row["translation"]
            yield translation[source_lang], translation[target_lang]

    return write_pairs_tsv(out_path, pairs())


def download_flores(cfg: dict, split: str, out_path: Path) -> int:
    flores_cfg = cfg["data"]["flores"]
    dataset = load_dataset(
        flores_cfg["hf_dataset"],
        flores_cfg["hf_config"],
        split=split,
        trust_remote_code=True,
    )
    pairs = (
        (row[flores_cfg["source_field"]], row[flores_cfg["target_field"]])
        for row in dataset
    )
    return write_pairs_tsv(out_path, pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg["data"]["raw_dir"])

    n = download_ccmatrix(cfg, raw_dir / "ccmatrix.tsv")
    print(f"CCMatrix: {n} pairs -> {raw_dir / 'ccmatrix.tsv'}")

    for split in ("dev", "devtest"):
        n = download_flores(cfg, split, raw_dir / f"flores_{split}.tsv")
        print(f"FLORES-200 {split}: {n} pairs -> {raw_dir / f'flores_{split}.tsv'}")


if __name__ == "__main__":
    main()
