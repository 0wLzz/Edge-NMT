"""Cross-prediction check: PyTorch checkpoint vs converted CoreML bundle.

Greedy-decodes the same FLORES dev sentences with both runtimes and reports
the exact-match rate (methodology: converted model must stay consistent with
the PyTorch original). Requires macOS for the CoreML runtime.

Usage:
    python -m quantization_coreml.verify \
        --checkpoint results/runs/<run>/quantized.pt \
        --coreml-dir results/coreml/<name> --sentences 20
"""

from __future__ import annotations

import argparse

import coremltools as ct
import numpy as np
import torch

from common.config import load_config, resolve_path
from common.results import load_json, save_json
from model.architectures.factory import load_checkpoint
from model.training.dataset import load_tokenizer, read_pairs


def coreml_greedy_decode(encoder, decoder, meta, source_ids: list[int]) -> list[int]:
    arch = meta["arch"]
    bos, eos = meta["bos_id"], meta["eos_id"]
    max_target = meta["max_target_length"]
    source = np.array([source_ids], dtype=np.int32)

    if arch in ("gru", "lstm"):
        state = encoder.predict({"source": source})
        inputs = {
            "encoder_states": state["encoder_states"],
            "hidden": state["hidden"],
        }
        if arch == "lstm":
            inputs["cell"] = state["cell"]
        tokens, token = [], bos
        for _ in range(max_target):
            outputs = decoder.predict(
                {"token": np.array([[token]], dtype=np.int32), **inputs}
            )
            token = int(outputs["logits"].argmax())
            if token == eos:
                break
            tokens.append(token)
            inputs["hidden"] = outputs["new_hidden"]
            if arch == "lstm":
                inputs["cell"] = outputs["new_cell"]
        return tokens

    memory = encoder.predict({"source": source})["memory"]
    prefix = np.full((1, max_target), meta["pad_id"], dtype=np.int32)
    prefix[0, 0] = bos
    tokens = []
    for position in range(max_target - 1):
        logits = decoder.predict({"prefix": prefix, "memory": memory})["logits"]
        token = int(logits[0, position].argmax())
        if token == eos:
            break
        tokens.append(token)
        prefix[0, position + 1] = token
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coreml-dir", required=True)
    parser.add_argument("--sentences", type=int, default=20)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg)
    model, checkpoint_meta = load_checkpoint(
        resolve_path(args.checkpoint), torch.device("cpu")
    )

    coreml_dir = resolve_path(args.coreml_dir)
    bundle_meta = load_json(coreml_dir / "meta.json")
    encoder = ct.models.MLModel(str(coreml_dir / "Encoder.mlpackage"))
    decoder = ct.models.MLModel(str(coreml_dir / "Decoder.mlpackage"))

    flores_dev = resolve_path(cfg["data"]["raw_dir"]) / "flores_dev.tsv"
    sources = [source for source, _ in read_pairs(flores_dev)[: args.sentences]]

    max_target = bundle_meta["max_target_length"]
    matches, samples = 0, []
    for text in sources:
        source_ids = tokenizer.encode(text)[: bundle_meta["max_source_length"] - 1]
        source_ids.append(bundle_meta["eos_id"])

        torch_tokens = model.greedy_decode(
            torch.tensor([source_ids]),
            bundle_meta["bos_id"],
            bundle_meta["eos_id"],
            max_target,
        )[0]
        coreml_tokens = coreml_greedy_decode(encoder, decoder, bundle_meta, source_ids)

        matched = torch_tokens == coreml_tokens
        matches += matched
        samples.append(
            {
                "source": text,
                "pytorch": tokenizer.decode(torch_tokens),
                "coreml": tokenizer.decode(coreml_tokens),
                "match": matched,
            }
        )

    match_rate = matches / len(sources)
    report = {"sentences": len(sources), "exact_match_rate": match_rate, "samples": samples}
    save_json(coreml_dir / "verification.json", report)

    print(f"Exact match: {matches}/{len(sources)} ({match_rate:.0%})")
    for sample in samples[:3]:
        print(f"  [{'OK' if sample['match'] else 'DIFF'}] {sample['source'][:60]}")
        print(f"      pytorch: {sample['pytorch'][:80]}")
        print(f"      coreml : {sample['coreml'][:80]}")
    print(f"Full report: {coreml_dir / 'verification.json'}")


if __name__ == "__main__":
    main()
