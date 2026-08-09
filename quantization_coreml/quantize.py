"""Finalize a QAT checkpoint: fold fake quantization into the weights.

The folded model is a plain float model whose Linear weights sit exactly on
the uint8 quantization grid, so it can be traced and converted to CoreML with
no custom ops while behaving like the quantized model. Also prints the
fp32-vs-int8 size report for the thesis tables.

Usage:
    python -m quantization_coreml.quantize --checkpoint results/runs/<run>/best.pt
"""

from __future__ import annotations

import argparse

import torch

from common.config import load_config, resolve_path
from common.results import save_json
from model.architectures.factory import load_checkpoint, save_checkpoint
from model.qat import fold_qat, int8_size_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="A checkpoint trained with --qat")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    load_config(args.config)  # validates the config exists; paths come from the checkpoint
    checkpoint_path = resolve_path(args.checkpoint)
    model, meta = load_checkpoint(checkpoint_path, torch.device("cpu"))

    if not meta.get("qat", False):
        raise SystemExit(
            "This checkpoint was not trained with QAT (--qat); nothing to fold."
        )

    size_report = int8_size_report(model)
    folded = fold_qat(model)
    print(f"Folded {folded} QATLinear layers back to nn.Linear")

    meta = {**meta, "qat": False, "quantized_folded": True}
    output_path = checkpoint_path.with_name("quantized.pt")
    save_checkpoint(output_path, model, meta)
    save_json(checkpoint_path.parent / "quantization_report.json", size_report)

    print(f"Quantized checkpoint -> {output_path}")
    print(
        f"Size report: fp32={size_report['fp32_size_mb']}MB, "
        f"int8 weights={size_report['int8_size_mb']}MB "
        f"({size_report['quantized_linear_params']} quantized params)"
    )


if __name__ == "__main__":
    main()
