"""Convert a trained checkpoint to CoreML .mlpackage files (encoder + decoder).

QAT checkpoints must be folded first (quantization_coreml/quantize.py) so the
graph contains no fake-quantize ops. Alongside the two .mlpackage files, the
SentencePiece model and a meta.json (special token ids, shapes) are copied so
the iOS app has everything it needs.

Usage:
    python -m quantization_coreml.convert_coreml \
        --checkpoint results/runs/<run>/quantized.pt --name transformer_kd_int8
"""

from __future__ import annotations

import argparse
import shutil

import coremltools as ct
import numpy as np
import torch

from common.config import load_config, resolve_path
from common.results import RESULTS_DIR, save_json
from model.architectures.factory import load_checkpoint
from quantization_coreml.export_wrappers import build_wrappers


def _length_shapes(max_length: int, make_shape) -> ct.EnumeratedShapes:
    """Enumerate every source length 1..max_length (CoreML caps this at 128).

    RangeDim hangs coremltools on these recurrent graphs, and enumerating all
    lengths lets the app feed the exact tokenized length — no padding, so
    outputs stay equivalent to PyTorch.
    """
    return ct.EnumeratedShapes(shapes=[make_shape(n) for n in range(1, max_length + 1)])


def _convert(traced, inputs, output_names, deployment_target):
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=name) for name in output_names],
        minimum_deployment_target=deployment_target,
        convert_to="mlprogram",
    )


def convert_encoder(wrapper, arch, coreml_cfg, deployment_target):
    max_source = coreml_cfg["max_source_length"]
    example_source = torch.randint(4, 100, (1, 12), dtype=torch.int64)
    traced = torch.jit.trace(wrapper, example_source, check_trace=False)

    source_shape = _length_shapes(max_source, lambda n: (1, n))
    inputs = [ct.TensorType(name="source", shape=source_shape, dtype=np.int32)]
    if arch == "lstm":
        output_names = ["encoder_states", "hidden", "cell"]
    elif arch == "gru":
        output_names = ["encoder_states", "hidden"]
    else:
        output_names = ["memory"]
    return _convert(traced, inputs, output_names, deployment_target)


def convert_decoder(wrapper, arch, meta, coreml_cfg, deployment_target):
    hparams = meta["hparams"]
    max_source = coreml_cfg["max_source_length"]
    max_target = coreml_cfg["max_target_length"]

    if arch in ("gru", "lstm"):
        hidden_dim = hparams["hidden_dim"]
        num_layers = hparams["num_layers"]
        example_token = torch.tensor([[meta["bos_id"]]], dtype=torch.int64)
        example_encoder_states = torch.randn(1, 12, 2 * hidden_dim)
        example_hidden = torch.randn(num_layers, 1, hidden_dim)
        state_shape = (num_layers, 1, hidden_dim)

        example_inputs = [example_token, example_encoder_states, example_hidden]
        inputs = [
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(
                name="encoder_states",
                shape=_length_shapes(max_source, lambda n: (1, n, 2 * hidden_dim)),
            ),
            ct.TensorType(name="hidden", shape=state_shape),
        ]
        output_names = ["logits", "new_hidden"]
        if arch == "lstm":
            example_inputs.append(torch.randn(*state_shape))
            inputs.append(ct.TensorType(name="cell", shape=state_shape))
            output_names.append("new_cell")

        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
        return _convert(traced, inputs, output_names, deployment_target)

    d_model = hparams["d_model"]
    example_prefix = torch.full((1, max_target), meta["pad_id"], dtype=torch.int64)
    example_prefix[0, 0] = meta["bos_id"]
    example_memory = torch.randn(1, 12, d_model)
    traced = torch.jit.trace(wrapper, [example_prefix, example_memory], check_trace=False)

    inputs = [
        ct.TensorType(name="prefix", shape=(1, max_target), dtype=np.int32),
        ct.TensorType(
            name="memory", shape=_length_shapes(max_source, lambda n: (1, n, d_model))
        ),
    ]
    return _convert(traced, inputs, ["logits"], deployment_target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="Output folder name under results/coreml/")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    coreml_cfg = cfg["coreml"]
    deployment_target = getattr(ct.target, coreml_cfg["minimum_deployment_target"])

    checkpoint_path = resolve_path(args.checkpoint)
    model, meta = load_checkpoint(checkpoint_path, torch.device("cpu"))
    if meta.get("qat", False):
        raise SystemExit(
            "Checkpoint still contains QAT wrappers. "
            "Run quantization_coreml.quantize first and convert quantized.pt."
        )

    arch = meta["arch"]
    encoder_wrapper, decoder_wrapper = build_wrappers(
        model, arch, coreml_cfg["max_target_length"]
    )

    out_dir = RESULTS_DIR / "coreml" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Converting encoder...")
    encoder_ml = convert_encoder(encoder_wrapper, arch, coreml_cfg, deployment_target)
    encoder_ml.save(str(out_dir / "Encoder.mlpackage"))

    print("Converting decoder...")
    decoder_ml = convert_decoder(decoder_wrapper, arch, meta, coreml_cfg, deployment_target)
    decoder_ml.save(str(out_dir / "Decoder.mlpackage"))

    tokenizer_model = resolve_path(cfg["tokenizer"]["model_prefix"]).with_suffix(".model")
    shutil.copy(tokenizer_model, out_dir / "tokenizer.model")
    save_json(
        out_dir / "meta.json",
        {
            **{k: meta[k] for k in ("arch", "hparams", "vocab_size", "pad_id", "bos_id", "eos_id")},
            "max_source_length": coreml_cfg["max_source_length"],
            "max_target_length": coreml_cfg["max_target_length"],
            "source_checkpoint": str(checkpoint_path),
        },
    )
    print(f"Saved CoreML bundle -> {out_dir}")


if __name__ == "__main__":
    main()
