"""Quantization-Aware Training: asymmetric per-tensor uint8 fake quantization.

QAT is applied to every nn.Linear (weights + input activations). Embeddings and
recurrent cells stay float; Linear layers hold the bulk of the parameters in
all three student architectures.

Toggle at build time with the --qat flag, or at runtime with set_qat_enabled().
After training, fold_qat() bakes the weight quantization error into the
weights and swaps the wrappers back to plain nn.Linear so the model can be
traced and converted to CoreML without custom ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANT_MIN = 0
QUANT_MAX = 255  # uint8


class FakeQuantize(nn.Module):
    """Asymmetric per-tensor fake quantization with an EMA min/max observer.

    Training mode: observe the tensor's range, then quantize-dequantize.
    Eval mode: quantize-dequantize with the frozen observed range.
    """

    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        self.enabled = True
        self.register_buffer("min_val", torch.tensor(0.0))
        self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def _observe(self, x: torch.Tensor) -> None:
        batch_min, batch_max = x.detach().min(), x.detach().max()
        if not bool(self.initialized):
            self.min_val.copy_(batch_min)
            self.max_val.copy_(batch_max)
            self.initialized.fill_(True)
        else:
            self.min_val.mul_(self.momentum).add_(batch_min * (1 - self.momentum))
            self.max_val.mul_(self.momentum).add_(batch_max * (1 - self.momentum))

    def quantization_params(self) -> tuple[float, int]:
        # Keep zero exactly representable (required for padding/zero activations).
        min_val = float(torch.minimum(self.min_val, torch.tensor(0.0)))
        max_val = float(torch.maximum(self.max_val, torch.tensor(0.0)))
        scale = max((max_val - min_val) / (QUANT_MAX - QUANT_MIN), 1e-8)
        zero_point = int(round(QUANT_MIN - min_val / scale))
        zero_point = max(QUANT_MIN, min(QUANT_MAX, zero_point))
        return scale, zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.training:
            self._observe(x)
        if not bool(self.initialized):
            return x
        scale, zero_point = self.quantization_params()
        return torch.fake_quantize_per_tensor_affine(
            x, scale, zero_point, QUANT_MIN, QUANT_MAX
        )


class QATLinear(nn.Module):
    """nn.Linear with fake quantization on weights and input activations."""

    def __init__(self, linear: nn.Linear, momentum: float = 0.99):
        super().__init__()
        self.linear = linear
        self.weight_quantizer = FakeQuantize(momentum)
        self.activation_quantizer = FakeQuantize(momentum)

    # nn.TransformerEncoderLayer inspects linear1.weight/bias for its fused
    # fast path; delegate so those checks keep working.
    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_quantizer(self.linear.weight)
        x = self.activation_quantizer(x)
        return F.linear(x, weight, self.linear.bias)


def apply_qat(model: nn.Module, momentum: float = 0.99) -> int:
    """Replace every nn.Linear in the model with a QATLinear. Returns the count."""
    # The fused transformer fast path reads weights directly and would skip
    # fake quantization during eval; force the regular (quantized) code path.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:
        pass

    replaced = 0

    def _wrap(module: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                # MultiheadAttention reads out_proj.weight directly (bypassing
                # forward), so its projections cannot be fake-quantized here.
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, QATLinear(child, momentum))
                replaced += 1
            elif not isinstance(child, QATLinear):  # never descend into wrappers
                _wrap(child)

    _wrap(model)
    return replaced


def set_qat_enabled(model: nn.Module, enabled: bool) -> None:
    """Turn fake quantization on/off without removing the wrappers."""
    for module in model.modules():
        if isinstance(module, FakeQuantize):
            module.enabled = enabled


def fold_qat(model: nn.Module) -> int:
    """Bake weight quantization into the weights and restore plain nn.Linear.

    The folded weights are exactly representable on the uint8 grid, so the
    float model behaves like the quantized one (activation quantization noise
    was already absorbed during training). Returns the number of folded layers.
    """
    folded = 0

    def _fold(module: nn.Module) -> None:
        nonlocal folded
        for name, child in list(module.named_children()):
            if isinstance(child, QATLinear):
                with torch.no_grad():
                    child.weight_quantizer.eval()
                    child.linear.weight.copy_(
                        child.weight_quantizer(child.linear.weight.data)
                    )
                setattr(module, name, child.linear)
                folded += 1
            else:
                _fold(child)

    _fold(model)
    return folded


def int8_size_report(model: nn.Module) -> dict:
    """Estimate on-disk sizes: fp32 vs uint8 weights for quantized Linear layers."""
    quantized_params = 0
    float_params = 0

    def _count(module: nn.Module) -> None:
        nonlocal quantized_params, float_params
        for child in module.children():
            if isinstance(child, QATLinear):
                quantized_params += child.linear.weight.numel()
                if child.linear.bias is not None:
                    float_params += child.linear.bias.numel()
            else:
                float_params += sum(
                    p.numel() for p in child.parameters(recurse=False)
                )
                _count(child)

    float_params += sum(p.numel() for p in model.parameters(recurse=False))
    _count(model)
    fp32_mb = (quantized_params + float_params) * 4 / 1e6
    int8_mb = (quantized_params * 1 + float_params * 4) / 1e6
    return {
        "quantized_linear_params": quantized_params,
        "float_params": float_params,
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
    }
