"""Gradual magnitude pruning: unstructured, per-tensor, applied during training.

Companion to model/qat.py. Where QAT fake-quantizes the Linear weights, this
module *sparsifies* them: it zeroes the lowest-magnitude weights of every
nn.Linear following a polynomial sparsity schedule (Zhu & Gupta, 2017,
"To prune, or not to prune"), so the network gradually adapts to the zeros
during fine-tuning instead of losing accuracy from a one-shot cut.

Design mirrors qat.py deliberately (custom, not coremltools.optimize) so the
two compose cleanly in one training run and both survive the existing CoreML
export path:

  * Targets the same layers as QAT: every nn.Linear (embeddings and recurrent
    cells stay dense). When QAT is on, the real nn.Linear lives inside a
    QATLinear wrapper as `.linear`; module traversal finds it either way, so
    pruning + QAT stack with no special-casing.
  * "Hard masking": the pruned positions are zeroed in-place after every
    optimizer step, so the forward pass always sees the sparse weights and the
    surviving weights keep training. Gradients still flow to all weights, but
    pruned positions are re-zeroed each step. With a monotonically increasing
    sparsity schedule this is equivalent to standard gradual pruning.
  * No buffers are registered on the model, so state_dict keys are unchanged
    and a pruned checkpoint is just a normal dense checkpoint whose Linear
    weights happen to contain zeros. Downstream (evaluate / quantize /
    convert) needs no changes; coremltools can later turn the zeroed dense
    weights into a sparse CoreML representation via
    ct.optimize.coreml.prune_weights if desired.

Typical use (see model/training/train.py --prune):

    pruner = MagnitudePruner(model, target_sparsity=0.5,
                             begin_step=steps_per_epoch,        # 1-epoch warmup
                             end_step=int(0.7 * total_steps),
                             update_frequency=100)
    pruner.prepare()
    for step in training:
        ...
        optimizer.step()
        pruner.step()          # ramps sparsity, re-applies the mask
    pruner.finalize()          # commit the zeros (no-op safety)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _polynomial_sparsity(
    step: int, begin_step: int, end_step: int, final_sparsity: float, power: int = 3
) -> float:
    """Sparsity at a given step on the Zhu & Gupta cubic ramp (0 -> final).

    Flat 0 before begin_step, flat `final_sparsity` after end_step, and a cubic
    ease-in between so most zeroing happens early while there is still time to
    recover accuracy.
    """
    if step <= begin_step:
        return 0.0
    if step >= end_step:
        return final_sparsity
    progress = (step - begin_step) / max(end_step - begin_step, 1)
    return final_sparsity * (1.0 - (1.0 - progress) ** power)


def _iter_linear(model: nn.Module):
    """Yield every nn.Linear in the model (including those wrapped by QATLinear)."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module


class MagnitudePruner:
    """Gradual, unstructured, per-tensor magnitude pruner for nn.Linear weights."""

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float,
        begin_step: int,
        end_step: int,
        update_frequency: int = 100,
        power: int = 3,
    ):
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        if end_step <= begin_step:
            raise ValueError("end_step must be greater than begin_step")
        self.model = model
        self.target_sparsity = target_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.update_frequency = max(1, update_frequency)
        self.power = power
        self.global_step = 0
        self._layers: list[nn.Linear] = []
        # mask per layer, created lazily on the first pruning event (so it lands
        # on the same device as the weight, after model.to(device)).
        self._masks: list[torch.Tensor | None] = []

    def prepare(self) -> int:
        """Collect the layers to prune. Returns the count."""
        self._layers = list(_iter_linear(self.model))
        self._masks = [None] * len(self._layers)
        return len(self._layers)

    @staticmethod
    def _mask_for(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Per-tensor magnitude mask keeping the (1 - sparsity) largest weights."""
        if sparsity <= 0.0:
            return torch.ones_like(weight)
        numel = weight.numel()
        k = int(round(sparsity * numel))
        if k <= 0:
            return torch.ones_like(weight)
        if k >= numel:
            return torch.zeros_like(weight)
        flat_abs = weight.detach().abs().reshape(-1)
        # kth smallest magnitude is the threshold; keep strictly-larger weights.
        threshold = torch.kthvalue(flat_abs, k).values
        return (weight.detach().abs() > threshold).to(weight.dtype)

    def _recompute_masks(self, sparsity: float) -> None:
        for i, layer in enumerate(self._layers):
            self._masks[i] = self._mask_for(layer.weight, sparsity)

    def _apply_masks(self) -> None:
        with torch.no_grad():
            for layer, mask in zip(self._layers, self._masks):
                if mask is not None:
                    layer.weight.data.mul_(mask)

    def step(self) -> None:
        """Call once per optimizer step. Ramps sparsity and re-applies the mask."""
        self.global_step += 1
        if self.global_step < self.begin_step:
            return  # dense warm-up phase
        if (self.global_step - self.begin_step) % self.update_frequency == 0 or (
            self.global_step >= self.end_step and self._masks[0] is None
        ):
            sparsity = _polynomial_sparsity(
                self.global_step,
                self.begin_step,
                self.end_step,
                self.target_sparsity,
                self.power,
            )
            self._recompute_masks(sparsity)
        self._apply_masks()

    def current_sparsity(self) -> float:
        """Overall fraction of zeroed weights across the pruned layers."""
        zeros, total = 0, 0
        for layer, mask in zip(self._layers, self._masks):
            total += layer.weight.numel()
            if mask is not None:
                zeros += int((mask == 0).sum().item())
        return zeros / max(total, 1)

    def restore_from_model(self, global_step: int) -> None:
        """Rebuild masks from the current (already-zeroed) weights, for --resume.

        The exact sparsity pattern is recoverable from which weights are zero,
        so a resumed run continues with the same mask and schedule position.
        """
        self.global_step = global_step
        self._masks = [
            (layer.weight.detach() != 0).to(layer.weight.dtype)
            if global_step >= self.begin_step
            else None
            for layer in self._layers
        ]

    def finalize(self) -> int:
        """Commit the mask permanently. Weights are already zeroed; this is a
        safety re-application. Returns the number of pruned layers."""
        self._apply_masks()
        return len(self._layers)


def sparsity_report(model: nn.Module) -> dict:
    """Per-model sparsity of nn.Linear weights, for the thesis tables."""
    zeros, total = 0, 0
    for layer in _iter_linear(model):
        w = layer.weight.detach()
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return {
        "linear_weight_params": total,
        "zeroed_params": zeros,
        "sparsity": round(zeros / max(total, 1), 4),
    }
