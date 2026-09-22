"""coremltools-based magnitude pruning (Apple's official optimizer).

Companion to model/experiments/prune_post_vs_train.py. Where model/pruning.py is
a hand-rolled gradual magnitude pruner, this module drives the *same* algorithm
through ``coremltools.optimize.torch`` so the experiment (and, if adopted, the
main training path) uses Apple's supported implementation rather than a custom
one. Both arms of the experiment share this single algorithm; they differ only
in *when* sparsity is introduced:

  * training-time pruning: a PolynomialDecayScheduler ramps sparsity across the
    fine-tuning run so surviving weights adapt to the zeros (Zhu & Gupta, 2017);
  * post-training pruning: a one-shot magnitude cut of the already-trained
    baseline (ConstantSparsityScheduler at step 0) with no weight updates.

Unstructured, per-tensor magnitude pruning of every ``nn.Linear`` (granularity
``per_scalar``); embeddings and recurrent cells stay dense, matching the layer
scope of model/qat.py and model/pruning.py.

``CoreMLMagnitudePruner`` exposes the exact interface model/training/trainer.py
expects from a pruner (``step``, ``current_sparsity``, ``global_step``,
``finalize``, ``restore_from_model``), so it drops into the existing training
loop with no changes to the Trainer. After ``finalize`` the model is a plain
dense model whose Linear weights contain zeros; coremltools turns those into a
sparse CoreML representation at convert time via
``ct.optimize.coreml.prune_weights`` (deploy-time equivalent of the post-hoc
arm), so downstream evaluate/quantize/convert need no changes.
"""

from __future__ import annotations
import torch.nn as nn

from model.pruning import sparsity_report
from coremltools.optimize.torch.pruning import (
    MagnitudePruner as _CTMagnitudePruner,
    MagnitudePrunerConfig as _CTMagnitudePrunerConfig,
)

class CoreMLMagnitudePruner:
    """Wrap ``coremltools.optimize.torch`` MagnitudePruner for the Trainer.

    Args:
        model: the model to prune in place (nn.Linear layers only).
        target_sparsity: final fraction of Linear weights to zero, in [0, 1).
        begin_step / end_step: schedule window in optimizer steps for the
            gradual (training-time) ramp. Ignored when ``one_shot=True``.
        update_frequency: recompute the mask every N steps during the ramp.
        power: exponent of the polynomial (cubic) ramp (Zhu & Gupta default 3).
        one_shot: if True, apply the full target sparsity at step 0 with no ramp
            (post-training pruning of an already-trained model).
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float,
        begin_step: int = 0,
        end_step: int = 1,
        update_frequency: int = 100,
        power: int = 3,
        one_shot: bool = False,
    ):
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        
        if not one_shot and end_step <= begin_step:
            raise ValueError("end_step must be greater than begin_step")

        self.model = model
        self.target_sparsity = target_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.update_frequency = max(1, update_frequency)
        self.power = power
        self.one_shot = one_shot
        self.global_step = 0

        if one_shot:
            # ConstantSparsityScheduler(begin_step=0): reaches target_sparsity at
            # step 0, so a single step() call performs the one-shot cut.
            global_config = {
                "scheduler": {"begin_step": 0},
                "target_sparsity": target_sparsity,
                "granularity": "per_scalar", # per_scalar is unstructured pruning
            }

        else:
            # PolynomialDecayScheduler over the ramp window; the mask is refreshed
            # only at the listed update steps, easing sparsity 0 -> target.
            update_steps = list(range(begin_step, end_step + 1, self.update_frequency))

            if not update_steps or update_steps[-1] != end_step:
                update_steps.append(end_step)

            global_config = {
                "scheduler": {"update_steps": update_steps, "power": power},
                "target_sparsity": target_sparsity,
                "granularity": "per_scalar",
            }

        config = _CTMagnitudePrunerConfig.from_dict({"global_config": global_config})

        # Exclude any nn.Linear layers that are children of nn.MultiheadAttention, since
        # the attention forward path bypasses the Linear modules and never projects their weights into the output.
        self._excluded = self._unpruneable_linear_names(model)

        for name in self._excluded:
            config.set_module_name(name, None)

        self._pruner = _CTMagnitudePruner(model, config)
        self._prepared = False
        self._n_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - len(self._excluded)

    @staticmethod
    def _unpruneable_linear_names(model: nn.Module) -> list[str]:
        """Fully-qualified names of nn.Linear layers that must not be pruned by
        the reparametrization approach: those inside an nn.MultiheadAttention,
        whose forward is bypassed by the functional attention path."""
        names: list[str] = []
        for mod_name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                for child_name, child in module.named_modules():
                    if child_name and isinstance(child, nn.Linear):
                        names.append(f"{mod_name}.{child_name}" if mod_name else child_name)
        return names

    def prepare(self) -> int:
        """Install the pruning reparametrization. Returns the pruned-layer count."""
        self.model = self._pruner.prepare(inplace=True)
        self._prepared = True
        return self._n_layers

    def step(self) -> None:
        """Call once per optimizer step: advance the schedule and re-mask."""
        if not self._prepared:
            self.prepare()
        self.global_step += 1
        self._pruner.step()

    def prune_now(self) -> None:
        """One-shot application without a training loop (post-training arm)."""
        if not self._prepared:
            self.prepare()

        self._pruner.step()
        self.global_step += 1

    def current_sparsity(self) -> float:
        """Overall fraction of zeroed Linear weights (best effort during training).

        Uses the pruner's own report while the reparametrization is live; after
        finalize the zeros are baked into the weights and sparsity_report is exact.
        """
        try:
            report = self._pruner.report()
            vals = []
            for _, entry in report.items():
                for key, value in entry.items():
                    if "sparsity" in key and isinstance(value, (int, float)):
                        vals.append(float(value))
            if vals:
                return sum(vals) / len(vals)
        except Exception:
            pass

        return sparsity_report(self.model)["sparsity"]

    def finalize(self) -> int:
        """Bake the mask into the weights and remove the reparametrization.

        Returns the number of pruned layers. The result is a plain dense model
        whose Linear weights contain zeros, safe for the existing export path.
        """

        if self._prepared:
            self.model = self._pruner.finalize(inplace=True)

        return self._n_layers

    def restore_from_model(self, global_step: int) -> None:
        """Resume support: re-arm the pruner and fast-forward the schedule.

        The zero pattern already lives in the loaded weights; stepping the
        scheduler back to ``global_step`` keeps the ramp position consistent so a
        resumed run continues with the same schedule.
        """
        
        if not self._prepared:
            self.prepare()

        self.global_step = global_step
        for _ in range(global_step):
            self._pruner.step()
