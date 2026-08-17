"""Reusable training loop: teacher-forced cross-entropy, early stopping, checkpointing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.architectures.factory import save_checkpoint


def build_lr_lambda(scheduler_cfg: dict, total_steps: int) -> Callable[[int], float]:
    """LR multiplier as a function of global step: linear warmup then decay.

    Returns a factor relative to the peak (optimizer) LR, so it composes with
    LambdaLR. `total_steps` is the planned number of optimizer steps
    (steps_per_epoch * epochs); early stopping may cut it short, which only
    means the decay tail is not reached.
    """
    sched_type = scheduler_cfg.get("type", "none")
    warmup_steps = max(1, int(scheduler_cfg.get("warmup_ratio", 0.0) * total_steps))
    min_lr_ratio = scheduler_cfg.get("min_lr_ratio", 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if sched_type == "cosine":
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if sched_type == "inverse_sqrt":
            return (warmup_steps ** 0.5) / (max(step, 1) ** 0.5)
        return 1.0  # "none": constant after warmup

    return lr_lambda


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """Record an epoch's validation loss. Returns True if this is a new best."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pad_id: int,
        device: torch.device,
        learning_rate: float,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        epochs: int = 20,
        early_stopping_patience: int = 3,
        run_dir: Path | None = None,
        checkpoint_meta: dict | None = None,
        quiet: bool = False,
        pruner=None,
        scheduler_cfg: dict | None = None,
        epoch_callback: Callable[[int, float], None] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.run_dir = run_dir
        self.checkpoint_meta = checkpoint_meta or {}
        self.quiet = quiet
        # Optional MagnitudePruner (model/pruning.py). When set, it ramps
        # sparsity and re-applies the weight mask after every optimizer step,
        # composing with QAT in the same one-shot run.
        self.pruner = pruner
        # Optional callback run after each epoch's validation with
        # (epoch, val_loss). The hyperparameter search passes one that reports to
        # Optuna and raises optuna.TrialPruned to stop unpromising trials; keeping
        # it a plain callback means the trainer never imports optuna.
        self.epoch_callback = epoch_callback
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id, label_smoothing=label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # LR schedule (linear warmup then decay), stepped every optimizer step.
        # Anchored to the planned run length so warmup/decay are consistent
        # between the search and the full runs.
        self.scheduler = None
        if scheduler_cfg and scheduler_cfg.get("type", "none") != "none":
            total_steps = max(1, len(train_loader) * epochs)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, build_lr_lambda(scheduler_cfg, total_steps)
            )
        self.early_stopping = EarlyStopping(early_stopping_patience)
        self.history: list[dict] = []
        self.start_epoch = 1

    def _batch_loss(self, batch) -> torch.Tensor:
        source, target_input, target_labels = (t.to(self.device) for t in batch)
        logits = self.model(source, target_input)
        return self.criterion(
            logits.reshape(-1, logits.shape[-1]), target_labels.reshape(-1)
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, batches = 0.0, 0
        iterator = self.train_loader
        if not self.quiet:
            iterator = tqdm(iterator, desc=f"Epoch {epoch}", leave=False)
        for batch in iterator:
            self.optimizer.zero_grad()
            loss = self._batch_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.pruner is not None:
                self.pruner.step()
            total_loss += loss.item()
            batches += 1
        return total_loss / max(batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, batches = 0.0, 0
        for batch in self.val_loader:
            total_loss += self._batch_loss(batch).item()
            batches += 1
        return total_loss / max(batches, 1)

    def save_state(self, path: Path) -> None:
        """Full resume state: weights + optimizer + early-stopping progress."""
        torch.save(
            {
                "meta": self.checkpoint_meta,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.history[-1]["epoch"] if self.history else 0,
                "best_loss": self.early_stopping.best_loss,
                "bad_epochs": self.early_stopping.bad_epochs,
                "history": self.history,
                # The sparsity pattern lives in the (zeroed) weights; only the
                # schedule position needs saving so --resume continues the ramp.
                "pruner_step": self.pruner.global_step if self.pruner else None,
            },
            path,
        )

    def load_state(self, path: Path) -> int:
        """Restore a save_state() checkpoint. Returns the epoch to resume from."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.early_stopping.best_loss = state["best_loss"]
        self.early_stopping.bad_epochs = state["bad_epochs"]
        self.history = state["history"]
        self.start_epoch = state["epoch"] + 1
        if self.pruner is not None and state.get("pruner_step") is not None:
            # Rebuild masks from the loaded (already-zeroed) weights and resume
            # the schedule where it left off.
            self.pruner.restore_from_model(state["pruner_step"])
        return self.start_epoch

    def train(self) -> float:
        """Run the full loop. Returns the best validation loss."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self.validate()
            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            if self.pruner is not None:
                record["sparsity"] = round(self.pruner.current_sparsity(), 4)
            self.history.append(record)
            # Report to any observer (e.g. Optuna) before early-stopping logic so
            # a trial can be pruned on this epoch's val_loss. The callback may
            # raise (optuna.TrialPruned) to abort the run.
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, val_loss)
            is_best = self.early_stopping.step(val_loss)
            if not self.quiet:
                marker = " *" if is_best else ""
                sparsity = (
                    f" sparsity={record['sparsity']:.3f}"
                    if self.pruner is not None
                    else ""
                )
                print(
                    f"Epoch {epoch}: train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f}{sparsity}{marker}"
                )
            if self.run_dir is not None:
                if is_best:
                    save_checkpoint(self.run_dir / "best.pt", self.model, self.checkpoint_meta)
                self.save_state(self.run_dir / "last.pt")
            if self.early_stopping.should_stop:
                if not self.quiet:
                    print(f"Early stopping after epoch {epoch}")
                break
        return self.early_stopping.best_loss
