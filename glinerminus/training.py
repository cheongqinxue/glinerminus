"""Training utilities for iterative pruning with knowledge distillation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from glinerminus.layer_pruning import prune_backbone_layer, run_layer_analysis
from glinerminus.modeling_gliner_minus import GlinerMinusModel

logger = logging.getLogger(__name__)


class GlinerMinusDataCollator:
    """Data collator for GlinerMinus that batches entity_index and spans as lists."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate batch, stacking tensors and preserving entity_index/spans lists."""
        batch: dict[str, Any] = {}
        first = features[0]
        for key in first:
            if key in ("entity_index", "spans"):
                batch[key] = [f[key] for f in features]
            elif isinstance(first[key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = [f[key] for f in features]
        return batch


@dataclass
class PruningTrainingArguments(TrainingArguments):
    """Training arguments extended for iterative layer pruning and distillation.

    Use with `PruningDistillationTrainer` to finetune a model while periodically
    removing redundant layers, guided by a teacher model's logits.
    """

    # Pruning
    target_num_layers: int = field(
        default=6,
        metadata={"help": "Target number of encoder layers after iterative pruning."},
    )
    pruning_epochs_per_phase: float = field(
        default=1.0,
        metadata={"help": "Epochs of training per pruning phase before removing one layer."},
    )
    pruning_steps_per_phase: int = field(
        default=-1,
        metadata={"help": "Steps per pruning phase (-1 to use pruning_epochs_per_phase)."},
    )

    # Distillation
    teacher_model_name_or_path: str = field(
        default="",
        metadata={"help": "Path or HF hub id of teacher model for distillation."},
    )
    distillation_temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for soft logits in distillation."},
    )
    distillation_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for task loss; (1-alpha) for distillation loss."},
    )


class PruningDistillationTrainer(Trainer):
    """Trainer for iterative pruning with knowledge distillation.

    Workflow:
    1. Finetune student with teacher logits (distillation loss + task loss)
    2. Run layer analysis to identify most redundant layer
    3. Prune that layer from the student
    4. Repeat until target_num_layers reached
    """

    def __init__(
        self,
        model: GlinerMinusModel,
        args: PruningTrainingArguments,
        teacher_model: GlinerMinusModel | None = None,
        pruning_dataloader: DataLoader | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, args=args, **kwargs)
        self.pruning_args = args
        self.pruning_dataloader = pruning_dataloader

        if teacher_model is not None:
            self.teacher_model = teacher_model
        elif args.teacher_model_name_or_path:
            self.teacher_model = GlinerMinusModel.from_pretrained(args.teacher_model_name_or_path)
        else:
            self.teacher_model = None

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
            # Teacher device is set in train() when accelerator is ready

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute combined task loss and distillation loss."""
        outputs = model(**inputs)
        task_loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
        student_logits = outputs.logits if hasattr(outputs, "logits") else None

        if task_loss is None:
            task_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        if self.teacher_model is not None and student_logits is not None:
            with torch.no_grad():
                teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
                teacher_outputs = self.teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits
            distill_loss = F.mse_loss(student_logits.float(), teacher_logits.float())
            alpha = self.pruning_args.distillation_alpha
            loss = alpha * task_loss + (1 - alpha) * distill_loss
        else:
            loss = task_loss

        return (loss, outputs) if return_outputs else loss

    def train(self, **kwargs: Any) -> Any:
        """Run iterative pruning training: finetune → prune → repeat."""
        args = self.pruning_args
        target_layers = args.target_num_layers
        original_max_steps = self.args.max_steps
        original_num_epochs = self.args.num_train_epochs

        # Compute steps per phase
        steps_per_phase = args.pruning_steps_per_phase
        if steps_per_phase <= 0:
            train_dl = self.get_train_dataloader()
            steps_per_epoch = len(train_dl) // (args.gradient_accumulation_steps or 1)
            if self.args.world_size and self.args.world_size > 1:
                steps_per_epoch = steps_per_epoch // self.args.world_size
            steps_per_phase = max(1, int(args.pruning_epochs_per_phase * steps_per_epoch))

        # Move teacher to same device as model
        if self.teacher_model is not None:
            device = next(self.model.parameters()).device
            self.teacher_model = self.teacher_model.to(device)

        phase = 0
        while True:
            encoder = getattr(self.model, "backbone", self.model)
            num_layers = getattr(
                getattr(encoder, "config", None),
                "num_hidden_layers",
                None,
            )
            if num_layers is None or num_layers <= target_layers:
                break

            phase += 1
            logger.info(
                "Phase %d: training with %d layers (target: %d) for %d steps.",
                phase,
                num_layers,
                target_layers,
                steps_per_phase,
            )
            self.args.max_steps = steps_per_phase
            self.args.num_train_epochs = -1.0  # Use max_steps
            super().train(**kwargs)

            if num_layers <= target_layers:
                break

            layer_to_prune = self._identify_and_prune_layer()
            if layer_to_prune is None:
                logger.warning("Could not identify layer to prune; stopping.")
                break

            # Recreate optimizer for pruned model
            self.optimizer = None
            self.lr_scheduler = None

        self.args.max_steps = original_max_steps
        self.args.num_train_epochs = original_num_epochs
        return None

    def _identify_and_prune_layer(self) -> int | None:
        """Run layer analysis, prune the most redundant layer, replace model backbone."""
        args = self.pruning_args
        if self.pruning_dataloader is None:
            logger.warning("No pruning_dataloader; cannot identify prunable layer.")
            return None

        device = self.args.device or next(self.model.parameters()).device
        distances, prunable = run_layer_analysis(
            self.model,
            self.pruning_dataloader,
            device=str(device),
            layers_to_skip=1,
        )
        if not prunable:
            return None
        layer_to_prune = prunable[0]
        logger.info("Pruning layer %d (1-based).", layer_to_prune)

        old_backbone = self.model.backbone
        new_backbone = prune_backbone_layer(old_backbone, layer_to_prune)
        new_backbone.to(device)
        self.model.backbone = new_backbone

        # Recreate optimizer/scheduler for new model
        self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
        return layer_to_prune
