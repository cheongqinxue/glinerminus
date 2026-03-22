"""Tests for pruning distillation trainer."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from glinerminus import (
    GlinerMinusConfig,
    GlinerMinusDataCollator,
    GlinerMinusModel,
    PruningDistillationTrainer,
    PruningTrainingArguments,
)


class DummyGlinerDataset(Dataset):
    """Minimal dataset for GlinerMinus with entity_index and spans."""

    def __init__(self, n: int = 8, seq_len: int = 16) -> None:
        self.n = n
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.ones(self.seq_len, dtype=torch.long),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "entity_index": [[1]],
            "spans": [(4, 6)],
            "labels": torch.zeros(1, 1, 3),
        }


def test_pruning_trainer_instantiates() -> None:
    """Trainer instantiates without teacher."""
    config = GlinerMinusConfig(
        backbone_model_name_or_path="hf-internal-testing/tiny-random-bert",
        num_bilinear_heads=2,
    )
    model = GlinerMinusModel(config)
    args = PruningTrainingArguments(
        output_dir="/tmp/test_prune",
        max_steps=2,
        target_num_layers=2,
        per_device_train_batch_size=2,
    )
    ds = DummyGlinerDataset(n=4)
    trainer = PruningDistillationTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=GlinerMinusDataCollator(),
        pruning_dataloader=None,
    )
    assert trainer.teacher_model is None
    assert trainer.pruning_args.target_num_layers == 2


def test_pruning_trainer_with_teacher_loads_from_path() -> None:
    """Trainer loads teacher from path when provided."""
    config = GlinerMinusConfig(
        backbone_model_name_or_path="hf-internal-testing/tiny-random-bert",
        num_bilinear_heads=2,
    )
    model = GlinerMinusModel(config)
    args = PruningTrainingArguments(
        output_dir="/tmp/test_prune",
        max_steps=1,
        target_num_layers=2,
        teacher_model_name_or_path="",
        per_device_train_batch_size=2,
    )
    teacher = GlinerMinusModel(config)
    ds = DummyGlinerDataset(n=4)
    trainer = PruningDistillationTrainer(
        model=model,
        args=args,
        teacher_model=teacher,
        train_dataset=ds,
        data_collator=GlinerMinusDataCollator(),
    )
    assert trainer.teacher_model is teacher
