"""Tests for `GlinerMinusModel`."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest
import torch

from glinerminus import GlinerMinusConfig, GlinerMinusModel, LABEL_IGNORE_INDEX


@pytest.fixture(scope="module")
def tiny_config() -> GlinerMinusConfig:
    """Small public encoder for fast CPU tests."""
    return GlinerMinusConfig(
        backbone_model_name_or_path="hf-internal-testing/tiny-random-bert",
        num_bilinear_heads=3,
    )


@pytest.fixture
def model(tiny_config: GlinerMinusConfig) -> GlinerMinusModel:
    """Fresh model per test (avoids leftover gradients)."""
    return GlinerMinusModel(tiny_config)


def test_forward_shapes_and_finite(model: GlinerMinusModel) -> None:
    """Bilinear logits match padded entity/span layout."""
    batch_size = 2
    seq_len = 16
    hidden = model.config.hidden_size
    assert hidden is not None
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    entity_index = [[1], [2, 3]]
    spans: list[list[tuple[int, int]]] = [
        [(4, 5), (6, 7)],
        [(4, 4)],
    ]
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_index=entity_index,
        spans=spans,
        return_dict=True,
    )
    assert out.logits is not None
    assert out.logits.shape == (batch_size, 2, 2, 3)
    assert torch.isfinite(out.logits).all()
    assert out.entity_mask is not None
    assert out.span_mask is not None
    assert out.entity_mask.shape == (batch_size, 2)
    assert out.span_mask.shape == (batch_size, 2)
    assert out.loss is None


def test_backward_reaches_head_and_backbone(model: GlinerMinusModel) -> None:
    """Gradients flow to bilinear weights and backbone parameters."""
    batch_size = 1
    seq_len = 12
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    entity_index = [[1]]
    spans: list[list[tuple[int, int]]] = [[(4, 6)]]
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_index=entity_index,
        spans=spans,
        return_dict=True,
    )
    assert out.logits is not None
    loss = out.logits.sum()
    loss.backward()
    assert model.bilinear_weight.grad is not None
    assert model.bilinear_weight.grad.abs().sum() > 0
    backbone_params = list(model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone_params)


def test_mse_loss_with_ignore_index(model: GlinerMinusModel) -> None:
    """Loss ignores positions with LABEL_IGNORE_INDEX."""
    batch_size = 1
    seq_len = 12
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    entity_index = [[1]]
    spans: list[list[tuple[int, int]]] = [[(4, 6)]]
    labels = torch.full((batch_size, 1, 1, 3), LABEL_IGNORE_INDEX, dtype=torch.long)
    labels[0, 0, 0, 0] = 0
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_index=entity_index,
        spans=spans,
        labels=labels,
        return_dict=True,
    )
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)


def test_tokenize_method(model: GlinerMinusModel) -> None:
    """Model.tokenize produces BatchEncoding with entity_index and spans."""
    enc = model.tokenize(
        entities=[["person"], ["org"]],
        texts=["John is a person", "Google is an org"],
        spans=[[(0, 4)], [(0, 6)]],
        padding=True,
        truncation=True,
    )
    assert "input_ids" in enc
    assert "attention_mask" in enc
    assert "entity_index" in enc
    assert "spans" in enc
    assert len(enc["entity_index"]) == 2
    assert len(enc["spans"]) == 2


def test_save_load_includes_tokenizer(tiny_config: GlinerMinusConfig) -> None:
    """Saved model directory contains tokenizer; reload preserves tokenizer and outputs."""
    model = GlinerMinusModel(tiny_config)
    with TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded = GlinerMinusModel.from_pretrained(d)
    assert loaded.tokenizer is not None
    assert len(loaded.tokenizer) == len(model.tokenizer)
    enc = model.tokenize([["a"]], ["hello"], [[(0, 4)]])
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        entity_index=enc["entity_index"],
        spans=enc["spans"],
        return_dict=True,
    )
    loaded_out = loaded(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        entity_index=enc["entity_index"],
        spans=enc["spans"],
        return_dict=True,
    )
    assert out.logits is not None and loaded_out.logits is not None
    assert torch.allclose(out.logits, loaded_out.logits)
