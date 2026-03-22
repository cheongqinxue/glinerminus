"""Tests for layer pruning utilities."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from glinerminus import GlinerMinusConfig, GlinerMinusModel
from glinerminus.layer_pruning import (
    angular_distance,
    compute_block_distances,
    compute_layer_distances,
    get_pooled_hidden_states,
    identify_prunable_layers,
    prune_backbone_layer,
    run_layer_analysis,
)


class TestAngularDistance:
    """Tests for angular_distance."""

    def test_identical_vectors_zero_distance(self) -> None:
        x = torch.randn(4, 32)
        dist = angular_distance(x, x)
        assert dist.shape == (4,)
        assert (dist <= 1e-3).all(), f"Expected near-zero, got {dist}"

    def test_orthogonal_vectors(self) -> None:
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0]])
        dist = angular_distance(x, y)
        assert torch.allclose(dist, torch.tensor([0.5]), atol=1e-5)  # 90° -> 0.5

    def test_opposite_vectors(self) -> None:
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[-1.0, 0.0, 0.0]])
        dist = angular_distance(x, y)
        assert torch.allclose(dist, torch.tensor([1.0]), atol=1e-5)  # 180° -> 1.0


class TestGetPooledHiddenStates:
    """Tests for get_pooled_hidden_states."""

    @pytest.fixture
    def hidden_states_and_mask(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        batch, seq, hidden = 2, 5, 8
        emb = torch.randn(batch, seq, hidden)
        layer1 = torch.randn(batch, seq, hidden)
        layer2 = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)
        mask[1, 3:] = False  # Second sequence has 3 valid tokens
        return (emb, layer1, layer2), mask

    def test_mean_pooling_shape(self, hidden_states_and_mask) -> None:
        (hidden_states, mask) = hidden_states_and_mask
        pooled = get_pooled_hidden_states(hidden_states, mask, pool_strategy="mean")
        assert len(pooled) == 2
        assert pooled[0].shape == (2, 8)
        assert pooled[1].shape == (2, 8)

    def test_last_pooling_shape(self, hidden_states_and_mask) -> None:
        (hidden_states, mask) = hidden_states_and_mask
        pooled = get_pooled_hidden_states(hidden_states, mask, pool_strategy="last")
        assert len(pooled) == 2
        assert pooled[0].shape == (2, 8)

    def test_cls_pooling_shape(self, hidden_states_and_mask) -> None:
        (hidden_states, mask) = hidden_states_and_mask
        pooled = get_pooled_hidden_states(hidden_states, mask, pool_strategy="cls")
        assert len(pooled) == 2
        assert pooled[0].shape == (2, 8)

    def test_invalid_strategy_raises(self, hidden_states_and_mask) -> None:
        (hidden_states, mask) = hidden_states_and_mask
        with pytest.raises(ValueError, match="Unknown pool_strategy"):
            get_pooled_hidden_states(hidden_states, mask, pool_strategy="invalid")


class TestComputeBlockDistances:
    """Tests for compute_block_distances."""

    def test_output_length(self) -> None:
        pooled = [torch.randn(4, 16) for _ in range(6)]
        dists = compute_block_distances(pooled, layers_to_skip=2)
        assert len(dists) == 4  # 6 - 2 = 4 blocks

    def test_distances_in_valid_range(self) -> None:
        pooled = [torch.randn(4, 16) for _ in range(6)]
        dists = compute_block_distances(pooled, layers_to_skip=1)
        for d in dists:
            assert 0 <= d <= 1


class TestComputeLayerDistances:
    """Tests for compute_layer_distances."""

    def test_returns_correct_format(self) -> None:
        num_layers = 4
        batch, seq, hidden = 2, 5, 8
        hidden_states = (
            torch.randn(batch, seq, hidden),  # emb
            *[torch.randn(batch, seq, hidden) for _ in range(num_layers)],
        )
        mask = torch.ones(batch, seq, dtype=torch.bool)
        result = compute_layer_distances(hidden_states, mask, layers_to_skip=1)
        assert len(result) == num_layers - 1
        for r in result:
            assert "block_start" in r
            assert "block_end" in r
            assert "average_distance" in r
            assert isinstance(r["block_start"], int)
            assert isinstance(r["block_end"], int)


class TestIdentifyPrunableLayers:
    """Tests for identify_prunable_layers."""

    def test_returns_lowest_distance_first(self) -> None:
        distances = [
            {"block_start": 1, "block_end": 2, "average_distance": 0.5},
            {"block_start": 2, "block_end": 3, "average_distance": 0.1},
            {"block_start": 3, "block_end": 4, "average_distance": 0.3},
        ]
        prunable = identify_prunable_layers(distances, top_k=2)
        assert prunable == [2, 3]

    def test_top_k_respected(self) -> None:
        distances = [
            {"block_start": i, "block_end": i + 1, "average_distance": float(i)}
            for i in range(1, 6)
        ]
        prunable = identify_prunable_layers(distances, top_k=3)
        assert prunable == [1, 2, 3]


class TestRunLayerAnalysis:
    """Tests for run_layer_analysis."""

    @pytest.fixture(scope="class")
    def model_and_dataloader(self) -> tuple[GlinerMinusModel, DataLoader]:
        config = GlinerMinusConfig(
            backbone_model_name_or_path="hf-internal-testing/tiny-random-bert",
            num_bilinear_heads=2,
        )
        model = GlinerMinusModel(config)
        # Simple dataloader: random input_ids, attention_mask
        input_ids = torch.randint(1, 100, (12, 16))
        attention_mask = torch.ones(12, 16, dtype=torch.long)
        ds = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(
            ds,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([b[0] for b in batch]),
                "attention_mask": torch.stack([b[1] for b in batch]),
            },
        )
        return model, loader

    def test_run_layer_analysis_returns_valid_output(
        self, model_and_dataloader: tuple[GlinerMinusModel, DataLoader]
    ) -> None:
        model, loader = model_and_dataloader
        distances, prunable = run_layer_analysis(
            model, loader, device="cpu", layers_to_skip=1
        )
        num_layers = model.backbone.config.num_hidden_layers
        assert len(distances) == num_layers - 1
        assert len(prunable) >= 1
        for d in distances:
            assert 0 <= d["average_distance"] <= 1
        assert all(1 <= p <= num_layers for p in prunable)

    def test_works_with_raw_encoder(self) -> None:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        input_ids = torch.randint(1, 100, (8, 12))
        attention_mask = torch.ones(8, 12, dtype=torch.long)
        ds = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(
            ds,
            batch_size=4,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x[0] for x in b]),
                "attention_mask": torch.stack([x[1] for x in b]),
            },
        )
        distances, prunable = run_layer_analysis(
            encoder, loader, device="cpu", layers_to_skip=1
        )
        assert len(distances) > 0
        assert len(prunable) > 0

    def test_prune_backbone_layer_reduces_layers(self) -> None:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        num_layers = backbone.config.num_hidden_layers
        pruned = prune_backbone_layer(backbone, layer_index=1)
        assert pruned.config.num_hidden_layers == num_layers - 1
        # Forward still works
        x = torch.randint(0, 100, (2, 8))
        out = pruned(x)
        assert out.last_hidden_state.shape == (2, 8, pruned.config.hidden_size)

    def test_prune_backbone_invalid_layer_raises(self) -> None:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with pytest.raises(ValueError, match="layer_index"):
            prune_backbone_layer(backbone, layer_index=0)
        with pytest.raises(ValueError, match="layer_index"):
            prune_backbone_layer(backbone, layer_index=99)

    def test_model_without_num_hidden_layers_raises(self) -> None:
        bad_config = type("BadConfig", (), {})()
        bad_backbone = type("BadBackbone", (), {"config": bad_config})()
        bad_model = type("BadModel", (), {"backbone": bad_backbone})()
        loader = DataLoader(
            [{"input_ids": torch.zeros(2, 8, dtype=torch.long), "attention_mask": torch.ones(2, 8)}],
            batch_size=1,
        )
        with pytest.raises(ValueError, match="num_hidden_layers"):
            run_layer_analysis(bad_model, loader, device="cpu")
