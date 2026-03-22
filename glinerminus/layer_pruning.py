"""Layer pruning utilities for encoder models.

Adapted from the PruneMe repository (https://github.com/arcee-ai/PruneMe) which was
designed for causal LLMs. For encoders, we use mean-pooling over non-padded tokens
instead of the last-token representation, since encoders have no inherent "next-token"
position and all token positions contribute to the sequence representation.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def angular_distance(x_l: Tensor, x_l_plus_n: Tensor) -> Tensor:
    """Compute the angular distance between two representations (in [0, 1]).

    Angular distance is 1 - (angle/π), derived from cosine similarity. Lower values
    indicate more similar representations.

    Args:
        x_l: First representation of shape (batch, hidden_dim).
        x_l_plus_n: Second representation of shape (batch, hidden_dim).

    Returns:
        Tensor of shape (batch,) with angular distances per sample.

    """
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True).clamp(min=1e-8)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True).clamp(min=1e-8)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def get_pooled_hidden_states(
    hidden_states: tuple[Tensor, ...],
    attention_mask: Tensor,
    pool_strategy: Literal["mean", "last", "cls"] = "mean",
) -> list[Tensor]:
    """Extract a single vector per layer from encoder hidden states.

    For encoders, we pool over the sequence dimension rather than using the last
    token (which is causal-LM specific). Mean pooling over non-padded tokens gives
    a sequence-level representation per layer.

    Args:
        hidden_states: Tuple of (embedding_output, layer_0_out, ..., layer_N_out).
            Length is num_hidden_layers + 1.
        attention_mask: Boolean mask of shape (batch, seq_len), True for valid tokens.
        pool_strategy: How to aggregate tokens per layer.
            - "mean": Mean over non-padded tokens (recommended for encoders).
            - "last": Last non-padded token per sequence (PruneMe-style, for compat).
            - "cls": First token per sequence (for models with [CLS] or similar).

    Returns:
        List of tensors of shape (batch, hidden_dim), one per encoder layer
        (excludes embedding layer). Length is num_hidden_layers.

    """
    # hidden_states[0] is embeddings; [1:] are encoder layers
    layer_outputs = hidden_states[1:]
    pooled: list[Tensor] = []
    batch_size, hidden_dim = layer_outputs[0].shape[0], layer_outputs[0].shape[2]

    for layer_hidden in layer_outputs:
        if pool_strategy == "mean":
            # (batch, seq_len, hidden) -> mask and mean over seq
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(layer_hidden).float()
            layer_hidden_masked = layer_hidden * mask_expanded
            sum_hidden = layer_hidden_masked.sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-8)
            batch_repr = sum_hidden / sum_mask
        elif pool_strategy == "last":
            batch_last_tokens = []
            for b in range(batch_size):
                last_idx = attention_mask[b].nonzero(as_tuple=True)[0].max()
                batch_last_tokens.append(layer_hidden[b, last_idx, :].unsqueeze(0))
            batch_repr = torch.cat(batch_last_tokens, dim=0)
        elif pool_strategy == "cls":
            batch_repr = layer_hidden[:, 0, :]  # First token
        else:
            msg = f"Unknown pool_strategy: {pool_strategy}"
            raise ValueError(msg)
        pooled.append(batch_repr)

    return pooled


def compute_block_distances(
    pooled_states: list[Tensor],
    layers_to_skip: int,
) -> list[float]:
    """Compute angular distances between layer L and layer L+n for each block.

    For each starting layer index l, computes the mean angular distance between
    pooled_states[l] and pooled_states[l + layers_to_skip] across the batch.
    Lower distance indicates the block is more redundant (good pruning candidate).

    Args:
        pooled_states: List of (batch, hidden_dim) tensors, one per layer.
        layers_to_skip: Number of layers in each block (n in L vs L+n comparison).

    Returns:
        List of mean angular distances, one per (valid) block start index.

    """
    num_layers = len(pooled_states)
    distances: list[float] = []
    for layer_idx in range(num_layers - layers_to_skip):
        dist = (
            angular_distance(
                pooled_states[layer_idx],
                pooled_states[layer_idx + layers_to_skip],
            )
            .mean()
            .item()
        )
        distances.append(dist)
    return distances


def compute_layer_distances(
    hidden_states: tuple[Tensor, ...],
    attention_mask: Tensor,
    layers_to_skip: int = 1,
    pool_strategy: Literal["mean", "last", "cls"] = "mean",
) -> list[dict[str, float | int]]:
    """Compute block-wise angular distances for layer pruning analysis.

    Args:
        hidden_states: Tuple from encoder with output_hidden_states=True.
        attention_mask: Attention mask (batch, seq_len).
        layers_to_skip: Block size (compare layer L with layer L+layers_to_skip).
        pool_strategy: Pooling strategy for encoder tokens.

    Returns:
        List of dicts with keys:
            - block_start: 1-based layer index (inclusive).
            - block_end: 1-based layer index (inclusive).
            - average_distance: Mean angular distance for this block.

    """
    pooled = get_pooled_hidden_states(hidden_states, attention_mask, pool_strategy)
    distances = compute_block_distances(pooled, layers_to_skip)

    results: list[dict[str, float | int]] = []
    for i, avg_dist in enumerate(distances):
        results.append(
            {
                "block_start": i + 1,
                "block_end": i + 1 + layers_to_skip,
                "average_distance": avg_dist,
            },
        )
    return results


def identify_prunable_layers(
    layer_distances: list[dict[str, float | int]],
    top_k: int = 1,
) -> list[int]:
    """Identify layers that are good candidates for pruning.

    Layers with the lowest average_distance (most similar to the next N layers)
    are considered most redundant and thus best candidates for removal.

    Args:
        layer_distances: Output from compute_layer_distances.
        top_k: Number of block indices to return (lowest distance first).

    Returns:
        List of 1-based block_start indices, sorted by ascending distance.

    """
    sorted_results = sorted(layer_distances, key=lambda d: float(d["average_distance"]))
    return [int(d["block_start"]) for d in sorted_results[:top_k]]


def run_layer_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str | torch.device = "cuda",
    layers_to_skip: int = 1,
    pool_strategy: Literal["mean", "last", "cls"] = "mean",
) -> tuple[list[dict[str, float | int]], list[int]]:
    """Run layer similarity analysis over a dataset.

    Uses the model's forward pass with output_hidden_states=True. Works with
    any encoder that exposes a .backbone attribute (e.g. GlinerMinusModel) or
    is itself an encoder (e.g. AutoModel).

    Args:
        model: Encoder or model with .backbone that returns hidden_states.
        dataloader: Yields batches. Each batch can be:
            - BatchEncoding (e.g. from tokenizer) with input_ids, attention_mask
            - dict with "input_ids" and "attention_mask" keys
        device: Device to run on.
        layers_to_skip: Block size for distance computation.
        pool_strategy: Pooling strategy for encoder tokens.

    Returns:
        Tuple of (all_layer_distances, prunable_layer_indices).
        - all_layer_distances: List of dicts with block_start, block_end,
          average_distance (averaged across all batches).
        - prunable_layer_indices: 1-based layer indices with lowest distance.

    Raises:
        ValueError: If model has no num_hidden_layers or hidden_states output.

    """
    # Resolve backbone for wrapper models like GlinerMinusModel
    encoder = getattr(model, "backbone", model)
    config = getattr(encoder, "config", None)
    num_layers = getattr(config, "num_hidden_layers", None) if config else None
    if num_layers is None:
        msg = "Model must have config.num_hidden_layers (encoder expected)"
        raise ValueError(msg)

    model.eval()
    all_distances: list[list[float]] = [[] for _ in range(num_layers - layers_to_skip)]

    # Use encoder (backbone) for forward pass: tokenized text only, no entity_index/spans
    with torch.no_grad():
        for batch in dataloader:
            if hasattr(batch, "keys"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
                else:
                    attention_mask = attention_mask.to(device)
            else:
                msg = "Batch must be BatchEncoding or dict with input_ids and attention_mask"
                raise ValueError(msg)

            backbone_out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = backbone_out.hidden_states
            if hidden_states is None:
                msg = "Encoder must return hidden_states when output_hidden_states=True"
                raise ValueError(msg)

            distances = compute_layer_distances(
                hidden_states,
                attention_mask,
                layers_to_skip=layers_to_skip,
                pool_strategy=pool_strategy,
            )
            for i, d in enumerate(distances):
                all_distances[i].append(float(d["average_distance"]))

    # Average across batches
    average_distances: list[dict[str, float | int]] = [
        {
            "block_start": i + 1,
            "block_end": i + 1 + layers_to_skip,
            "average_distance": float(np.mean(d)),
        }
        for i, d in enumerate(all_distances)
    ]
    prunable = identify_prunable_layers(average_distances, top_k=min(5, len(average_distances)))

    if prunable:
        min_idx = int(prunable[0])
        min_dist = average_distances[min_idx - 1]["average_distance"]
        logger.info(
            "Layer %d to %d has minimum average distance %.4f. Consider pruning this block.",
            min_idx,
            min_idx + layers_to_skip,
            float(min_dist),
        )

    return average_distances, prunable


def prune_backbone_layer(
    backbone: torch.nn.Module,
    layer_index: int,
) -> torch.nn.Module:
    """Remove a single layer from an encoder backbone.

    Creates a new backbone with num_hidden_layers - 1, copying weights from all
    layers except the pruned one. Supports BERT-style encoders (encoder.layer).

    Args:
        backbone: Encoder model (e.g. BertModel, AutoModel) with config.
        layer_index: 1-based layer index to remove (from identify_prunable_layers).

    Returns:
        New backbone with one fewer layer and weights copied.

    Raises:
        ValueError: If backbone has no encoder.layer or config.

    """
    import re

    from transformers import AutoModel

    encoder = getattr(backbone, "encoder", None)
    if encoder is None:
        msg = "Backbone must have an encoder attribute (e.g. BERT, RoBERTa)"
        raise ValueError(msg)
    layer_module = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layer_module is None:
        msg = "Encoder must have .layer or .layers (ModuleList)"
        raise ValueError(msg)
    config = getattr(backbone, "config", None)
    if config is None:
        msg = "Backbone must have config"
        raise ValueError(msg)

    num_layers = len(layer_module)
    if layer_index < 1 or layer_index > num_layers:
        msg = f"layer_index must be 1-based in [1, {num_layers}], got {layer_index}"
        raise ValueError(msg)

    prune_idx = layer_index - 1  # 0-based

    # Create new config with fewer layers
    new_config_dict = config.to_dict()
    new_config_dict["num_hidden_layers"] = num_layers - 1
    new_config = config.__class__(**new_config_dict)

    # Create new backbone
    new_backbone = AutoModel.from_config(new_config)

    # Map state dict: skip prune_idx, shift layer indices after it down
    old_state = backbone.state_dict()
    new_state: dict[str, torch.Tensor] = {}
    # Match encoder.layer.N. or encoder.layers.N.
    pattern = re.compile(r"^(.*encoder\.(?:layer|layers))\.(\d+)\.(.*)$")

    for key, value in old_state.items():
        match = pattern.match(key)
        if match:
            prefix, layer_str, suffix = match.groups()
            layer_num = int(layer_str)
            if layer_num == prune_idx:
                continue
            new_layer_num = layer_num - 1 if layer_num > prune_idx else layer_num
            new_key = f"{prefix}.{new_layer_num}.{suffix}"
            new_state[new_key] = value
        else:
            new_state[key] = value

    missing, unexpected = new_backbone.load_state_dict(new_state, strict=False)
    if missing:
        logger.debug("Pruned backbone missing keys: %s", missing[:5])
    return new_backbone
