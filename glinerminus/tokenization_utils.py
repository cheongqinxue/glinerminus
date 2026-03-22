"""Tokenization helpers for entity–text inputs."""

from __future__ import annotations

from typing import Any

import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def tokenize_entities_text(
    entities: list[list[str]],
    texts: list[str],
    spans: list[list[tuple[int, int]]],
    ent_token: str,
    tokenizer: PreTrainedTokenizerFast,
    **kwargs: Any,
) -> BatchEncoding:
    """Build batch encodings with ``entity_index`` and ``spans`` for `GlinerMinusModel`.

    Each input string is ``{ent_token}{ent_token.join(entities)}{sep}{text}``. Character
    ``spans`` are aligned to token positions in the text segment via ``offset_mapping``.

    Args:
        entities: One list of entity strings per example (same length as ``texts``).
        texts: Raw text for each example (character offsets in ``spans`` refer to these).
        spans: For each example, a list of ``(char_start, char_end)`` inclusive spans in
            ``texts[i]``.
        ent_token: Special token string marking entity type tokens (e.g. ``"<ent>"``).
        tokenizer: Fast tokenizer used to encode inputs.
        **kwargs: Passed to ``tokenizer.__call__`` (e.g. ``padding``, ``truncation``).
            ``return_offsets_mapping`` and ``return_tensors`` are set internally.

    Returns:
        ``BatchEncoding`` with ``input_ids``, ``attention_mask``, ``entity_index`` (per-row
        token indices of ``ent_token``), and ``spans`` (per-row inclusive token index pairs in
        the full encoded sequence).

    Raises:
        ValueError: If ``entities`` and ``texts`` lengths differ.
        TypeError: If ``entities`` is not a list of lists.
    """

    def make_input_str(entity: list[str], text: str) -> str:
        return f"{ent_token}{ent_token.join(entity)}{tokenizer.sep_token}{text}"

    if not isinstance(entities, list) or not entities or not isinstance(entities[0], list):
        msg = f"entities must be a non-empty list of lists, got {type(entities)}"
        raise TypeError(msg)
    if len(entities) != len(texts):
        msg = f"Entities length ({len(entities)}) must match text length ({len(texts)})"
        raise ValueError(msg)
    input_strs: list[str] = [make_input_str(e, t) for e, t in zip(entities, texts, strict=True)]
    if "return_offsets_mapping" in kwargs:
        kwargs.pop("return_offsets_mapping")
    if "return_tensors" in kwargs:
        kwargs.pop("return_tensors")
    inputs: BatchEncoding = tokenizer(
        input_strs,
        return_tensors="pt",
        return_offsets_mapping=True,
        **kwargs,
    )
    ent_token_id: int = tokenizer.convert_tokens_to_ids(ent_token)
    rows, cols = torch.nonzero(inputs.input_ids == ent_token_id, as_tuple=True)
    inputs["entity_index"] = [[] for _ in range(len(texts))]
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        inputs["entity_index"][row].append(col)
    inputs["spans"] = [[] for _ in range(len(texts))]
    for i, seq_spans in enumerate(spans):
        text = texts[i]
        input_str = input_strs[i]
        offset = len(input_str) - len(text)
        for start_, end_ in seq_spans:
            start = start_ + offset
            end = end_ + offset
            start_token_idx: int | None = None
            end_token_idx: int | None = None
            for j, (tid, (s, e)) in enumerate(
                zip(inputs.input_ids[i], inputs.offset_mapping[i], strict=True),
            ):
                if tid.item() in tokenizer.all_special_ids:
                    continue
                if start_token_idx is not None and end_token_idx is not None:
                    break
                if s <= start < e:
                    start_token_idx = j
                if s < end <= e:
                    end_token_idx = j
            inputs["spans"][i].append((start_token_idx, end_token_idx))
    inputs.pop("offset_mapping")
    return inputs
