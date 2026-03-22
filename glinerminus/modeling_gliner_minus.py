"""Entity–span bilinear head over a frozen or trainable encoder backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from glinerminus.configuration_gliner_minus import GlinerMinusConfig
from glinerminus.tokenization_utils import tokenize_entities_text
from transformers.tokenization_utils_base import BatchEncoding

LABEL_IGNORE_INDEX = -100


@dataclass
class GlinerMinusOutput(ModelOutput):
    """Output of `GlinerMinusModel`.

    Args:
        loss: Mean squared error over valid positions when ``labels`` is provided; otherwise
            ``None``.
        logits: Bilinear scores of shape ``(batch, max_entities, max_spans, num_heads)``.
        entity_mask: ``True`` for real entity slots after padding.
        span_mask: ``True`` for real span slots after padding (invalid spans are ``False``).
        hidden_states: Optional tuple of backbone hidden states when requested.
    """

    loss: Tensor | None = None
    logits: Tensor | None = None
    entity_mask: Tensor | None = None
    span_mask: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


class GlinerMinusModel(PreTrainedModel):
    """Encoder backbone plus per-sequence entity×span bilinear scores.

    Expects ``entity_index`` and ``spans`` as nested Python lists (see `forward`).

    Args:
        config: `GlinerMinusConfig` instance.
    """

    config_class = GlinerMinusConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_save: list[str] = []

    def __init__(self, config: GlinerMinusConfig, tokenizer: Any | None = None) -> None:
        super().__init__(config)
        if not config.backbone_model_name_or_path:
            msg = "backbone_model_name_or_path must be set in GlinerMinusConfig"
            raise ValueError(msg)
        if tokenizer is None:
            self.backbone = AutoModel.from_pretrained(config.backbone_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_name_or_path)
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [config.ent_token]},
            )
            self.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = tokenizer
            backbone_config = AutoConfig.from_pretrained(config.backbone_model_name_or_path)
            backbone_config.vocab_size = len(tokenizer)
            self.backbone = AutoModel.from_config(backbone_config)
        hidden_size = config.hidden_size or int(self.backbone.config.hidden_size)
        self.config.hidden_size = hidden_size  # type: ignore[misc]
        self.layer_norm_entity = nn.LayerNorm(hidden_size)
        self.layer_norm_span = nn.LayerNorm(hidden_size)
        self.bilinear_weight = nn.Parameter(
            torch.empty(config.num_bilinear_heads, hidden_size, hidden_size),
        )
        for k in range(self.bilinear_weight.size(0)):
            nn.init.xavier_uniform_(self.bilinear_weight[k])

    def get_input_embeddings(self) -> nn.Module:
        """Return the backbone's input embeddings (for weight tying / resizing)."""
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the backbone's input embeddings."""
        self.backbone.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int | None = None) -> nn.Embedding:
        """Resize token embeddings on the backbone."""
        return self.backbone.resize_token_embeddings(new_num_tokens)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        """Enable gradient checkpointing on the backbone encoder."""
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the backbone encoder."""
        self.backbone.gradient_checkpointing_disable()

    def tokenize(
        self,
        entities: list[list[str]],
        texts: list[str],
        spans: list[list[tuple[int, int]]],
        **kwargs: Any,
    ) -> BatchEncoding:
        """Create a `BatchEncoding` for the given entities, texts, and character spans.

        Delegates to `tokenize_entities_text` with this model's tokenizer and entity token.

        Args:
            entities: One list of entity strings per example (same length as ``texts``).
            texts: Raw text for each example (character offsets in ``spans`` refer to these).
            spans: For each example, a list of ``(char_start, char_end)`` inclusive spans
                in ``texts[i]``.
            **kwargs: Passed to the tokenizer (e.g. ``padding``, ``truncation``).

        Returns:
            BatchEncoding with ``input_ids``, ``attention_mask``, ``entity_index``,
            and ``spans`` (token-level, inclusive).
        """
        return tokenize_entities_text(
            entities=entities,
            texts=texts,
            spans=spans,
            ent_token=self.config.ent_token,
            tokenizer=self.tokenizer,
            **kwargs,
        )

    def save_pretrained(self, save_directory: str | None = None, **kwargs: Any) -> None:
        """Save model and tokenizer to ``save_directory``."""
        super().save_pretrained(save_directory, **kwargs)
        if save_directory is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls: type[GlinerMinusModel],
        pretrained_model_name_or_path: str | None,
        *model_args: Any,
        **kwargs: Any,
    ) -> GlinerMinusModel:
        """Load model and tokenizer from ``pretrained_model_name_or_path``.

        If a tokenizer exists in the save directory, it is loaded from there.
        Otherwise, loads from the backbone path and adds the entity special token.
        """
        config = kwargs.pop("config", None)
        if config is None and pretrained_model_name_or_path is not None:
            config = GlinerMinusConfig.from_pretrained(pretrained_model_name_or_path)
        if config is None:
            msg = "config must be provided when pretrained_model_name_or_path is None"
            raise ValueError(msg)
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(config.backbone_model_name_or_path)
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [config.ent_token]},
            )
        kwargs["tokenizer"] = tokenizer
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        entity_index: list[list[int]] | None = None,
        spans: list[list[tuple[int | None, int | None]]] | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> GlinerMinusOutput | tuple[Any, ...]:
        """Compute bilinear logits for each entity×span pair within each sequence.

        For batch index ``b``, only pairs ``(entity_slot, span_slot)`` with both masks true
        are meaningful; logits at padded positions should be ignored (e.g. masked loss).

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            attention_mask: Optional mask ``(batch, seq_len)`` for the backbone.
            entity_index: For each sequence, token positions of entity markers (length ``E_b``).
            spans: For each sequence, inclusive token span pairs ``(start, end)`` in the
                encoded sequence. Entries with ``None`` are skipped.
            labels: Optional targets matching ``logits`` shape ``(batch, max_E, max_S, K)``.
                Use ``LABEL_IGNORE_INDEX`` (-100) for ignored positions.
            output_hidden_states: If true, backbone returns all hidden states.
            return_dict: If false, return a tuple ``(loss, logits, ...)`` when applicable.

        Returns:
            `GlinerMinusOutput` (or tuple if ``return_dict`` is false).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if entity_index is None or spans is None:
            msg = "entity_index and spans are required"
            raise ValueError(msg)
        backbone_kw_keys = {"labels", "entity_index", "spans", "return_dict"}
        backbone_kwargs = {k: v for k, v in kwargs.items() if k not in backbone_kw_keys}
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
            **backbone_kwargs,
        )
        hidden = backbone_outputs.last_hidden_state
        batch_size, _, hidden_dim = hidden.shape
        device = hidden.device
        num_heads = self.config.num_bilinear_heads

        max_e = max((len(e) for e in entity_index), default=0)
        max_s = max((len(s) for s in spans), default=0)

        logits = hidden.new_zeros(batch_size, max_e, max_s, num_heads)
        entity_mask = torch.zeros(batch_size, max_e, dtype=torch.bool, device=device)
        span_mask = torch.zeros(batch_size, max_s, dtype=torch.bool, device=device)

        seq_len = hidden.size(1)
        for b in range(batch_size):
            ent_pos = entity_index[b]
            seq_spans = spans[b]
            e_b = len(ent_pos)
            s_b = len(seq_spans)
            if e_b:
                entity_mask[b, :e_b] = True
            span_valid: list[bool] = []
            span_rows: list[Tensor] = []
            for start, end in seq_spans:
                if start is None or end is None:
                    span_valid.append(False)
                    span_rows.append(torch.zeros(hidden_dim, device=device, dtype=hidden.dtype))
                    continue
                s_i, e_i = int(start), int(end)
                ok = 0 <= s_i <= e_i < seq_len
                span_valid.append(ok)
                if not ok:
                    span_rows.append(torch.zeros(hidden_dim, device=device, dtype=hidden.dtype))
                    continue
                span_rows.append(hidden[b, s_i : e_i + 1].sum(dim=0))
            if s_b:
                span_mask[b, :s_b] = torch.tensor(span_valid, dtype=torch.bool, device=device)
            if e_b == 0 or s_b == 0:
                continue

            entity_matrix = hidden[b, ent_pos, :]
            span_features = torch.stack(span_rows, dim=0)

            ent_n = self.layer_norm_entity(entity_matrix)
            span_n = self.layer_norm_span(span_features)
            logits_b = torch.einsum("qa,kab,jb->qjk", ent_n, self.bilinear_weight, span_n)
            logits[b, :e_b, :s_b] = logits_b

        loss: Tensor | None = None
        if labels is not None:
            if labels.shape != logits.shape:
                msg = f"labels shape {tuple(labels.shape)} must match logits {tuple(logits.shape)}"
                raise ValueError(msg)
            labels_f = labels.float()
            pair_mask = entity_mask.unsqueeze(2) & span_mask.unsqueeze(1)
            valid = pair_mask.unsqueeze(-1) & (labels != LABEL_IGNORE_INDEX)
            diff = F.mse_loss(logits, labels_f, reduction="none")
            diff = diff.masked_fill(~valid, 0.0)
            denom = valid.sum().clamp(min=1.0)
            loss = diff.sum() / denom

        if not return_dict:
            out_tuple = (loss, logits, entity_mask, span_mask)
            if output_hidden_states:
                out_tuple = out_tuple + (backbone_outputs.hidden_states,)
            return out_tuple

        return GlinerMinusOutput(
            loss=loss,
            logits=logits,
            entity_mask=entity_mask,
            span_mask=span_mask,
            hidden_states=backbone_outputs.hidden_states if output_hidden_states else None,
        )
