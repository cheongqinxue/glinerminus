"""Configuration for the GlinerMinus entity–span scoring model."""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class GlinerMinusConfig(PretrainedConfig):
    """Configuration class for `GlinerMinusModel`.

    Args:
        backbone_model_name_or_path: Hugging Face hub id or local path for `AutoModel`.
        num_bilinear_heads: Number of bilinear head matrices ``K`` in the einsum head.
        hidden_size: If set, used for the head; otherwise taken from the backbone at init.
        **kwargs: Forwarded to `PretrainedConfig`.
    """

    model_type = "gliner_minus"

    def __init__(
        self,
        backbone_model_name_or_path: str = "",
        num_bilinear_heads: int = 3,
        hidden_size: int | None = None,
        ent_token: str = "<ent>",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path
        self.num_bilinear_heads = num_bilinear_heads
        self.hidden_size = hidden_size
        self.ent_token = ent_token
