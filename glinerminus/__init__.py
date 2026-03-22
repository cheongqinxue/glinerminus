"""GlinerMinus: entity-span scoring with a Hugging Face encoder backbone."""

from transformers import AutoConfig, AutoModel

from glinerminus.configuration_gliner_minus import GlinerMinusConfig
from glinerminus.layer_pruning import (
    compute_block_distances,
    compute_layer_distances,
    get_pooled_hidden_states,
    identify_prunable_layers,
    prune_backbone_layer,
    run_layer_analysis,
)
from glinerminus.modeling_gliner_minus import GlinerMinusModel, GlinerMinusOutput, LABEL_IGNORE_INDEX
from glinerminus.tokenization_utils import tokenize_entities_text
from glinerminus.training import (
    GlinerMinusDataCollator,
    PruningDistillationTrainer,
    PruningTrainingArguments,
)

AutoConfig.register(GlinerMinusConfig.model_type, GlinerMinusConfig)
AutoModel.register(GlinerMinusConfig, GlinerMinusModel)

__all__ = [
    "GlinerMinusConfig",
    "GlinerMinusDataCollator",
    "GlinerMinusModel",
    "GlinerMinusOutput",
    "LABEL_IGNORE_INDEX",
    "PruningDistillationTrainer",
    "PruningTrainingArguments",
    "compute_block_distances",
    "compute_layer_distances",
    "get_pooled_hidden_states",
    "identify_prunable_layers",
    "prune_backbone_layer",
    "run_layer_analysis",
    "tokenize_entities_text",
]
