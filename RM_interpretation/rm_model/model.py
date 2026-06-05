"""Contains classes and function for RM model."""
from __future__ import annotations
import numpy as np

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Olmo2PreTrainedModel,GPTNeoXForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from ..neural_abalation.abalate import get_neuron_score
MODEL_PATHS={'olmo-2-1b':'allenai/OLMo-2-0425-1B',
             'smollm2-1.7b':'HuggingFaceTB/SmolLM2-1.7B',
             'smollm2-135m':'HuggingFaceTB/SmolLM2-135M',
             'smollm2-360m':'HuggingFaceTB/SmolLM2-360M',
             'gpt2-large':'openai-community/gpt2-large',
             'gpt2-medium':'openai-community/gpt2-medium',
             'gpt2-small':'openai-community/gpt2',

              }

class Olmo2ForSequenceClassification(GenericForSequenceClassification, Olmo2PreTrainedModel): ...  # noqa: D101

def get_model(model_name:str,rm_type:str,seed:int,traing:bool=True):
    """Returns and AutoModelForSequenceClassification that can be used for traning the RM."""
    if traing :
        model_name=MODEL_PATHS[model_name]
    elif model_name in MODEL_PATHS:
        model_name=f'./save/models/{model_name}_{rm_type}_{seed}/'


    if 'olmo-2' in model_name.lower():
        return Olmo2ForSequenceClassification.from_pretrained(model_name,
		dtype=torch.bfloat16,
                num_labels=1
		),AutoTokenizer.from_pretrained(model_name)
    elif 'pythia' in model_name.lower():
        return GPTNeoXForSequenceClassification.from_pretrained(model_name,
               dtype=torch.bfloat16,
               num_labels=1
               ),AutoTokenizer.from_pretrained(model_name)
    else:
        return AutoModelForSequenceClassification.from_pretrained(model_name,
		dtype=torch.bfloat16,
                num_labels=1
		),AutoTokenizer.from_pretrained(model_name)

class MaskingNeurons:
    """ To Mask Neurons based on the rm model type.
    """
    def __init__(
        self,
        args,
        rm_type:str,
        mask_percentage: float=10.0,
    ):
        self.masking_map={}
        if rm_type in ['harmless','helpful']:
            neuron_scores=get_neuron_score(args.model_name,rm_type,str(args.seed),False,'absolute')
            self.masking_map = self.get_top_indices(
                    neuron_scores, mask_percentage
                )
        elif rm_type=='helpful-harmless':

            neuron_scores1=get_neuron_score(args.model_name,'helpful',str(args.seed),False,'absolute')
            neuron_scores2=get_neuron_score(args.model_name,'harmless',str(args.seed),False,'absolute')

            ValueError("Not coded yet.")
        else:
            ValueError("Not coded yet.")

    @staticmethod
    def get_top_indices(score: np.ndarray, p: float):
        """
        score shape:
          - MLP: (layers, neurons)
          - Attention: (layers, heads)
        """
        threshold = np.percentile(score, 100 - p)
        indices = np.argwhere(score >= threshold)

        mask_map = {}
        for layer, idx in indices:
            mask_map.setdefault(int(layer), set()).add(int(idx))

        return mask_map

    def mask_mlp_neurons(self,block, neuron_indices):

        if not hasattr(block.mlp, "_original_forward"):
            block.mlp._original_forward = block.mlp.forward

        hidden_dim = block.mlp.up_proj.out_features

        mask = torch.ones(hidden_dim, device=block.mlp.up_proj.weight.device)
        mask[neuron_indices] = 0.0
        mask = mask.view(1, 1, -1)

        block.mlp.register_buffer("neuron_mask", mask)

        def masked_forward(x):
            x_up = block.mlp.up_proj(x)
            x_up = x_up * block.mlp.neuron_mask.to(
                device=x_up.device,
                dtype=x_up.dtype
            )

            x_gate = block.mlp.gate_proj(x)
            x = F.silu(x_gate) * x_up
            x = block.mlp.down_proj(x)
            return x

        block.mlp.forward = masked_forward
    
    def apply_mask(self,model):
        for layer_idx, neurons in self.masking_map.items():
            block = model.model.layers[layer_idx]
            self.mask_mlp_neurons(block, list(neurons))

    def prune_mlp_neurons(self, block, neuron_indices):
        """
        Permanently prunes MLP neurons and safely allows further training.
        """

        neuron_indices = torch.as_tensor(
            neuron_indices, device=block.mlp.up_proj.weight.device
        )

        with torch.no_grad():
            # Up projection: remove neuron rows
            block.mlp.up_proj.weight[neuron_indices] = 0.0
            if block.mlp.up_proj.bias is not None:
                block.mlp.up_proj.bias[neuron_indices] = 0.0

            # Down projection: remove neuron columns
            block.mlp.down_proj.weight[:, neuron_indices] = 0.0

        # ---- Gradient blocking (CRITICAL for training) ----

        # Prevent pruned neurons from receiving gradients
        def up_grad_hook(grad):
            grad[neuron_indices] = 0.0
            return grad

        def down_grad_hook(grad):
            grad[:, neuron_indices] = 0.0
            return grad

        block.mlp.up_proj.weight.register_hook(up_grad_hook)
        block.mlp.down_proj.weight.register_hook(down_grad_hook)
    

    def apply_pruning(self, model):
        for layer_idx, neurons in self.masking_map.items():
            block = model.model.layers[layer_idx]
            self.prune_mlp_neurons(block, list(neurons))





