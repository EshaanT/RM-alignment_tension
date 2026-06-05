"""Code for abalation of model."""
from __future__ import annotations

import json

import numpy as np
import torch


def get_neuron_score(model_name:str,rm_type:str,seed:str,randomize:bool=False,scoring_type:str='change')->np.ndarray:
    """Computes and returns neuron scores to be used for abalation.

    Args:
        model_name (str): Name of the mdoel
        rm_type (str): RM model type
        seed (str): model seed

    Returns:
        returns numpy arracy containg neural importance scores.
    """
    path_chosen='save/results/{model_name}_{rm_type}_{seed}/chosen_activation_{scoring_type}.json'
    path_rejected='save/results/{model_name}_{rm_type}_{seed}/rejected_activation_{scoring_type}.json'

    with open(path_chosen.format(model_name=model_name,rm_type=rm_type,seed=seed,scoring_type=scoring_type), "r") as f:
        neural_score_chosen = np.array(json.load(f))
    with open(path_rejected.format(model_name=model_name,rm_type=rm_type,seed=seed,scoring_type=scoring_type), "r") as f:
        neural_score_rejected = np.array(json.load(f))
    if randomize:
        np.random.seed(int(seed))
        dim=neural_score_chosen.shape
        neural_score_chosen=np.random.randn(dim[0],dim[1])
        neural_score_rejected=np.random.randn(dim[0],dim[1])

    
    return (neural_score_chosen+neural_score_rejected)/2

def get_attention_score(model_name:str,rm_type:str,seed:str)->np.ndarray:
    """Computes and returns neuron scores to be used for abalation.

    Args:
        model_name (str): Name of the mdoel
        rm_type (str): RM model type
        seed (str): model seed

    Returns:
        returns numpy arracy containg neural importance scores.
    """
    path_chosen='save/results/{model_name}_{rm_type}_{seed}/chosen_attention_change.json'
    path_rejected='save/results/{model_name}_{rm_type}_{seed}/rejected_attention_change.json'

    with open(path_chosen.format(model_name=model_name,rm_type=rm_type,seed=seed), "r") as f:
        neural_score_chosen = np.array(json.load(f))
    with open(path_rejected.format(model_name=model_name,rm_type=rm_type,seed=seed), "r") as f:
        neural_score_rejected = np.array(json.load(f))
    return (neural_score_chosen+neural_score_rejected)/2


class AblateNeurons:
    """General ablation utility for transformer models (GPT-2, GPT-NeoX/Pythia).

    Supports:
      - MLP neuron ablation.
      - Attention head ablation.
    """

    def __init__(
        self,
        ablate_percentage: float,
        ablate_type: str,  # "mlp" or "attention" or "entropy"
        neuron_scores: np.ndarray | None |list= None, # "List of scores or ordred indices to abalate"
        neuron_scores_ref: np.ndarray= None, # Used when only hateful neurons are to be abalated
    ) -> None:
        """Initialisation of class."""
        self.ablate_percentage = ablate_percentage
        self.ablate_type = ablate_type
        self.hooks = []

        self.ablate_map = None
        if  neuron_scores is not None:
            if ablate_type in ['mlp','attention']:
                self.ablate_map = self.get_top_indices(
                    neuron_scores, ablate_percentage
                )
            elif ablate_type=='entropy':
                self.ablate_map=self.get_p_indices(neuron_scores,ablate_percentage)
            elif ablate_type == 'only_specific':
                top_indices1=self.get_top_indices(
                    neuron_scores, ablate_percentage
                )
                top_indices2=self.get_top_indices(
                    neuron_scores_ref, ablate_percentage
                )
                self.ablate_map={}
                for layer,neurons in top_indices1.items():
                    if layer in top_indices2:
                        specific_neurons=[x for x in neurons if x not in top_indices2[layer]]
                    else:
                        specific_neurons=neurons
                    self.ablate_map[layer]=specific_neurons
            elif ablate_type == 'both_intersection':
                top_indices1=self.get_top_indices(
                    neuron_scores, ablate_percentage
                )
                top_indices2=self.get_top_indices(
                    neuron_scores_ref, ablate_percentage
                )
                self.ablate_map={}
                for layer,neurons in top_indices1.items():
                    if layer in top_indices2:
                        specific_neurons=[x for x in neurons if x in top_indices2[layer]]
                    else:
                        specific_neurons=neurons
                    self.ablate_map[layer]=specific_neurons
                for layer,neurons in top_indices2.items():
                    if layer not in self.ablate_map:
                        self.ablate_map[layer]=neurons
            else:
                raise ValueError("Incorrect abalate type")

    # -------------------------
    # SELECT TOP INDICES
    # -------------------------
    @staticmethod
    def get_top_indices(score: np.ndarray, p: float)->{}:
        """Score shape:

          MLP: (layers, neurons)
          Attention: (layers, heads).
        
        Returns:
            return map of neutons to abalate.
        """
        threshold = np.percentile(score, 100 - p)
        indices = np.argwhere(score >= threshold)

        ablate_map = {}
        for layer, idx in indices:
            ablate_map.setdefault(int(layer), set()).add(int(idx))

        return ablate_map

    @staticmethod
    def get_p_indices(order_index: list, p: float)->{}:
        """
        Score shape:
          - MLP: (layers, neurons)
          - Attention: (layers, heads)
        """

        ablate_map = {}
        for layer, idxs in enumerate(order_index):
            ablate_map[layer]=idxs.tolist()
        #print(ablate_map)
        return ablate_map

    # -------------------------
    # MODEL ADAPTER
    # -------------------------
    def _get_layers(self, model):
        if hasattr(model, "transformer"):  # GPT-2
            return model.transformer.h
        elif hasattr(model,'model'): # smollm2 and opt
            if hasattr(model.model,'layers'): # smollm2
                return model.model.layers
            elif hasattr(model.model,'decoder'):
                return model.model.decoder.layers
        elif hasattr(model, "gpt_neox"):  # Pythia
            return model.gpt_neox.layers
        else:
            raise ValueError("Unsupported model architecture")

    def _get_num_heads(self, model):
        if hasattr(model.config, "n_head"):
            return model.config.n_head  # GPT-2
        elif hasattr(model.config, "num_attention_heads"):
            return model.config.num_attention_heads  # NeoX
        else:
            raise ValueError("Cannot find number of heads")

    # -------------------------
    # MLP ABLATION
    # -------------------------
    def _mlp_hook(self, layer_idx):
        def hook(module, inputs, output):
            if self.ablate_map is None:
                return torch.zeros_like(output)

            if layer_idx in self.ablate_map:
                out = output.clone()
                neuron_idx = list(self.ablate_map[layer_idx])
                if out.dim() == 3:
                    out[:, :, neuron_idx] = 0
                elif out.dim() == 2:
                    out[:, neuron_idx] = 0

                return out

            return output

        return hook

    def ablate_mlp(self, model):
        layers = self._get_layers(model)

        for i, layer in enumerate(layers):
            # GPT-2: hook c_fc output (before projection)
            if hasattr(layer,'mlp'):
                if hasattr(layer.mlp, "act"):
                    hook = layer.mlp.act.register_forward_hook(
                        self._mlp_hook(i)
                    )
                else:
                    # NeoX fallback
                    hook = layer.mlp.act_fn.register_forward_hook(
                        self._mlp_hook(i)
                    )
            elif hasattr(layer,'activation_fn'):
                hook = layer.activation_fn.register_forward_hook(
                        self._mlp_hook(i)
                    )

            self.hooks.append(hook)

    # -------------------------
    # ATTENTION ABLATION
    # -------------------------
    def _attention_hook(self, layer_idx, num_heads):
        def hook(module, inputs):
            x = inputs[0]

            if self.ablate_map is None:
                return (torch.zeros_like(x),)

            if layer_idx in self.ablate_map:
                b, s, h = x.shape
                head_dim = h // num_heads

                heads = x.view(b, s, num_heads, head_dim)

                for head in self.ablate_map[layer_idx]:
                    heads[:, :, head, :] = 0

                x = heads.view(b, s, h)

            return (x,)

        return hook

    def ablate_attention(self, model):
        layers = self._get_layers(model)
        num_heads = self._get_num_heads(model)

        for i, layer in enumerate(layers):
            if hasattr(layer.attn, "c_proj"):  # GPT-2
                module = layer.attn.c_proj
            else:  # NeoX
                module = layer.attention.dense

            hook = module.register_forward_pre_hook(
                self._attention_hook(i, num_heads)
            )
            self.hooks.append(hook)

    # -------------------------
    # APPLY
    # -------------------------
    def apply(self, model):
        if self.ablate_type == "attention":
            self.ablate_attention(model)
        else:
            self.ablate_mlp(model)
    # -------------------------
    # CLEANUP
    # -------------------------
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def apply_ablation(args, model):
    """
    Apply ablation based on args.

    Supported types:
      - top_attention
      - all_attention
      - top_mlp
      - all_mlp
    """

    ablator = None

    # -------------------------
    # ATTENTION (TOP %)
    # -------------------------
    if args.abalate_neuron_type in ["top_attention", "top_attention_diff"]:
        head_scores = get_attention_score(
            args.model_name,
            args.rm_type,
            str(args.seed)
        )

        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="attention",
            neuron_scores=head_scores
        )

    elif args.abalate_neuron_type == "random_activations":

        neural_scores = get_neuron_score(
            args.model_name,
            args.rm_type,
            str(args.seed),
            True
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="mlp",
            neuron_scores=neural_scores
        )

    elif args.abalate_neuron_type == "top_activations":

        neural_scores = get_neuron_score(
            args.model_name,
            args.rm_type,
            str(args.seed)
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="mlp",
            neuron_scores=neural_scores
        )

    elif args.abalate_neuron_type == "top_activations_absolute":

        neural_scores = get_neuron_score(
            args.model_name,
            args.rm_type,
            str(args.seed),
            False,
            'absolute'
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="mlp",
            neuron_scores=neural_scores
        )
    elif args.abalate_neuron_type == "top_activations_prob":

        neural_scores = get_neuron_score(
            args.model_name,
            args.rm_type,
            str(args.seed),
            False,
            'prob'
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="mlp",
            neuron_scores=neural_scores
        )
    elif args.abalate_neuron_type == "only_top_helpful_activations_absolute":
        helpful_neural_scores=get_neuron_score(
            args.model_name,
            'helpful',
            str(args.seed),
            False,
            'absolute'
        )
        harmless_neural_scores=get_neuron_score(
            args.model_name,
            'harmless',
            str(args.seed),
            False,
            'absolute'
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="only_specific",
            neuron_scores=helpful_neural_scores,
            neuron_scores_ref=harmless_neural_scores,
        )
    elif args.abalate_neuron_type == "only_top_harmless_activations_absolute":
        helpful_neural_scores=get_neuron_score(
            args.model_name,
            'helpful',
            str(args.seed),
            False,
            'absolute'
        )
        harmless_neural_scores=get_neuron_score(
            args.model_name,
            'harmless',
            str(args.seed),
            False,
            'absolute'
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="only_specific",
            neuron_scores=harmless_neural_scores,
            neuron_scores_ref=helpful_neural_scores,
        )

    elif args.abalate_neuron_type == "both_top_helpful_harmless_activations_absolute":
        helpful_neural_scores=get_neuron_score(
            args.model_name,
            'helpful',
            str(args.seed),
            False,
            'absolute'
        )
        harmless_neural_scores=get_neuron_score(
            args.model_name,
            'harmless',
            str(args.seed),
            False,
            'absolute'
        )
        ablator = AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="both_intersection",
            neuron_scores=harmless_neural_scores,
            neuron_scores_ref=helpful_neural_scores,
        )

    elif args.abalate_neuron_type=='top_activations_entropy':
        helpful_neural_scores=get_neuron_score(
            args.model_name,
            'helpful',
            str(args.seed),
            False,
            'prob'
        )
        harmless_neural_scores=get_neuron_score(
            args.model_name,
            'harmless',
            str(args.seed),
            False,
            'prob'
        )
        activation_probs=torch.stack([torch.tensor(helpful_neural_scores),torch.tensor(harmless_neural_scores)], dim=-1)
        filter_rate=0.95
        top_rate=args.abalate_neuron_percentage/100
        activation_bar_ratio = 0.95
        largest=False
        num_layers=activation_probs.shape[0]
        normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
        normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
        log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
        entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
        flattened_probs = activation_probs.flatten()
        top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
        top_position = (activation_probs > top_prob_value).sum(dim=-1)
        entropy[top_position == 0] = torch.inf
        flattened_entropy = entropy.flatten()
        top_entropy_value = round(len(flattened_entropy) * top_rate)
        _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
        row_index = index // entropy.size(1)
        col_index = index % entropy.size(1)
        selected_probs = activation_probs[row_index, col_index] # n x lang
            # for r, c in zip(row_index, col_index):
            #     print(r, c, activation_probs[r][c])

        print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
        selected_probs = selected_probs.transpose(0, 1)
        activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
        print(activation_bar)
        print((selected_probs > activation_bar).sum(dim=1).tolist())
        lang, indice = torch.where(selected_probs > activation_bar)

        merged_index = torch.stack((row_index, col_index), dim=-1)
        final_indice = []
        for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
            lang_index = [tuple(row.tolist()) for row in merged_index[index]]
            lang_index.sort()
            layer_index = [[] for _ in range(num_layers)]
            for l, h in lang_index:
                layer_index[l].append(h)
            for l, h in enumerate(layer_index):
                layer_index[l] = torch.tensor(h).long()
            final_indice.append(layer_index)

        harmless_neurons = final_indice[1]

        helpful_neurons = final_indice[0]
        neural_scores= helpful_neurons if args.rm_type=='helpful' else harmless_neurons

        ablator=AblateNeurons(
            ablate_percentage=args.abalate_neuron_percentage,
            ablate_type="entropy",
            neuron_scores=neural_scores
        )

    elif args.abalate_neuron_type == "all_attention":
        ablator = AblateNeurons(
            ablate_percentage=100,
            ablate_type="attention",
            neuron_scores=None
        )

    # -------------------------
    # MLP (ALL)
    # -------------------------
    elif args.abalate_neuron_type == "all_activations":
        ablator = AblateNeurons(
            ablate_percentage=100,
            ablate_type="mlp",
            neuron_scores=None
        )

    else:
        raise ValueError(f"Unknown ablation type: {args.abalate_neuron_type}")

    # -------------------------
    # APPLY HOOKS
    # -------------------------
    ablator.apply(model)

    return ablator
