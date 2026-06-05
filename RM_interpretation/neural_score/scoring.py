"""Code for neuron scoring."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    import transformers
    from torch.utils.data import DataLoader

class ChangeAnalyzer:
    """General activation difference analyzer for transformer models (GPT-2, GPT-NeoX/Pythia).

    Computes:
      - Attention change per head
      - MLP RMS change per neuron

    Uses forward hooks (no TraceDict).
    """

    def __init__(
        self,
        base_model: transformers.PreTrainedModel|None,
        trained_model: transformers.PreTrainedModel,
    ):
        self.base_model = base_model
        self.trained_model = trained_model

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
            return model.config.num_attention_heads  # NeoX and smollm2
        else:
            raise ValueError("Cannot find number of heads")

    def _get_hidden_size(self, model):
        return model.config.hidden_size

    def _get_intermediate_size(self, model, layer):
        # GPT-2
        if hasattr(model.config, "n_embd"):
            return 4*model.config.n_embd
        # NeoX and smollm2
        elif hasattr(model.config, "intermediate_size"):
            return model.config.intermediate_size
        elif hasattr(model.config, "ffn_dim"):
            return model.config.ffn_dim
        else:
            raise ValueError("Cannot determine intermediate size")

    # -------------------------
    # HOOK HELPERS
    # -------------------------
    def _collect_outputs(self, model, input_ids, hook_fns):
        model.eval()
        hooks = []
        outputs = {}

        for name, module, fn in hook_fns:
            hooks.append(module.register_forward_hook(fn(name, outputs)))

        with torch.no_grad():
            _ = model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        return outputs

    # -------------------------
    # ATTENTION
    # -------------------------
    def _get_attention_modules(self, model):
        layers = self._get_layers(model)
        modules = []

        for i, layer in enumerate(layers):
            if hasattr(layer.attn, "c_proj"):  # GPT-2
                module = layer.attn.c_proj
            else:  # NeoX
                module = layer.attention.dense

            modules.append((f"layer_{i}", module))

        return modules

    def _attention_hook(self, name, store):
        def hook(module, inputs, output):
            store[name] = inputs[0].detach()
        return hook

    def compute_attention_change(
        self,
        dataloader: DataLoader,
        input_type: str,
        device: str | None = None
    ):

        if device is None:
            device = next(self.trained_model.parameters()).device

        layers = self._get_layers(self.trained_model)
        num_layers = len(layers)
        num_heads = self._get_num_heads(self.trained_model)
        hidden_size = self._get_hidden_size(self.trained_model)
        head_dim = hidden_size // num_heads

        activation = torch.zeros(num_layers, num_heads, device=device)
        total_tokens = 0

        base_modules = self._get_attention_modules(self.base_model)
        train_modules = self._get_attention_modules(self.trained_model)

        for batch in tqdm(dataloader, desc=f"Attention change ({input_type})"):
            input_ids = batch[f"{input_type}_input_ids"].to(device)
            attention_mask = batch[f"{input_type}_attention_mask"].to(device)

            base_out = self._collect_outputs(
                self.base_model,
                input_ids,
                [(n, m, self._attention_hook) for n, m in base_modules],
            )

            train_out = self._collect_outputs(
                self.trained_model,
                input_ids,
                [(n, m, self._attention_hook) for n, m in train_modules],
            )

            total_tokens += attention_mask.sum().item()

            for i in range(num_layers):
                b = base_out[f"layer_{i}"]
                t = train_out[f"layer_{i}"]

                diff = t - b  # (B, S, H)
                B, S, H = diff.shape

                heads = diff.view(B, S, num_heads, head_dim)

                mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
                heads = heads * mask

                scores = torch.norm(heads, dim=-1)
                activation[i] += scores.sum(dim=(0, 1))

        return activation / total_tokens

    # -------------------------
    # MLP
    # -------------------------
    def _get_mlp_modules(self, model):
        layers = self._get_layers(model)
        modules = []

        for i, layer in enumerate(layers):
            if hasattr(layer,'mlp'):
                if hasattr(layer.mlp, "act"):  # GPT-2
                    module = layer.mlp.act
                elif hasattr(layer.mlp, "act_fn"):  # NeoX
                    module = layer.mlp.act_fn
            elif hasattr(layer,'activation_fn'):
                module=layer.activation_fn

            modules.append((f"layer_{i}", module))

        return modules

    def _mlp_hook(self, name, store):
        def hook(module, inputs, output):
            store[name] = output.detach()
        return hook

    def compute_mlp_rms_change(
        self,
        dataloader: DataLoader,
        input_type: str,
        device: str | None = None
    ):

        if device is None:
            device = next(self.base_model.parameters()).device

        layers = self._get_layers(self.base_model)
        num_layers = len(layers)
        hidden_size = self._get_intermediate_size(self.base_model, layers[0])

        running = torch.zeros(num_layers, hidden_size, device=device)
        total_tokens = 0

        base_modules = self._get_mlp_modules(self.base_model)
        train_modules = self._get_mlp_modules(self.trained_model)

        for batch in tqdm(dataloader, desc=f"MLP RMS change ({input_type})"):
            input_ids = batch[f"{input_type}_input_ids"].to(device)
            attention_mask = batch[f"{input_type}_attention_mask"].to(device)

            base_out = self._collect_outputs(
                self.base_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in base_modules],
            )

            train_out = self._collect_outputs(
                self.trained_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in train_modules],
            )

            total_tokens += attention_mask.sum().item()

            for i in range(num_layers):
                b = base_out[f"layer_{i}"]
                t = train_out[f"layer_{i}"]

                diff_sq = (t - b) ** 2
                mask = attention_mask.unsqueeze(-1)

                diff_sq = diff_sq * mask
                running[i] += diff_sq.sum(dim=(0, 1))

        return (running / total_tokens).sqrt()
    
    def compute_mlp_activation(
            self,
        dataloader: DataLoader,
        input_type: str,
        device: str | None = None
    ):
        if device is None:
            device = next(self.trained_model.parameters()).device

        layers = self._get_layers(self.trained_model)
        num_layers = len(layers)
        hidden_size = self._get_intermediate_size(self.trained_model, layers[0])

        running = torch.zeros(num_layers, hidden_size, device=device)
        total_tokens = 0

        train_modules = self._get_mlp_modules(self.trained_model)

        for batch in tqdm(dataloader, desc=f"MLP absolute ({input_type})"):
            input_ids = batch[f"{input_type}_input_ids"].to(device)
            attention_mask = batch[f"{input_type}_attention_mask"].to(device)


            train_out = self._collect_outputs(
                self.trained_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in train_modules],
            )

            total_tokens += attention_mask.sum().item()

            for i in range(num_layers):
                act = train_out[f"layer_{i}"]
                mask = attention_mask.unsqueeze(-1)

                act = act * mask
                running[i] += act.sum(dim=(0, 1))

        return running / total_tokens
    def compute_mlp_activation_prob(
                self,
        dataloader: DataLoader,
        input_type: str,
        device: str | None = None
    ):
        if device is None:
            device = next(self.trained_model.parameters()).device

        layers = self._get_layers(self.trained_model)
        num_layers = len(layers)
        hidden_size = self._get_intermediate_size(self.trained_model, layers[0])

        running = torch.zeros(num_layers, hidden_size, device=device)
        total_tokens = 0

        train_modules = self._get_mlp_modules(self.trained_model)

        for batch in tqdm(dataloader, desc=f"MLP absolute ({input_type})"):
            input_ids = batch[f"{input_type}_input_ids"].to(device)
            attention_mask = batch[f"{input_type}_attention_mask"].to(device)


            train_out = self._collect_outputs(
                self.trained_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in train_modules],
            )

            total_tokens += attention_mask.sum().item()

            for i in range(num_layers):
                act = train_out[f"layer_{i}"]
                mask = attention_mask.unsqueeze(-1)

                act = act * mask
                running[i] += (act>0).sum(dim=(0, 1))

        return running / total_tokens

    def compute_mlp_scores(    self,
        dataloader: DataLoader,
        input_type: str,
        device: str | None = None
    ):
        if device is None:
            device = next(self.trained_model.parameters()).device

        layers = self._get_layers(self.trained_model)
        num_layers = len(layers)
        hidden_size = self._get_intermediate_size(self.trained_model, layers[0])

        running = torch.zeros(num_layers, hidden_size, device=device)
        running_diffrence = torch.zeros(num_layers, hidden_size, device=device)
        running_prob = torch.zeros(num_layers, hidden_size, device=device)


        total_tokens = 0

        base_modules = self._get_mlp_modules(self.base_model)
        train_modules = self._get_mlp_modules(self.trained_model)

        for batch in tqdm(dataloader, desc=f"MLP absolute and change ({input_type})"):
            input_ids = batch[f"{input_type}_input_ids"].to(device)
            attention_mask = batch[f"{input_type}_attention_mask"].to(device)

            base_out = self._collect_outputs(
                self.base_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in base_modules],
            )


            train_out = self._collect_outputs(
                self.trained_model,
                input_ids,
                [(n, m, self._mlp_hook) for n, m in train_modules],
            )

            total_tokens += attention_mask.sum().item()

            for i in range(num_layers):

                b = base_out[f"layer_{i}"]
                t = train_out[f"layer_{i}"]

                diff_sq = (t - b) ** 2

                mask = attention_mask.unsqueeze(-1)

                diff_sq = diff_sq * mask
                t=t*mask
                running_diffrence[i] += diff_sq.sum(dim=(0, 1))
                running[i]+=t.sum(dim=(0, 1))
                running_prob[i]+=(t>0).sum(dim=(0, 1))

        return running_diffrence / total_tokens,running / total_tokens,running_prob/total_tokens
