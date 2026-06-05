"""Contains code relavent for importing datasets."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.utils.data
from datasets import concatenate_datasets, load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class NeuronScoringDataset(torch.utils.data.Dataset):
    """Class for neural scroing dataset."""
    def __init__(self, dataset_type: str, tokenizer: PreTrainedTokenizer,max_length: int=1024)->None:
        """NeuronScoringDataset initialisation.

        Args:
            dataset_type (str): Type of dataset to load.
            tokenizer (PreTrainedTokenizer): model's tokeniser.

        Raises:
            ValueError: If invalid dataset_type.
        """
        if dataset_type == "helpful":
            dataset_help = load_dataset("Anthropic/hh-rlhf",data_dir="helpful-base")
            train_help1=dataset_help['train']

            dataset_help=load_dataset("Anthropic/hh-rlhf",data_dir="helpful-online")
            train_help2=dataset_help['train']

            dataset_help=load_dataset("Anthropic/hh-rlhf",data_dir="helpful-rejection-sampled")
            train_help3=dataset_help['train']

            dataset=concatenate_datasets([train_help1,train_help2,train_help3])
        elif dataset_type == "harmless":
            dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")["train"]
        elif dataset_type=='helpful-harmless':
            dataset = load_dataset("Anthropic/hh-rlhf")['train']

        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        self.data = dataset
        self.tokenizer = tokenizer
        self.max_length=max_length

    def __len__(self)->int:
        """Length of dataset."""
        return len(self.data)

    def __getitem__(self, idx:int)->dict:
        """Returns tokenized datapoint for dataloader."""
        sample = self.data[idx]

        chosen = self.tokenizer(sample["chosen"], padding="max_length",
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt")
        rejected = self.tokenizer(sample["rejected"], padding="max_length",
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt")

        return {
            "chosen_input_ids": chosen["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
        }
