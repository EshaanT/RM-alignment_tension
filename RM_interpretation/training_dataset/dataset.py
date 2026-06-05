"""Contains code relavent for importing datasets."""
from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset


def rm_traing_dataset(dataset_type:str,seed:int=0)->tuple(Dataset,Dataset):
    """Function to get reward model traning dataset.

    Args:
        dataset_type (str): Sets what type of  daatset to be returned. Either, helpful, harmless, or helpful-harmless
        seed (int, optional): seed value for the run. Defaults to 0.

    Returns:
        tuple: The traing and testing dataset
    """
    if dataset_type=='helpful':
        dataset_help = load_dataset("Anthropic/hh-rlhf",data_dir="helpful-base")
        train_help1=dataset_help['train']
        test_help1=dataset_help['test']

        dataset_help=load_dataset("Anthropic/hh-rlhf",data_dir="helpful-online")
        train_help2=dataset_help['train']
        test_help2=dataset_help['test']

        dataset_help=load_dataset("Anthropic/hh-rlhf",data_dir="helpful-rejection-sampled")
        train_help3=dataset_help['train']
        test_help3=dataset_help['test']

        train_help=concatenate_datasets([train_help1,train_help2,train_help3]).shuffle(seed)
        test_help=concatenate_datasets([test_help1,test_help2,test_help3])

        return train_help,test_help

    elif dataset_type=='harmless':
        dataset_harm = load_dataset("Anthropic/hh-rlhf",data_dir="harmless-base")
        train_harm=dataset_harm['train'].shuffle(seed)
        test_harm=dataset_harm['test']

        return train_harm,test_harm

    elif dataset_type=='helpful-harmless':

        dataset_hh = load_dataset("Anthropic/hh-rlhf")
        train_hh=dataset_hh['train']
        test_hh=dataset_hh['test']

        #dataset_harm = load_dataset("Anthropic/hh-rlhf",data_dir="harmless-base")
        #train_harm=dataset_harm['train']
        #test_harm=dataset_harm['test']

        #train_hh=concatenate_datasets([train_help,train_harm]).shuffle(seed)
        #test_hh=concatenate_datasets([test_help,test_harm])

        return train_hh,test_hh
    else:
        raise ValueError("Incorect ",dataset_type," for RM traning.")



