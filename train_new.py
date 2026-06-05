"""Script to train the Reward Models."""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import transformers
from trl import RewardConfig, RewardTrainer

from RM_interpretation.rm_model import MaskingNeurons, get_model
from RM_interpretation.training_dataset import rm_traing_dataset

print(torch.cuda.is_available())
print(torch.cuda.is_bf16_supported())


def initialize_reward_head(model)->None:
    """Initialize reward head weights."""
    if hasattr(model, "score"):
        nn.init.normal_(model.score.weight, std=0.02)
        if model.score.bias is not None:
            nn.init.zeros_(model.score.bias)

    if hasattr(model, "classifier"):
        nn.init.normal_(model.classifier.weight, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)


def main(args: argparse.Namespace) -> None:
    """Main function for training."""
    if not os.path.exists('./save/logging'):
        os.makedirs('./save/logging')

    if not os.path.exists('./save/models'):
        os.makedirs('./save/models')

    run_name = args.model_name + '_' + args.dataset_type + '_' + str(args.seed)

    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    print("--Loading Dataset--")
    traing_dataset, testing_dataset = rm_traing_dataset(
        dataset_type=args.dataset_type,
        seed=args.seed
    )

    print("--Loading Model--")
    model, tokenizer = get_model(args.model_name,args.dataset_type,args.seed,True)
    if args.masking:
        masker=MaskingNeurons(args,args.dataset_type,args.mask_p)
        masker.apply_mask(model)
        run_name+='_masked_top_'+str(args.mask_p)

    if args.pruning:
        masker=MaskingNeurons(args,args.dataset_type,args.mask_p)
        masker.apply_pruning(model)
        run_name+='_pruning_top_'+str(args.mask_p)

    model.train()

    # initialize reward head
    initialize_reward_head(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--Making Reward Config--")

    training_args = RewardConfig(
        output_dir=f'./save/models/{run_name}',

        per_device_eval_batch_size=args.batch_size,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=f'./save/logging/{run_name}',
        logging_strategy='epoch',
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=traing_dataset,
        eval_dataset=testing_dataset,
    )

    print("--Training Begins--")
    trainer.train()

    print("--Saving Model--")
    trainer.save_model(f'./save/models/{run_name}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Name of the model to benchmark.",
    )

    parser.add_argument(
        "--device_id",
        type=str,
        default=None,
        help="Device ID to use for benchmarking."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for model training."
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
        help="Epochs to train."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed value for model training."
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        help="Type of reward model to train.",
        choices=['helpful', 'harmless', 'helpful-harmless']
    )

    parser.add_argument("--masking", action="store_true",
                    help="masks neurons if passed")
    parser.add_argument("--pruning", action="store_true",
                    help="masks neurons if passed")
    parser.add_argument(
        "--mask_p",
        type=float,
        default=99.0,
        help="Learning rate."
    )

    args = parser.parse_args()

    main(args)
