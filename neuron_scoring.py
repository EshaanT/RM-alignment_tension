"""Script for Evaluating Model."""
from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace

from torch.utils.data import DataLoader

from RM_interpretation.neural_score import ChangeAnalyzer, NeuronScoringDataset
from RM_interpretation.rm_model import get_model


def main(args:Namespace)->None:
    """Main function for Evaluation.

    Args:
        args (Namespace): arguments for main function.
    """
    device=args.device_id
    print("Device Index",device)

    trained_model,tokenizer=get_model(args.model_name,args.rm_type,args.seed,False)
    print("Moving Model to GPU",device)
    trained_model.to(f'cuda:{device}')

    tokenizer.padding_side = "right"

    trained_model.config.pad_token_id = tokenizer.pad_token_id

    dataset = NeuronScoringDataset(args.rm_type, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    if 'change' in args.scoring:
        base_model,_=get_model(args.model_name,args.rm_type,args.seed,True)
        base_model.to(f'cuda:{device}')
        base_model.config.pad_token_id= tokenizer.pad_token_id
    else:
        base_model=None

    catcher = ChangeAnalyzer(base_model,trained_model)

    for input_type in ['chosen','rejected']:

        if args.scoring=='activation_change':
            activation_per_neuron = catcher.compute_mlp_rms_change(loader,input_type,f'cuda:{device}')
        elif args.scoring=="activation_absolute":
            activation_per_neuron = catcher.compute_mlp_activation(loader,input_type,f'cuda:{device}')
        elif args.scoring=='activation_prob':
            activation_per_neuron= catcher.compute_mlp_activation_prob(loader,input_type,f'cuda:{device}')
        elif args.scoring=='all_change_prob_and_absolute':
            activation_change,activation_per_neuron,activation_prob=catcher.compute_mlp_scores(loader,input_type,f'cuda:{device}')
            activation_change = [t.detach().cpu().tolist() for t in activation_change]
            activation_prob=[t.detach().cpu().tolist() for t in activation_prob]
        else:
            raise ValueError("Wrong neuron score setup.")
        activation_per_neuron = [t.detach().cpu().tolist() for t in activation_per_neuron]

        save_path=f'./save/results/{args.model_name}_{args.rm_type}_{args.seed}'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.scoring!='all_change_prob_and_absolute':

            json_path=f'{save_path}/{input_type}_{args.scoring}.json'

            with open(json_path, "w") as f:
                json.dump(activation_per_neuron, f, indent=4)
        else:
            json_path=f'{save_path}/{input_type}_activation_absolute.json'

            with open(json_path, "w") as f:
                json.dump(activation_per_neuron, f, indent=4)
            json_path=f'{save_path}/{input_type}_activation_change.json'

            with open(json_path, "w") as f:
                json.dump(activation_change, f, indent=4)

            json_path=f'{save_path}/{input_type}_activation_prob.json'

            with open(json_path, "w") as f:
                json.dump(activation_prob, f, indent=4)


if __name__ == "__main__":

    parser=ArgumentParser()

    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Name of the model to benchmark.",
        )
    parser.add_argument(
        "--device_id",
        type=str,
        default=0,
        help="Device ID to use for benchmarking. If None, uses all device."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model eval."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max length of string."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed value for model training."
    )
    parser.add_argument(
        "--rm_type",
        type=str,
        help="Gives type of reward model to eval.",
        choices=['helpful','harmless','helpful-harmless']
        )
    parser.add_argument(
        "--scoring",
        type=str,
        help="What component to score.",
        default='attentions'
        )

    args=parser.parse_args()


    main(args)
