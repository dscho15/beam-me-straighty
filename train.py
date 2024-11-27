from typing import cast, Optional, Sequence
from pathlib import Path
import argparse
import torch
from frogbox import read_json_config, SupervisedPipeline, SupervisedConfig
from einops import rearrange


def parse_arguments(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, default="configs/example.json"
    )
    parser.add_argument(
        "-d", "--device", type=torch.device, default=torch.device("cuda:0")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument(
        "--logging",
        type=str,
        choices=["online", "offline"],
        default="online",
    )
    parser.add_argument("--wandb-id", type=str, required=False)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments()
    config = cast(SupervisedConfig, read_json_config(args.config))

    def input_transform(x, y):
        return x, rearrange(y, "b k -> (b k)") # collapse sequence into batch

    def model_transform(output):
        return rearrange(output, "b k n -> (b k) n") # collapse sequence into batch

    pipeline = SupervisedPipeline(
        config=config,
        device=args.device,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group,
        trainer_input_transform=input_transform,
        trainer_model_transform=model_transform,
        evaluator_input_transform=input_transform,
        evaluator_model_transform=model_transform,
    )

    params = sum(p.numel() for p in pipeline.model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {params:.2f}M")

    pipeline.run()
