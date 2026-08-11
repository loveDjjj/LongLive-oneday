# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer
from utils.config import normalize_config, validate_sla_cag_training_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for checkpoints and evaluation artifacts")
    parser.add_argument("--metrics-path", type=str, default="", help="JSONL metrics output path")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from the latest checkpoint in the output directory")

    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    validate_sla_cag_training_config(config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    output_dir = args.output_dir
    if not output_dir:
        parser.error("--output-dir is required")

    config_name = Path(output_dir).name
    config.config_name = config_name
    config.output_dir = output_dir
    config.metrics_path = args.metrics_path
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume

    trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
