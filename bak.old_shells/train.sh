#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:2 -C K80 --time=5-00:00:00 -J base_twoL --output=train.out --error=train.err
#SBATCH --mem=80GB
#SBATCH -c 5

python src/MHQA_trainer.py --config_path config.json

