#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 -C K80 --time=10:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=20GB
#SBATCH -c 5

python src/MHQA_final_evaluater.py \
    --in_path data/dev_v2.json \
    --out_path logs/rst.dev.$1\.txt \
    --model_prefix logs/MHQA.$1 \
    --word_vec_path data/vectors.txt.st.top100k 

