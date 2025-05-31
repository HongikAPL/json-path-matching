#!/bin/bash
# BERT: distillbert-base-uncased, Roberta VarCLR: distrilroberta-base

LR=7e-6
MASK=0.30
LAMBDA=0.005

python train.py \
    --model_name_or_path ./varclr \
    --generator_name distilroberta-base \
    --train_file data/train.txt \
    --output_dir ./output \
    --cache_dir ./cache \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir ./log \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight $LAMBDA \
    --masking_ratio $MASK
    # 분산학습
    # --fp16 --masking_ratio $MASK
