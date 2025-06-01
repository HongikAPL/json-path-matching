#!/bin/bash
# model_name_or_path
# BERT: bert-base-uncased
# Roberta: roberta-base
# VarCLR: ./varclr
# generator_name
# BERT: distillbert-base-uncased
# Roberta, VarCLR: distrilroberta-base

LR=7e-6
MASK=0.30
LAMBDA=0.005

python train.py \
    --model_name_or_path ./varclr \
    --generator_name distilroberta-base \
    --train_file data/train.txt \
    --output_dir varclr-finetuned/output \
    --cache_dir varclr-finetuned/cache \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_auc \
    --load_best_model_at_end True\
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir varclr-finetuned/log \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight $LAMBDA \
    --masking_ratio $MASK
    # 분산학습
    # --fp16 --masking_ratio $MASK
