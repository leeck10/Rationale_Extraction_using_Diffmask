#!/bin/bash

python run_sst.py \
    --gpu 0 --seed 0 \
    --model bert-base-uncased \
    --dataset coco \
    --train_filename '' \
    --val_filename '' \
    --num_labels 3 \
    --epochs 25 --batch_size 64 --learning_rate 3e-5 \
    --save_name ''

    # --train_rationale 
    # --val_rationale 

    # --token_cls