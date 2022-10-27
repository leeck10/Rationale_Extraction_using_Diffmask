#!/bin/bash
model=
dataset=nsmc
train_filename=
val_filename=
num_labels=2
save_name=

python run_sst.py \
    --gpu 0 --seed 0 \
    --model $model \
    --dataset $dataset \
    --train_filename $train_filename \
    --val_filename $val_filename \
    --num_labels $num_labels \
    --epochs 25 --batch_size 64 --learning_rate 3e-5 \
    --save_name $save_name
