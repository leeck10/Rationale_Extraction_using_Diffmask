#!/bin/bash
model=
dataset=nsmc
train_filename=
val_filename=
num_labels=2
save_name=
model_path=
gate=input

python run_sst_diffmask.py \
    --gpu 0 --seed 0 \
    --model $model \
    --model_path $model_path \
    --dataset $dataset \
    --train_filename $train_filename \
    --val_filename $val_filename \
    --num_labels $num_labels \
    --gate $gate \
    --epochs 25 --batch_size 64 --learning_rate 3e-5 \
    --save_name $save_name

