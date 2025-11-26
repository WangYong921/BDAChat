#!/bin/bash

dataset=$1
model_path=$2

local_data_dir=$3
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Start building the command
cmd="python videollava/eval/eval.py \
    --dataset $dataset \
    --model_path $model_path \
    --local_data_dir $local_data_dir \
    --load_8bit \
    --prompt_strategy interleave \
    --chronological_prefix 
"


eval $cmd
