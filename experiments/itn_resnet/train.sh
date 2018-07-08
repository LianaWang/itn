#!/bin/bash
use_gpu=1
if [ "$1" != "" ]
then
    use_gpu=$1
fi
export CUDA_VISIBLE_DEVICES=${use_gpu};

if [ ! -d './log' ]; then
    mkdir ./log
fi

LOG_FILE="./log/train_`date +'%m%d'`.txt"
exec 2> >(tee -a "$LOG_FILE") #all output in this shell will log to LOG_FILE
printf "\n\n\n\n\n\n\n\nTraining Start: `date +'%y%m%d_%H:%M:%S'`\n"

python ../../tools/train_roi_conv_net.py  --cfg=./config.yml --weights=../../data/backbone_models/ResNet-50-model.caffemodel
