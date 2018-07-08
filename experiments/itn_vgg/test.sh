#!/bin/bash
use_gpu=0

if [ "$1" != "" ]
then
    use_gpu=$1
fi

export CUDA_VISIBLE_DEVICES=${use_gpu};

threshold=0.9

if [ ! -d './log' ]; then
    mkdir ./log
fi
if [ ! -d './output' ]; then
    mkdir ./output
fi

LOG_FILE="./log/test_`date +'%m%d'`.txt"
#exec &> >(tee -a "$LOG_FILE") #all output in this shell will log to LOG_FILE
printf "\n\n\n\n\n\n\n\nTest Start: `date +'%y%m%d_%H:%M:%S'`\n"

python ../../tools/test_roi_conv_net.py --cfg=./config.yml --weights=./weight/ic15_itn_resnet.caffemodel --thres=${threshold} --overwrite=0
