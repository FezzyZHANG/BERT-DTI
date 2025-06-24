#!/bin/bash

# Get the absolute path of the current script
MODEL_DIR=$(dirname $(realpath $0))

CONFIG_PATH=$MODEL_DIR/config.yaml
OUTPUT_DIR=$1
SPLIT_PATH=$2
TENSORBOARD_LOGDIR=$3
SEED=$4
TRAIN_PATH=$SPLIT_PATH/train.parquet
VAL_PATH=$SPLIT_PATH/val.parquet
TEST_PATH=$SPLIT_PATH/test.parquet

python3 $MODEL_DIR/main.py \
    --config_path $CONFIG_PATH \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --test_path $TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --tensorboard_logdir $TENSORBOARD_LOGDIR \
    --seed $SEED
