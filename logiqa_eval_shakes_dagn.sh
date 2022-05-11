#!/usr/bin/env bash
export RECLOR_DIR=Shakespeare_logiqa
export TASK_NAME=logiqa
export MODEL_TYPE=DAGN
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN
export SAVE_DIR=dagn
export MODEL_DIR=Checkpoints/$TASK_NAME/$SAVE_DIR/checkpoint-6400/

CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
    --disable_tqdm \
    --task_name $TASK_NAME \
    --tokenizer_name roberta-large \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_DIR \
    --do_predict \
    --data_dir $RECLOR_DIR \
    --graph_building_block_version $GRAPH_VERSION \
    --data_processing_version $DATA_PROCESSING_VERSION \
    --model_version $MODEL_VERSION \
    --merge_type 4 \
    --gnn_version $GNN_VERSION \
    --use_gcn \
    --gcn_steps 2 \
    --eval_steps 1000 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
    --numnet_drop 0.2
