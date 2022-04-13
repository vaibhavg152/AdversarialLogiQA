#!/usr/bin/env bash
export RECLOR_DIR=/scratch/vm2241/DAGN/reclor_data
export LOGIQA_DIR=/scratch/vm2241/DAGN/logiqa_data
export TASK_NAME=logiqa
export MODEL_DIR=roberta-large
export MODEL_TYPE=DAGN
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN
export SAVE_DIR=dagn

CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
    --disable_tqdm \
    --task_name $TASK_NAME \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_DIR \
    --init_weights \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $LOGIQA_DIR \
    --graph_building_block_version $GRAPH_VERSION \
    --data_processing_version $DATA_PROCESSING_VERSION \
    --model_version $MODEL_VERSION \
    --merge_type 4 \
    --gnn_version $GNN_VERSION \
    --use_gcn \
    --gcn_steps 2 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --roberta_lr 5e-6 \
    --gcn_lr 5e-6 \
    --proj_lr 5e-6 \
    --num_train_epochs 10 \
    --output_dir /scratch/vm2241/DAGN/Checkpoints/$TASK_NAME/${SAVE_DIR} \
    --logging_steps 200 \
    --save_steps 800 \
    --adam_epsilon 1e-6 \
    --overwrite_output_dir
    --weight_decay 0.01
