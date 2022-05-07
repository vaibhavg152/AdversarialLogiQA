#!/usr/bin/env bash
export RECLOR_DIR=/scratch/vm2241/AdversarialLogiQA/reclor_data
export LOGIQA_DIR=/scratch/vm2241/AdversarialLogiQA/logiqa_data
export TASK_NAME=logiqa
export MODEL_TYPE=DAGN
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN
export SAVE_DIR=dagn
export MODEL_DIR=/scratch/vm2241/AdversarialLogiQA/CheckpointsEntity/$TASK_NAME/$SAVE_DIR/checkpoint-6400/

CUDA_VISIBLE_DEVICES=0 python3 -u run_multiple_choice.py --data_dir $LOGIQA_DIR --output_dir /scratch/vm2241/AdversarialLogiQA/Checkpoints/$TASK_NAME/${SAVE_DIR} \
    --disable_tqdm \
    --tokenizer_name roberta-large \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_DIR \
    --model_type $MODEL_TYPE \
    --do_eval \
    --do_predict \
    --graph_building_block_version $GRAPH_VERSION \
    --data_processing_version $DATA_PROCESSING_VERSION \
    --model_version $MODEL_VERSION \
    --merge_type 4 \
    --gnn_version $GNN_VERSION \
    --use_gcn \
    --gcn_steps 2 \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --overwrite_output_dir
