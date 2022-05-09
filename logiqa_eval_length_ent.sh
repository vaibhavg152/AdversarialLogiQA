#!/usr/bin/env bash
export LOGIQA_DIR=Shakespeare_logiqa
export TASK_NAME=logiqa
export MODEL_TYPE=DAGN
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN
export SAVE_DIR=dagn
export MODEL_DIR=CheckpointsEntity/$TASK_NAME/$SAVE_DIR/checkpoint-6400/

source /ext3/miniconda3/etc/profile.d/conda.sh
for i in `seq 5 5 80`
do
	echo $i
	conda activate data
	python adding_irrelevant_info.py $i
	conda activate dagn
	export LOGIQA_DIR=Shakespeare_logiqa
	rm -r $LOGIQA_DIR/cached_data/
	echo "running eval"
	CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
	    --disable_tqdm \
	    --task_name $TASK_NAME \
	    --tokenizer_name roberta-large \
	    --model_type $MODEL_TYPE \
	    --model_name_or_path $MODEL_DIR \
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
	    --eval_steps 1000 \
	    --max_seq_length 256 \
	    --per_device_eval_batch_size 4 \
	    --gradient_accumulation_steps 4 \
	    --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
	    --numnet_drop 0.2 > results/out_ent_shakes_$i.txt 2>&1
	export LOGIQA_DIR=Brown_logiqa
	rm -r $LOGIQA_DIR/cached_data/
        CUDA_VISIBLE_DEVICES=0 python3 run_multiple_choice.py \
            --disable_tqdm \
            --task_name $TASK_NAME \
            --tokenizer_name roberta-large \
            --model_type $MODEL_TYPE \
            --model_name_or_path $MODEL_DIR \
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
            --eval_steps 1000 \
            --max_seq_length 256 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
            --numnet_drop 0.2 > results/out_ent_brown_$i.txt 2>&1

done
