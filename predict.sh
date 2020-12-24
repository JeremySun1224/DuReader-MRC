export DATA_DIR=/env/comprehension_contest/dureader_robust-data
export TASK_NAME=DuReader
export CHECKPOINT=checkpoint-4000
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATASET_FILE=dev-v1.1.json
export OUTPUT_DIR=roberta_large_adv_output
export PRED_FILE=/env/comprehension_contest/$OUTPUT_DIR/$TASK_NAME/predictions_.json
python main.py --model_type bert --model_name_or_path $OUTPUT_DIR/$TASK_NAME/$CHECKPOINT --do_eval --do_lower_case --max_seq_length 256 --data_dir=$DATA_DIR  --output_dir ./$OUTPUT_DIR/$TASK_NAME/ --overwrite_output_dir --overwrite_cache --max_answer_length 30

# python evaluate.py $DATA_DIR/$DATASET_FILE $PRED_FILE 
