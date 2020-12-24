export DATA_DIR=/env/comprehension_contest/dureader_robust-data
export TASK_NAME=DuReader
export CHECKPOINT=checkpoint-5000
export CUDA_VISIBLE_DEVICES=1,2,3
export DATASET_FILE=dev-v1.1.json
export MODEL_NAME=roberta_large_brightmart_output
export PRED_FILE=/env/comprehension_contest/$MODEL_NAME/$TASK_NAME/predictions_.json
python main.py --model_type bert --model_name_or_path $MODEL_NAME/$TASK_NAME/$CHECKPOINT --do_eval --do_lower_case --max_seq_length 256 --data_dir=$DATA_DIR  --output_dir ./$MODEL_NAME/$TASK_NAME/ --overwrite_cache --max_answer_length 30

python evaluate.py $DATA_DIR/$DATASET_FILE $PRED_FILE
