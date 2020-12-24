export DATA_DIR=/env/comprehension_contest/dureader_robust-data
export TASK_NAME=DuReader
# export CHECKPOINT=checkpoint-3000
export CUDA_VISIBLE_DEVICES=1,2,3
export DATASET_FILE=dev-v1.1.json
# export MODEL_NAME=roberta_large_adv_output
export PRED_FILE=/env/comprehension_contest/ensemble_results/predictions_.json
python ensemble.py --model_type bert --do_eval --do_lower_case --max_seq_length 256 --data_dir=$DATA_DIR  --output_dir ./ensemble_results --max_answer_length 30 --overwrite_cache --overwrite_output --overwrite_output_dir

python evaluate.py $DATA_DIR/$DATASET_FILE $PRED_FILE
