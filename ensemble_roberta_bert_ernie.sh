export DATA_DIR=/content/dureader/data
export TASK_NAME=DuReader
# export CHECKPOINT=checkpoint-3000
export CUDA_VISIBLE_DEVICES=0
export DATASET_FILE=dev-v1.1.json
# export MODEL_NAME=roberta_large_adv_output
export PRED_FILE=/content/dureader/ensemble_roberta_bert_ernie_results/predictions_.json
python /content/dureader/ensemble.py --model_type bert \
                                     --do_eval \
                                     --do_lower_case \
                                     --max_seq_length 256 \
                                     --data_dir=$DATA_DIR \
                                     --per_gpu_eval_batch_size 512 \
                                     --output_dir /content/dureader/ensemble_roberta_bert_ernie_results \
                                     --max_answer_length 30 \
                                     --overwrite_cache \
                                     --overwrite_output \
                                     --overwrite_output_dir \
                                    #  --no_cuda \

# python /content/dureader/evaluate.py $DATA_DIR/$DATASET_FILE $PRED_FILE
