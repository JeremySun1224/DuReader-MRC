export DATA_DIR=/content/dureader/data
export TASK_NAME=DuReader
export CHECKPOINT=checkpoint-2000
export CUDA_VISIBLE_DEVICES=0
export DATASET_FILE=dev-v1.1.json
export MODEL_NAME=dureader/output/chinese_roberta_wwm_ext_pytorch
export PRED_FILE=$MODEL_NAME/$TASK_NAME/predictions_.json
python /content/dureader/main.py --model_type bert \
                                 --model_name_or_path $MODEL_NAME/$TASK_NAME/$CHECKPOINT \
                                 --do_eval \
                                 --do_lower_case \
                                 --per_gpu_eval_batch_size 512 \
                                 --max_seq_length 256 \
                                 --data_dir=$DATA_DIR \
                                 --output_dir ./$MODEL_NAME/$TASK_NAME/ \
                                 --overwrite_cache \
                                 --max_answer_length 30 \

python /content/dureader/evaluate.py $DATA_DIR/$DATASET_FILE $PRED_FILE