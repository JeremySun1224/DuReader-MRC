export PYTHONIOENCODING=utf-8
export DATA_DIR=/content/dureader/data
export TASK_NAME=DuReader
export CUDA_VISIBLE_DEVICES=0
export OUTPUT_DIR=/content/dureader/output/chinese_wwm_pytorch
export MODEL_NAME_OR_PATH=/content/dureader/pretrained_model/chinese_wwm_pytorch
python /content/dureader/main.py --model_type bert \
                                 --model_name_or_path $MODEL_NAME_OR_PATH \
                                 --do_train --data_dir=$DATA_DIR \
                                 --max_seq_length 256 \
                                 --per_gpu_train_batch_size 16 \
                                 --per_gpu_eval_batch_size 512 \
                                 --gradient_accumulation_steps 4 \
                                 --do_lower_case \
                                 --learning_rate 2e-5 \
                                 --num_train_epochs 3.0 \
                                 --output_dir $OUTPUT_DIR/$TASK_NAME/ \
                                 --save_steps 1000 \
                                 --overwrite_output_dir \
                                 --overwrite_cache \
                                 --max_answer_length 30 \
