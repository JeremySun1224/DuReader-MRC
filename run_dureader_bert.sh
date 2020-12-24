export PYTHONIOENCODING=utf-8
export DATA_DIR=/env/comprehension_contest/dureader_robust-data
export TASK_NAME=DuReader
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OUTPUT_DIR=./bert_output
python main.py --model_type bert --model_name_or_path chinese_bert_wwm_ext_pytorch --do_train --data_dir=$DATA_DIR --max_seq_length 256 --per_gpu_train_batch_size 16 --do_lower_case --learning_rate 3e-5 --num_train_epochs 10.0 --output_dir $OUTPUT_DIR/$TASK_NAME/ --save_steps 1000 --overwrite_output_dir --overwrite_cache --max_answer_length 30
