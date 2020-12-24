export TRAIN_FILE=dureader_robust-data/train.raw
export TEST_FILE=dureader_robust-data/dev.raw

python run_language_modeling.py \
    --output_dir=trained_models \
    --model_type=bert \
    --model_name_or_path=chinese_wwm_ext_pytorch_large/ \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --line_by_line \
    --eval_data_file=$TEST_FILE \
    --overwrite_cache \
    --mlm \
    --num_train_epochs=4.0 \
    --per_gpu_train_batch_size=2 \