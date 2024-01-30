scale=$1

python3 -u run_language_modeling.py \
    --output_dir="/root/autodl-tmp/commongen_pythia-${scale}m/" \
    --model_type=GPTNeoX \
    --model_name_or_path="/root/autodl-tmp/models/pythia-${scale}m" \
    --do_train \
    --train_data_input_file=../data/commongen_data/train_ipt.txt \
    --train_data_output_file=../data/commongen_data/train_opt.txt \
    --do_eval \
    --eval_data_input_file=../data/commongen_data/dev_ipt.txt \
    --eval_data_output_file=../data/commongen_data/dev_opt.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --save_strategy steps \
    --save_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --line_by_line \
    --learning_rate 3e-6 \
    --num_train_epochs 5 \
    --overwrite_output_dir
