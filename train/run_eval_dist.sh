scale=$1

python3 -u eval_dist.py \
    --model_type=gpt2 \
    --model_name_or_path=/root/autodl-tmp/scorer_prob/epoch-0 \
    --eval_model_path=/root/autodl-tmp/commongen_pythia-${scale}m \
    --eval_data_input_file=../data/commongen_data/dev_ipt.txt \
    --eval_data_output_file=../data/commongen_data/dev_opt.txt \
    --per_device_eval_batch_size 16 \
    --line_by_line \
    --modeling=prob \
    --sample_k 8