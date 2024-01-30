modeling=${1:-"prob"}
margin=${2:-"6.4"}
lr=${3:-"5e-5"}

python3 -u train_scorer.py \
    --output_dir=/root/autodl-tmp/commongen_scorer_${modeling} \
    --model_type=gpt2 \
    --model_name_or_path=/root/autodl-tmp/models/gpt2 \
    --do_train \
    --train_data_input_file=../data/commongen_data/mini/train_ipt.txt \
    --train_data_output_file=../data/commongen_data/mini/train_opt.txt \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_sample_batch_size 16 \
    --line_by_line \
    --learning_rate ${lr} \
    --num_train_epochs 3 \
    --sample_k 8 \
    --margin ${margin} \
    --modeling=${modeling}