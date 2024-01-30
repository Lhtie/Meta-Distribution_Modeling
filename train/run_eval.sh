epoch=$1
margin=${2:-"1.0"}

python3 -u eval.py \
    --model_type=gpt2 \
    --model_name_or_path=/root/autodl-tmp/commongen_scorer_prob/epoch-${epoch} \
    --eval_model_path=/root/autodl-tmp/commongen_pythia-160m \
    --eval_data_input_file=../data/commongen_data/dev_ipt.txt \
    --eval_data_output_file=../data/commongen_data/dev_opt.txt \
    --per_device_eval_batch_size 16 \
    --line_by_line \
    --modeling=prob \
    --geval \
    --geval_file=.cache/gpt4_cp_detailed.json
    # --spice \
    # --spice_ipt_file=.cache/sent_scores-scorer_prob-epoch-${epoch}-commongen_pythia-160m.json \
    # --spice_res_file=.cache/epoch-${epoch}.json
