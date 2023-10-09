# pitfuls_of_multi_choice

We find that even minor alterations in option placements or symbols can modify an LLM's response. This inconsistency suggests that multiple-choice evaluations might not fully capture an LLM's capabilities, and some decisions might stem from specific lexical patterns rather than genuine understanding. To investigate further, we devised three methods to analyze these inconsistencies and introduced a more stringent evaluation criterion by integrating various methods.

# An usage example for C-eval shuffle-location

ckpt_path=$1                 # model path \
token_path=$2                # tokenizer path \
cards=$3                     # which gpu need to use \
name=$4                      # output dir name \

# usage of common version without vllm
CUDA_VISIBLE_DEVICES=$cards python C-eval/bash/eval_my_vllm_disturb_v1.py \
--ckpt_dir $ckpt_path \
--token_dir $token_path \
--out_file out_dir_path/$name \
--temp_dir temp_dir_path/$name \
--batch_size 8 \
--few_shot True \
--cot False > log_dir_path/$name.log

# usage of common version without vllm
CUDA_VISIBLE_DEVICES=$cards python C-eval/bash/eval_my_disturb_v1.py \
--ckpt_dir $ckpt_path \
--token_dir $token_path \
--out_file out_dir_path/$name \
--temp_dir temp_dir_path/$name \
--batch_size 8 \
--few_shot True \
--cot False \
--model_type $name > log_dir_path/$name.log

# We also provide stricter evaluation metric. 
# For example, we can see whether model consistenetly answer correct in origin, shuffle-loc, and replace-signal.

python stricter/bash/shuffle_and_replace.py \
--shuffle_temp_dir your_shuffle_output_temp_dir \
--replace_temp_dir your_replace_output_temp_dir \
--output_filename outfile

