import json
import os
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import random
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
import tensor_parallel as tp
import accelerate
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8",line_buffering=True)


TASKS = [
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]
choices_new = ['😩', '😰', '😶', '😲'] #v3
# choices_new = ['H', 'G', 'R', 'Q'] #v2
# choices_new = ['T', 'M', 'U', 'P'] #v1
# choices_new = ['🌈', '✊', '🌱', '🐱'] #v4

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def format_example_replace(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    ans_newlabel = choices_new[choices.index(df.iloc[idx, k + 1])]
    for j in range(k):
        prompt += "\n{}. {}".format(choices_new[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(ans_newlabel)
    return prompt, ans_newlabel

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def gen_prompt_replace(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example_replace(train_df, i)[0]
    return prompt

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir, token_dir, model_type):
    n_gpus = torch.cuda.device_count()
    # import pdb;pdb.set_trace()
    if 'llama' in token_dir or 'Llama' in token_dir:
        tokenizer = LlamaTokenizer.from_pretrained(token_dir,use_fast=False,padding_side="left",)
    else:
        tokenizer = AutoTokenizer.from_pretrained(token_dir, padding_side='left', trust_remote_code=True)

    try:
        if tokenizer.unk_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        elif 'gpt-j' in token_dir and tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        else:
            tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    except:
        pass
    if 'opt' in token_dir:
        tokenizer.pad_token_id = 0
        tokenizer.unk_token_id = 0
    if 'gpt-j' not in token_dir:
        tokenizer.bos_token_id = 1
    
    # import pdb;pdb.set_trace()
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'auto', torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer_replace(model, tokenizer, prompts, my_batch_size=8):
    batch_size = my_batch_size
    answers = []
    if 'Qwen' in args.token_dir:
        batch_size=1
    for batch_input in batch_split(prompts, batch_size):
        if batch_size==1:
            encode_inputs = tokenizer(batch_input, return_tensors="pt").to("cuda")
        else:
            encode_inputs = prepare_input(tokenizer, batch_input)
        if 'token_type_ids' in encode_inputs:
            del encode_inputs['token_type_ids']  #鸭嘴兽的多余值
        # import pdb;pdb.set_trace()
        if 'gpt' in args.token_dir or 'mpt' in args.token_dir:
            outputs = model.generate(**encode_inputs, max_new_tokens=2)
        else:
            outputs = model.generate(**encode_inputs, max_new_tokens=5)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        # import pdb;pdb.set_trace()
    answers = [answer.split('Answer:')[answer.count('Answer:')] for answer in answers]
    # import pdb;pdb.set_trace()
    return answers

def batch_infer(model, tokenizer, prompts, my_batch_size=8):
    batch_size = my_batch_size
    answers = []
    if 'Qwen' in args.token_dir:
        batch_size=1
    for batch_input in batch_split(prompts, batch_size):
        if batch_size==1:
            encode_inputs = tokenizer(batch_input, return_tensors="pt").to("cuda")
        else:
            encode_inputs = prepare_input(tokenizer, batch_input)
        if 'token_type_ids' in encode_inputs:
            del encode_inputs['token_type_ids']  #鸭嘴兽的多余值
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    # import pdb;pdb.set_trace()
    return answers

def eval(args, subject, dev_df, test_df, tokenizer, model, my_batch_size=8):
    cors = []
    records=[]
    print(subject)
    prompt_ends=[]
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        label = test_df.iloc[i, test_df.shape[1]-1]
        prompt_end, label = format_example_replace(test_df, i, include_answer=False)
        # import pdb;pdb.set_trace()
        prompt_ends.append(prompt_end)
        train_prompt = gen_prompt_replace(dev_df, subject, k)
        if args.zero_shot:
            train_prompt = "The following ia an multiple choice question (with answers) about {}.\n\n".format(format_subject(subject))
        prompt = train_prompt + prompt_end
        # import pdb;pdb.set_trace()
        while len(tokenizer.tokenize(prompt)) + 5> 2048: # bos token
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)

        records.append({'prompt':prompt, 'answer':label})
    # import pdb;pdb.set_trace()
    pred_answers = batch_infer_replace(model, tokenizer, [record['prompt'] for record in records], my_batch_size)
    gold_answers = [record['answer'] for record in records]
    cors_orig = []
    if args.compare==True:
        records=[]
        prompt_ends_orig=[]
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end_orig = format_example(test_df, i, include_answer=False)
            prompt_ends.append(prompt_end_orig)
            train_prompt_orig = gen_prompt(dev_df, subject, k)
            if args.zero_shot:
                train_prompt = "The following ia an multiple choice question (with answers) about {}.\n\n".format(format_subject(subject))
            prompt = train_prompt_orig + prompt_end_orig
            # import pdb;pdb.set_trace()
            while len(tokenizer.tokenize(prompt)) + 2> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)

            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})
            # import pdb;pdb.set_trace()
        pred_answers_orig = batch_infer(model, tokenizer, [record['prompt'] for record in records], my_batch_size)
        gold_answers_orig = [record['answer'] for record in records]
    if args.temp_dir!='':
        with open(args.temp_dir + '/%s'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers):
                try:
                    if args.compare==True:
                        f.write('Q: {}\nA_model: {}\nA_correct: {}\nA_model_orig: {}\nA_correct_orig: {}\n\n'.format(prompt_ends[idx], temp, gold_answers[idx], pred_answers_orig[idx], gold_answers_orig[idx]))
                    else:
                        f.write('Q: {}\nA_model: {}\nA_correct: {}\n\n'.format(prompt_ends[idx], temp, gold_answers[idx]))
                except:
                    # import pdb;pdb.set_trace()
                    f.write('error!\n\n')
    # print('!!!!')
    for pred, gold in zip(pred_answers, gold_answers):
        cor = gold in pred
        cors.append(cor)
    acc = np.mean(cors)
    cors = np.array(cors)
    # import pdb;pdb.set_trace()
    if args.compare==True:
        for pred, gold in zip(pred_answers_orig, gold_answers_orig):
            cor = pred == gold
            cors_orig.append(cor)
        acc_orig = np.mean(cors_orig)
        cors_orig = np.array(cors_orig)
        return cors, acc, cors_orig, acc_orig
    else:
        return cors, acc

def main(ckpt_dir: str, token_dir: str, model_type: str):
    output_filename = args.out_file
    if args.temp_dir!='':
        os.makedirs(args.temp_dir, exist_ok=True)
    model, tokenizer = load(ckpt_dir, token_dir, model_type)
    start_time = time.time()
    if args.compare==True:
        with open(output_filename, 'w', encoding='utf-8') as f:
            all_all_cors=[]
            all_all_cors_orig=[]
            all_all_both=[]         #两个都true才true
            for oneclass in args.classes:
                f.write(oneclass+'\n')
                all_cors = []
                all_cors_orig = []
                all_both=[]
                for subject in tqdm(args.new_categs[oneclass]):
                    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
                    test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
                    cors, acc, cors_orig, acc_orig = eval(args, subject, dev_df, test_df, tokenizer, model, args.batch_size)
                    both=[]
                    for i in range(len(cors)):
                        if cors[i]==True and cors_orig[i]==True:
                            both.append(True)
                        else:
                            both.append(False)
                    both_acc = np.mean(both)
                    both = np.array(both)
                    f.write("Average accuracy {:.3f} , orig acc {:.3f} , both acc {:.3f} - {}\n".format(acc, acc_orig, both_acc, subject))
                    all_cors.append(cors)  #一个class的
                    all_all_cors.append(cors)   #整个的
                    all_cors_orig.append(cors_orig)  #一个class的
                    all_all_cors_orig.append(cors_orig)   #整个的
                    all_both.append(both)
                    all_all_both.append(both)

                weighted_acc = np.mean(np.concatenate(all_cors))
                weighted_acc_orig = np.mean(np.concatenate(all_cors_orig))
                weighted_acc_both = np.mean(np.concatenate(all_both))
                f.write("{} - Average accuracy: {:.3f}, orig acc: {:.3f}, both acc: {:.3f} \n\n".format(oneclass, weighted_acc, weighted_acc_orig, weighted_acc_both))
            weighted_all_acc = np.mean(np.concatenate(all_all_cors))
            weighted_all_acc_orig = np.mean(np.concatenate(all_all_cors_orig))
            weighted_all_acc_both = np.mean(np.concatenate(all_all_both))
            f.write("Average {:.3f} , orig Average {:.3f} , both Average {:.3f} \n".format(weighted_all_acc, weighted_all_acc_orig, weighted_all_acc_both))
    else:
        with open(output_filename, 'w', encoding='utf-8') as f:
            all_all_cors=[]
            for oneclass in args.classes:
                f.write(oneclass+'\n')
                all_cors = []
                for subject in tqdm(args.new_categs[oneclass]):
                    dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
                    test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
                    cors, acc = eval(args, subject, dev_df, test_df, tokenizer, model, args.batch_size)
                    
                    f.write("Average accuracy {:.3f} - {}\n".format(acc, subject))
                    all_cors.append(cors)  #一个class的
                    all_all_cors.append(cors)   #整个的

                weighted_acc = np.mean(np.concatenate(all_cors))
                f.write("{} - Average accuracy: {:.3f}\n\n".format(oneclass, weighted_acc))
            weighted_all_acc = np.mean(np.concatenate(all_all_cors))
            f.write("Average {:.3f}\n".format(weighted_all_acc))

if __name__ == "__main__":
    import categories
    subcategs=categories.subcategories
    categs=categories.categories
    new_subcategs={}
    new_categs={}
    
    for i in subcategs:
        if subcategs[i][0] in new_subcategs:
            new_subcategs[subcategs[i][0]].append(i)
        else:
            new_subcategs[subcategs[i][0]]=[i]

    for i in categs:
        result=[]
        for j in categs[i]:
            result+=new_subcategs[j]
        new_categs[i]=result

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--token_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--temp_dir', type=str, default='')
    parser.add_argument('--zero_shot', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--classes", "-c", choices=["Humanities", "Social sciences", "STEM", "Other"],
                        default=["Humanities", "Social sciences", "STEM", "Other"], nargs="+")
    parser.add_argument('--compare', type=bool, default=False)              #我用来判断要不要做类比的，正常为false      
    args = parser.parse_args()
    args.new_subcategs=new_subcategs
    args.new_categs=new_categs
    
    main(args.ckpt_dir, args.token_dir, args.model_type)

