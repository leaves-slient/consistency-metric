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
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8",line_buffering=True)


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

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def format_example_judge(df, idx, include_answer=True):
    prompt_question = df.iloc[idx, 0]
    k = df.shape[1] - 2
    ans_label = df.iloc[idx, k + 1]
    correct=choices.index(ans_label)
    incorrect=[0,1,2,3]
    incorrect.remove(correct)

    prompt_correct = prompt_question
    # import pdb;pdb.set_trace()
    prompt_correct += "\nResponse: {}".format(df.iloc[idx, correct+1])
    prompt_correct += "\nJudge: "

    prompt_incorrect1 = prompt_question
    prompt_incorrect2 = prompt_question
    prompt_incorrect3 = prompt_question
    
    prompt_incorrect1 += "\nResponse: {}".format(df.iloc[idx, incorrect[0]+1])
    prompt_incorrect1 += "\nJudge: "

    prompt_incorrect2 += "\nResponse: {}".format(df.iloc[idx, incorrect[1]+1])
    prompt_incorrect2 += "\nJudge: "

    prompt_incorrect3 += "\nResponse: {}".format(df.iloc[idx, incorrect[2]+1])
    prompt_incorrect3 += "\nJudge: "

    return prompt_correct, prompt_incorrect1, prompt_incorrect2, prompt_incorrect3
    
def format_example_shot_judge(df, idx, include_answer=True):
    # import pdb;pdb.set_trace()
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    ans_label = df.iloc[idx, k + 1]
    correct=choices.index(ans_label)
    incorrect=[0,1,2,3]
    incorrect.remove(correct)
    random.shuffle(incorrect)
    order=random.randint(0,1)
    if order==0: # 先加错的再加对的
        prompt += "\nResponse: {}".format(df.iloc[idx, incorrect[0]+1])
        prompt += "\nJudge: incorrect"
        prompt += "\n\n"
        prompt += df.iloc[idx, 0]
        prompt += "\nResponse: {}".format(df.iloc[idx, correct+1])
        prompt += "\nJudge: correct"
        prompt += "\n\n"
    elif order==1: # 先加对的再加错的
        prompt += "\nResponse: {}".format(df.iloc[idx, correct+1])
        prompt += "\nJudge: correct"
        prompt += "\n\n"
        prompt += df.iloc[idx, 0]
        prompt += "\nResponse: {}".format(df.iloc[idx, incorrect[0]+1])
        prompt += "\nJudge: incorrect"
        prompt += "\n\n"
    return prompt

def gen_prompt_judge(train_df, subject, k=-1):
    prompt = "Determine whether the following descriptions of {} are correct or incorrect.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example_shot_judge(train_df, i)
    # import pdb;pdb.set_trace()
    return prompt

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir, token_dir, model_type):
    n_gpus = torch.cuda.device_count()
    if 'llama' in token_dir or 'Llama' in token_dir or 'Llama' in ckpt_dir or 'llama' in ckpt_dir:
        tokenizer = LlamaTokenizer.from_pretrained(token_dir,use_fast=False,padding_side="left",)
    else:
        tokenizer = AutoTokenizer.from_pretrained(token_dir, padding_side='left', trust_remote_code=True)

    try:
        if tokenizer.unk_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        elif 'gpt-j' in token_dir and tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        elif 'mpt' in token_dir and tokenizer.eos_token_id:
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
        # print('111')
        # print(answers)
        # import pdb;pdb.set_trace()
    answers = [answer[-1] for answer in answers]
    # import pdb;pdb.set_trace()
    return answers

def batch_infer_judge(model, tokenizer, prompts, my_batch_size=8):
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
        aa = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        bb = [i.split('Judge:')[i.count('Judge:')].strip() for i in aa]
        answers.extend(bb)
        # import pdb;pdb.set_trace()
        # print('111')
        # print(answers)
        # import pdb;pdb.set_trace()
    # answers = [answer[-1] for answer in answers]
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
        prompt_end = format_example(test_df, i, include_answer=False)
        prompt_ends.append(prompt_end)
        train_prompt = gen_prompt(dev_df, subject, k)
        if args.zero_shot:
            train_prompt = "The following ia an multiple choice question (with answers) about {}.\n\n".format(format_subject(subject))
        prompt = train_prompt + prompt_end
        # import pdb;pdb.set_trace()
        while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
            prompt_split = prompt.split("\n\n")
            prompt_split.pop(1)
            prompt = '\n\n'.join(prompt_split)

        label = test_df.iloc[i, test_df.shape[1]-1]
        records.append({'prompt':prompt, 'answer':label})
        # import pdb;pdb.set_trace()
    pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records], my_batch_size)
    gold_answers = [record['answer'] for record in records]
    if args.temp_dir!='':
        with open(args.temp_dir + '/%s'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers):
                try:
                    f.write('Q: {}\nA_model: {}\nA_correct: {}\n\n'.format(prompt_ends[idx], temp, gold_answers[idx]))
                except:
                    f.write('error! \n\n')
    # print('!!!!')
    for pred, gold in zip(pred_answers, gold_answers):
        cor = pred == gold
        cors.append(cor)
    acc = np.mean(cors)
    cors = np.array(cors)
    return cors, acc

def judge_eval(args, subject, dev_df, test_df, tokenizer, model, my_batch_size=8):
    cors = []
    incors1 = []
    incors2 = []
    incors3 = []
    records=[]
    print(subject)
    prompt_ends_correct=[]
    prompt_ends_incorrect1=[]
    prompt_ends_incorrect2=[]
    prompt_ends_incorrect3=[]
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_correct, prompt_incorrect1, prompt_incorrect2, prompt_incorrect3 = format_example_judge(test_df, i, include_answer=False)
        # import pdb;pdb.set_trace()
        prompt_ends_correct.append(prompt_correct)
        prompt_ends_incorrect1.append(prompt_incorrect1)
        prompt_ends_incorrect2.append(prompt_incorrect2)
        prompt_ends_incorrect3.append(prompt_incorrect3)
        train_prompt = gen_prompt_judge(dev_df, subject, k)
        # import pdb;pdb.set_trace()
        if args.zero_shot:
            train_prompt = "Determine whether the following description of {} is correct or incorrect.\n\n".format(format_subject(subject))
        prompt_cor = train_prompt + prompt_correct
        prompt_incor1 = train_prompt + prompt_incorrect1
        prompt_incor2 = train_prompt + prompt_incorrect2
        prompt_incor3 = train_prompt + prompt_incorrect3
        # import pdb;pdb.set_trace()
        while len(tokenizer.tokenize(prompt_cor)) + 1> 2048: # bos token
            prompt_split = prompt_cor.split("\n\n")
            prompt_split.pop(1)
            prompt_cor = '\n\n'.join(prompt_split)

        while len(tokenizer.tokenize(prompt_incor1)) + 1> 2048: # bos token
            prompt_split = prompt_incor1.split("\n\n")
            prompt_split.pop(1)
            prompt_incor1 = '\n\n'.join(prompt_split)

        while len(tokenizer.tokenize(prompt_incor2)) + 1> 2048: # bos token
            prompt_split = prompt_incor2.split("\n\n")
            prompt_split.pop(1)
            prompt_incor2 = '\n\n'.join(prompt_split)
        
        while len(tokenizer.tokenize(prompt_incor3)) + 1> 2048: # bos token
            prompt_split = prompt_incor3.split("\n\n")
            prompt_split.pop(1)
            prompt_incor3 = '\n\n'.join(prompt_split)

        records.append({'prompt_cor':prompt_cor, 'prompt_incor1':prompt_incor1, 'prompt_incor2':prompt_incor2, 'prompt_incor3':prompt_incor3})
        # import pdb;pdb.set_trace()
    pred_answers_cor = batch_infer_judge(model, tokenizer, [record['prompt_cor'] for record in records], my_batch_size)
    # import pdb;pdb.set_trace()
    pred_answers_incor1 = batch_infer_judge(model, tokenizer, [record['prompt_incor1'] for record in records], my_batch_size)
    pred_answers_incor2 = batch_infer_judge(model, tokenizer, [record['prompt_incor2'] for record in records], my_batch_size)
    pred_answers_incor3 = batch_infer_judge(model, tokenizer, [record['prompt_incor3'] for record in records], my_batch_size)
    if args.temp_dir!='':
        with open(args.temp_dir + '/%s-correct'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers_cor):
                try:
                    f.write('Q: {}\nA_model: {}\nA_correct: correct\n\n'.format(prompt_ends_correct[idx], temp))
                except:
                    f.write('error!\n\n')
        with open(args.temp_dir + '/%s-incorrect1'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers_incor1):
                try:
                    f.write('Q: {}\nA_model: {}\nA_correct: incorrect\n\n'.format(prompt_ends_incorrect1[idx], temp))
                except:
                    f.write('error!\n\n')
        with open(args.temp_dir + '/%s-incorrect2'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers_incor2):
                try:
                    f.write('Q: {}\nA_model: {}\nA_correct: incorrect\n\n'.format(prompt_ends_incorrect2[idx], temp))
                except:
                    f.write('error!\n\n')
        with open(args.temp_dir + '/%s-incorrect3'%subject, 'w', encoding='utf-8')as f:
            for idx,temp in enumerate(pred_answers_incor3):
                try:
                    f.write('Q: {}\nA_model: {}\nA_correct: incorrect\n\n'.format(prompt_ends_incorrect3[idx], temp))
                except:
                    f.write('error!\n\n')
    # print('!!!!')
    for pred in pred_answers_cor:
        cor = pred == 'correct'
        cors.append(cor)

    for pred in pred_answers_incor1:
        cor = pred == 'incorrect'
        incors1.append(cor)
    
    for pred in pred_answers_incor2:
        cor = pred == 'incorrect'
        incors2.append(cor)

    for pred in pred_answers_incor3:
        cor = pred == 'incorrect'
        incors3.append(cor)

    acc_cor = np.mean(cors)
    acc_incor1 = np.mean(incors1)
    acc_incor2 = np.mean(incors2)
    acc_incor3 = np.mean(incors3)
    cors = np.array(cors)
    incors1 = np.array(incors1)
    incors2 = np.array(incors2)
    incors3 = np.array(incors3)
    return cors, incors1, incors2, incors3, acc_cor, acc_incor1, acc_incor2, acc_incor3

def main(ckpt_dir: str, token_dir: str, model_type: str):
    output_filename = args.out_file
    if args.temp_dir!='':
        os.makedirs(args.temp_dir, exist_ok=True)
    model, tokenizer = load(ckpt_dir, token_dir, model_type)
    start_time = time.time()
    with open(output_filename, 'w') as f:
        all_all_choice_cors=[]   #选择题的所有答案

        all_all_judge_cors=[]   #判断题所有正确答案的对错情况
        all_all_judge_incors1=[]  #判断题所有错误答案  错误选项1
        all_all_judge_incors2=[]
        all_all_judge_incors3=[]

        all_all_choice_judge_cors=[]   #选择题对了后，判断题所有正确答案的对错情况
        all_all_choice_judge_incors1=[]  #选择题对了后，判断题所有错误答案的对错情况  错误选项1
        all_all_choice_judge_incors2=[]
        all_all_choice_judge_incors3=[]

        for oneclass in args.classes:
            f.write(oneclass+'\n')
            all_choice_cors = []   #选择题一个类的答案

            all_judge_cors = []  #判断题正确答案的对错情况
            all_judge_incors1 = []  #判断题错误答案的对错情况
            all_judge_incors2 = []
            all_judge_incors3 = []

            all_choice_judge_cors = []  #选择题对了后，判断题正确答案的对错情况
            all_choice_judge_incors1 = []  #选择题对了后，判断题错误答案的对错情况
            all_choice_judge_incors2 = []
            all_choice_judge_incors3 = []
            for subject in tqdm(args.new_categs[oneclass]):
                dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
                test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
                choice_cors, choice_acc = eval(args, subject, dev_df, test_df, tokenizer, model, args.batch_size)
                judge_cors, judge_incors1, judge_incors2, judge_incors3, judge_acc_cor, judge_acc_incor1, judge_acc_incor2, \
                judge_acc_incor3 = judge_eval(args, subject, dev_df, test_df, tokenizer, model, args.batch_size)

                choice_judge_cors=[]
                choice_judge_incors1=[]
                choice_judge_incors2=[]
                choice_judge_incors3=[]

                for idx,i in enumerate(choice_cors):
                    choice_judge_cors.append(i==True and judge_cors[idx]==True)
                    choice_judge_incors1.append(i==True and judge_incors1[idx]==True)
                    choice_judge_incors2.append(i==True and judge_incors2[idx]==True)
                    choice_judge_incors3.append(i==True and judge_incors3[idx]==True)
                
                f.write("Average accuracy of select choice {:.3f} - {}\n".format(choice_acc, subject))
                f.write("Average accuracy of judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format(judge_acc_cor, judge_acc_incor1, judge_acc_incor2, judge_acc_incor3, subject))
                f.write("Average accuracy of choice correct then judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format\
                (np.mean(choice_judge_cors), np.mean(choice_judge_incors1), np.mean(choice_judge_incors2), np.mean(choice_judge_incors3), subject))
                
                choice_judge_cors=np.array(choice_judge_cors)
                choice_judge_incors1=np.array(choice_judge_incors1)
                choice_judge_incors2=np.array(choice_judge_incors2)
                choice_judge_incors3=np.array(choice_judge_incors3)

                all_choice_cors.append(choice_cors)  #一个class的
                all_all_choice_cors.append(choice_cors)   #整个的

                all_judge_cors.append(judge_cors)  #判断题正确答案的对错情况
                all_all_judge_cors.append(judge_cors)  
                all_judge_incors1.append(judge_incors1)  #判断题错误答案的对错情况
                all_all_judge_incors1.append(judge_incors1)  
                all_judge_incors2.append(judge_incors2)  #判断题错误答案的对错情况
                all_all_judge_incors2.append(judge_incors2) 
                all_judge_incors3.append(judge_incors3)  #判断题错误答案的对错情况
                all_all_judge_incors3.append(judge_incors3) 

                all_choice_judge_cors.append(choice_judge_cors)  #判断题正确答案的对错情况
                all_all_choice_judge_cors.append(choice_judge_cors)  
                all_choice_judge_incors1.append(choice_judge_incors1)  #判断题错误答案的对错情况
                all_all_choice_judge_incors1.append(choice_judge_incors1)  
                all_choice_judge_incors2.append(choice_judge_incors2)  #判断题错误答案的对错情况
                all_all_choice_judge_incors2.append(choice_judge_incors2) 
                all_choice_judge_incors3.append(choice_judge_incors3)  #判断题错误答案的对错情况
                all_all_choice_judge_incors3.append(choice_judge_incors3) 
                

            weighted_choice_acc = np.mean(np.concatenate(all_choice_cors))
            f.write("{} - Average accuracy of select choice: {:.3f}\n\n".format(oneclass, weighted_choice_acc))

            weighted_judge_acc = np.mean(np.concatenate(all_judge_cors))
            weighted_judge_acc_wrong1 = np.mean(np.concatenate(all_judge_incors1))
            weighted_judge_acc_wrong2 = np.mean(np.concatenate(all_judge_incors2))
            weighted_judge_acc_wrong3 = np.mean(np.concatenate(all_judge_incors3))
            f.write("{} - Average accuracy of judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n\n".format\
            (oneclass, weighted_judge_acc, weighted_judge_acc_wrong1, weighted_judge_acc_wrong2, weighted_judge_acc_wrong3))

            weighted_choice_judge_acc = np.mean(np.concatenate(all_choice_judge_cors))
            weighted_choice_judge_acc_wrong1 = np.mean(np.concatenate(all_choice_judge_incors1))
            weighted_choice_judge_acc_wrong2 = np.mean(np.concatenate(all_choice_judge_incors2))
            weighted_choice_judge_acc_wrong3 = np.mean(np.concatenate(all_choice_judge_incors3))
            f.write("{} - Average accuracy of choice correct with judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n\n".format\
            (oneclass, weighted_choice_judge_acc, weighted_choice_judge_acc_wrong1, weighted_choice_judge_acc_wrong2, weighted_choice_judge_acc_wrong3))

        weighted_all_choice_acc = np.mean(np.concatenate(all_all_choice_cors))
        f.write("Average of select choice {:.3f}\n".format(weighted_all_choice_acc))

        weighted_all_judge_acc = np.mean(np.concatenate(all_all_judge_cors))
        weighted_all_judge_acc_wrong1 = np.mean(np.concatenate(all_all_judge_incors1))
        weighted_all_judge_acc_wrong2 = np.mean(np.concatenate(all_all_judge_incors2))
        weighted_all_judge_acc_wrong3 = np.mean(np.concatenate(all_all_judge_incors3))
        f.write("Average of judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n".\
        format(weighted_all_judge_acc, weighted_all_judge_acc_wrong1, weighted_all_judge_acc_wrong2, weighted_all_judge_acc_wrong3))

        weighted_all_choice_judge_acc = np.mean(np.concatenate(all_all_choice_judge_cors))
        weighted_all_choice_judge_acc_wrong1 = np.mean(np.concatenate(all_all_choice_judge_incors1))
        weighted_all_choice_judge_acc_wrong2 = np.mean(np.concatenate(all_all_choice_judge_incors2))
        weighted_all_choice_judge_acc_wrong3 = np.mean(np.concatenate(all_all_choice_judge_incors3))
        f.write("Average of choice correct then judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n".\
        format(weighted_all_choice_judge_acc, weighted_all_choice_judge_acc_wrong1, weighted_all_choice_judge_acc_wrong2, weighted_all_choice_judge_acc_wrong3))

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
    args = parser.parse_args()
    args.new_subcategs=new_subcategs
    args.new_categs=new_categs
    
    main(args.ckpt_dir, args.token_dir, args.model_type)

