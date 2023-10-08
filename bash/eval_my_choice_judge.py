import json
import os
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import fire
import pandas as pd
import torch
import random
import tensor_parallel as tp
from tqdm import tqdm
# from fairscale.nn.model_parallel.initialize import initialize_model_parallel
# from llama import ModelArgs, Tokenizer, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from evaluators.evaluator import Evaluator
from time import sleep
import re
from typing import List
import string
from vllm import LLM, SamplingParams

choices = ["A", "B", "C", "D"]

generate_args = {
    "max_gen_length": 256,
    "temperature": 0.8,
    "top_p": 0.95,
}

class Mytest_Evaluator(Evaluator):
    def __init__(self, model, tokenizer, choices, k, model_name="Any"):
        self.model = model
        self.tokenizer = tokenizer
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.puncs = list(string.punctuation)
        self.patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
        ]


    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example
    
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt
    
    def format_example_judge(self, line, include_answer=True, cot=False):
        prompt_question = line['question']
        ans_label = line["answer"]
        idx_toword={0:'A', 1:'B', 2:'C', 3:'D'}

        correct=choices.index(ans_label)
        incorrect=[0,1,2,3]
        incorrect.remove(correct)

        prompt_correct = prompt_question
        # import pdb;pdb.set_trace()
        prompt_correct += "\n回答: {}".format(line[ans_label])
        prompt_correct += "\n判断: "

        prompt_incorrect1 = prompt_question
        prompt_incorrect2 = prompt_question
        prompt_incorrect3 = prompt_question
        
        prompt_incorrect1 += "\n回答: {}".format(line[idx_toword[incorrect[0]]])
        prompt_incorrect1 += "\n判断: "

        prompt_incorrect2 += "\n回答: {}".format(line[idx_toword[incorrect[1]]])
        prompt_incorrect2 += "\n判断: "

        prompt_incorrect3 += "\n回答: {}".format(line[idx_toword[incorrect[2]]])
        prompt_incorrect3 += "\n判断: "

        return prompt_correct, prompt_incorrect1, prompt_incorrect2, prompt_incorrect3
        
    def format_example_shot_judge(self, line, include_answer=True, cot=False):
        # import pdb;pdb.set_trace()
        prompt = line['question']
        ans_label = line["answer"]
        idx_toword={0:'A', 1:'B', 2:'C', 3:'D'}
        correct=choices.index(ans_label)
        incorrect=[0,1,2,3]
        incorrect.remove(correct)
        random.shuffle(incorrect)
        order=random.randint(0,1)
        if order==0: # 先加错的再加对的
            prompt += "\n回答: {}".format(line[idx_toword[incorrect[0]]])
            prompt += "\n判断: 错误"
            prompt += "\n\n"
            prompt += line['question']
            prompt += "\n回答: {}".format(line[ans_label])
            prompt += "\n判断: 正确"
            prompt += "\n\n"
        elif order==1: # 先加对的再加错的
            prompt += "\n回答: {}".format(line[ans_label])
            prompt += "\n判断: 正确"
            prompt += "\n\n"
            prompt += line['question']
            prompt += "\n回答: {}".format(line[idx_toword[incorrect[0]]])
            prompt += "\n判断: 错误"
            prompt += "\n\n"
        return prompt

    def generate_few_shot_prompt_judge(self, subject, dev_df, cot=False):
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        prompt = f"判断以下关于中国{subject}考试题目的回答是正确的还是错误的。\n\n"
        for i in range(k):
            prompt += self.format_example_shot_judge(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs,max_new_tokens=512,top_p=top_p,temperature=temperature)
        return self.tokenizer.decode(outputs[0])
    
    def generate_batch(
        self,
        prompts: list,
        cot: bool = False,
        max_gen_len: int = 512,
        batch_size: int=10,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        answers = []
        for i in tqdm(range((len(prompts) // batch_size) + 1)):
            if i == len(prompts) // batch_size:
                if len(prompts)-1 >= i * batch_size:
                    prompt = prompts[i * batch_size:]
                else:
                    break
            else:
                prompt = prompts[i * batch_size: (i + 1) * batch_size]
            # import pdb;pdb.set_trace()
            if batch_size==1:
                input_tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            else:
                input_tokens = self.tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
            if cot:
                # outputs = self.model.generate(**input_tokens, max_new_tokens=300,top_p=top_p,temperature=temperature)
                outputs = self.model.generate(**input_tokens, max_new_tokens=512,top_p=top_p,temperature=temperature)
            else:
                outputs = self.model.generate(**input_tokens, max_new_tokens=1)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            torch.cuda.empty_cache()
            extract_outputs = []
            for output, myprompt in zip(outputs, prompt):
                try:
                    extract_output = output.split(myprompt)[1]
                except:
                    extract_output = output
                    print('有一处try失败，不过可以忽视')
                extract_outputs.append(extract_output)
            answers.extend(extract_outputs)
            # import pdb;pdb.set_trace()
        print('此任务测试完成')
        return answers
    
    def generate_batch_judge(
        self,
        prompts: list,
        cot: bool = False,
        max_gen_len: int = 512,
        batch_size: int=10,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        answers = []
        for i in tqdm(range((len(prompts) // batch_size) + 1)):
            if i == len(prompts) // batch_size:
                if len(prompts)-1 >= i * batch_size:
                    prompt = prompts[i * batch_size:]
                else:
                    break
            else:
                prompt = prompts[i * batch_size: (i + 1) * batch_size]
            # import pdb;pdb.set_trace()
            if batch_size==1:
                input_tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            else:
                input_tokens = self.tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
            if cot:
                # outputs = self.model.generate(**input_tokens, max_new_tokens=300,top_p=top_p,temperature=temperature)
                outputs = self.model.generate(**input_tokens, max_new_tokens=512,top_p=top_p,temperature=temperature)
            else:
                outputs = self.model.generate(**input_tokens, max_new_tokens=4)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            torch.cuda.empty_cache()
            extract_outputs = []
            for output, myprompt in zip(outputs, prompt):
                try:
                    extract_output = output.split(myprompt)[1]
                except:
                    extract_output = output
                    print('有一处try失败，不过可以忽视')
                extract_outputs.append(extract_output)
            answers.extend(extract_outputs)
            # import pdb;pdb.set_trace()
        print('此任务测试完成')
        return answers

    def eval_subject(self, subject_name, batch_size, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False,**kwargs):
        if 'qwen' in self.model_name:
            batch_size=1
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
            few_shot_prompt_new = self.generate_few_shot_prompt_judge(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"
            few_shot_prompt_new = f"你是一个中文人工智能助手，判断以下关于中国{subject}考试题目的回答是正确的还是错误的。"

        questions_orig=[]
        answers_orig = list(test_df['answer'])
        all_prompts_orig = []
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            questions_orig.append(question)
            full_prompt = few_shot_prompt + question
            # import pdb;pdb.set_trace()
            while len(self.tokenizer.tokenize(full_prompt)) + 1> 2048: # bos token
                prompt_split = full_prompt.split("\n\n")
                prompt_split.pop(1)
                full_prompt = '\n\n'.join(prompt_split)
            all_prompts_orig.append(full_prompt)

        response_strs_orig = self.generate_batch(
                all_prompts_orig,
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        
        score_orig, correct_ratio_orig = self.compute(response_strs_orig,answers_orig,few_shot,cot)
        with open(os.path.join(save_result_dir, f'{subject_name}'),'w',encoding="utf-8")as f:
            for idx,temp in enumerate(response_strs_orig):
                f.write('Q: {}\nA_model: {}\nA_correct: {}\n\n'.format(questions_orig[idx], temp, answers_orig[idx]))
                
        # Judge
        questions_correct=[]
        questions_incorrect1=[]
        questions_incorrect2=[]
        questions_incorrect3=[]
        records=[]
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question_correct, question_incorrect1, question_incorrect2, question_incorrect3 = self.format_example_judge(row, include_answer=False)
            questions_correct.append(question_correct)
            questions_incorrect1.append(question_incorrect1)
            questions_incorrect2.append(question_incorrect2)
            questions_incorrect3.append(question_incorrect3)
            full_prompt_correct = few_shot_prompt_new + question_correct
            full_prompt_incorrect1 = few_shot_prompt_new + question_incorrect1
            full_prompt_incorrect2 = few_shot_prompt_new + question_incorrect2
            full_prompt_incorrect3 = few_shot_prompt_new + question_incorrect3
            # import pdb;pdb.set_trace()
            while len(self.tokenizer.tokenize(full_prompt_correct)) + 1> 2048: # bos token
                prompt_split = full_prompt_correct.split("\n\n")
                prompt_split.pop(1)
                full_prompt_correct = '\n\n'.join(prompt_split)
            while len(self.tokenizer.tokenize(full_prompt_incorrect1)) + 1> 2048: # bos token
                prompt_split = full_prompt_incorrect1.split("\n\n")
                prompt_split.pop(1)
                full_prompt_incorrect1 = '\n\n'.join(prompt_split)
            while len(self.tokenizer.tokenize(full_prompt_incorrect2)) + 1> 2048: # bos token
                prompt_split = full_prompt_incorrect2.split("\n\n")
                prompt_split.pop(1)
                full_prompt_incorrect2 = '\n\n'.join(prompt_split)
            while len(self.tokenizer.tokenize(full_prompt_incorrect3)) + 1> 2048: # bos token
                prompt_split = full_prompt_incorrect3.split("\n\n")
                prompt_split.pop(1)
                full_prompt_incorrect3 = '\n\n'.join(prompt_split)
            records.append({'prompt_cor':full_prompt_correct, 'prompt_incor1':full_prompt_incorrect1, 'prompt_incor2':full_prompt_incorrect2, 'prompt_incor3':full_prompt_incorrect3})

        pred_answers_cor = self.generate_batch_judge(
                [record['prompt_cor'] for record in records],
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        pred_answers_incor1 = self.generate_batch_judge(
                [record['prompt_incor1'] for record in records],
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        pred_answers_incor2 = self.generate_batch_judge(
                [record['prompt_incor2'] for record in records],
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        pred_answers_incor3 = self.generate_batch_judge(
                [record['prompt_incor3'] for record in records],
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        # import pdb;pdb.set_trace()
        with open(os.path.join(save_result_dir, f'{subject_name}-correct'),'w',encoding="utf-8")as f:
            for idx,temp in enumerate(pred_answers_cor):
                f.write('Q: {}\nA_model: {}\nA_correct: 正确\n\n'.format(questions_correct[idx], temp))
        with open(os.path.join(save_result_dir, f'{subject_name}-incorrect1'),'w',encoding="utf-8")as f:
            for idx,temp in enumerate(pred_answers_incor1):
                f.write('Q: {}\nA_model: {}\nA_correct: 错误\n\n'.format(questions_incorrect1[idx], temp))
        with open(os.path.join(save_result_dir, f'{subject_name}-incorrect2'),'w',encoding="utf-8")as f:
            for idx,temp in enumerate(pred_answers_incor2):
                f.write('Q: {}\nA_model: {}\nA_correct: 错误\n\n'.format(questions_incorrect2[idx], temp))
        with open(os.path.join(save_result_dir, f'{subject_name}-incorrect3'),'w',encoding="utf-8")as f:
            for idx,temp in enumerate(pred_answers_incor3):
                f.write('Q: {}\nA_model: {}\nA_correct: 错误\n\n'.format(questions_incorrect3[idx], temp))

        cors = []
        incors1 = []
        incors2 = []
        incors3 = []

        for pred in pred_answers_cor:
            cor = '正确' in pred
            cors.append(cor)

        for pred in pred_answers_incor1:
            cor = '错误' in pred
            incors1.append(cor)
        
        for pred in pred_answers_incor2:
            cor = '错误' in pred
            incors2.append(cor)

        for pred in pred_answers_incor3:
            cor = '错误' in pred
            incors3.append(cor)
              
        acc_cor = np.mean(cors)
        acc_incor1 = np.mean(incors1)
        acc_incor2 = np.mean(incors2)
        acc_incor3 = np.mean(incors3)
        cors = np.array(cors)
        incors1 = np.array(incors1)
        incors2 = np.array(incors2)
        incors3 = np.array(incors3)

        # import pdb;pdb.set_trace()
        
        return score_orig, cors, incors1, incors2, incors3, correct_ratio_orig, acc_cor, acc_incor1, acc_incor2, acc_incor3

    def compute(self,response_strs,answers,few_shot,cot):
        score = []
        correct_num=0
        for idx,response_str in enumerate(response_strs):
            # import pdb;pdb.set_trace()
            if cot:
                ans_list=re.findall(r"答案是(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[0],answers[idx]):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0
            else:
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,answers[idx]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        if len(ans_list)>0 and (ans_list[-1]==answers[idx]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            score.append(correct)
        correct_ratio = correct_num/len(response_strs)
        return score, correct_ratio

def load(
    ckpt_dir,
    token_dir,
    ntrain,
    model_type
) -> Mytest_Evaluator:
    start_time = time.time()
    # import pdb;pdb.set_trace()
    n_gpus = torch.cuda.device_count()
    if 'llama' in token_dir or 'Llama' in token_dir or 'Llama' in ckpt_dir or 'llama' in ckpt_dir:
        tokenizer = LlamaTokenizer.from_pretrained(token_dir,use_fast=False,padding_side="left",)
    else:
        tokenizer = AutoTokenizer.from_pretrained(token_dir, padding_side='left', trust_remote_code=True)
    # import pdb;pdb.set_trace()
    try:
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    except:
        print('token id we cannot change')

    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = "auto", torch_dtype=torch.float16, trust_remote_code=True)
    # import pdb;pdb.set_trace()
    model.eval()
    model.config.use_cache = True

    evaluator = Mytest_Evaluator(
        model=model,
        tokenizer=tokenizer,
        choices=choices,
        k=ntrain,
        model_name=model_type
    )
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return evaluator


def main(
        ckpt_dir: str,
        token_dir: str,
        batch_size: int = 12,
        param_size: int = 13,
        ntrain: int = 5,
        few_shot: bool = False,
        cot: bool = False,
        max_seq_len: int = 2048,
        data_dir: str = "/share/zhouwenjie/workspace/evaluate/ceval-main/data",
        temp_dir: str = "logs",
        out_file: str = "",
        model_type: str = 'None',
):
    subjects=[
        "accountant",
        "advanced_mathematics",
        "art_studies",
        "basic_medicine",
        "business_administration",
        "chinese_language_and_literature",
        "civil_servant",
        "clinical_medicine",
        "college_chemistry",
        "college_economics",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "education_science",
        "electrical_engineer",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_chinese",
        "high_school_geography",
        "high_school_history",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "mao_zedong_thought",
        "marxism",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_geography",
        "middle_school_history",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_politics",
        "modern_chinese_history",
        "operating_system",
        "physician",
        "plant_protection",
        "probability_and_statistics",
        "professional_tour_guide",
        "sports_science",
        "tax_accountant",
        "teacher_qualification",
        "urban_and_rural_planner",
        "veterinary_medicine",
    ]
    evaluator = load(
        ckpt_dir,
        token_dir,
        ntrain=ntrain,
        model_type=model_type,
    )
    save_result_dir = os.path.join(
        temp_dir, f"{'_CoT' if cot else ''}{'_few-shot' if few_shot else ''}")
    os.makedirs(save_result_dir, exist_ok=True)
    with open(out_file,'w')as f:
        all_choice_cors = []   #选择题一个类的答案

        all_judge_cors = []  #判断题正确答案的对错情况
        all_judge_incors1 = []  #判断题错误答案的对错情况
        all_judge_incors2 = []
        all_judge_incors3 = []

        all_choice_judge_cors = []  #选择题对了后，判断题正确答案的对错情况
        all_choice_judge_incors1 = []  #选择题对了后，判断题错误答案的对错情况
        all_choice_judge_incors2 = []
        all_choice_judge_incors3 = []
        for subject_name in tqdm(subjects):
            val_file_path = os.path.join(data_dir+'/val', f'{subject_name}_val.csv')
            val_df = pd.read_csv(val_file_path)
            if few_shot:
                dev_file_path = os.path.join(data_dir+'/dev', f'{subject_name}_dev.csv')
                dev_df = pd.read_csv(dev_file_path)
                choice_cors, judge_cors, judge_incors1, judge_incors2, judge_incors3, choice_acc, judge_acc_cor, judge_acc_incor1, judge_acc_incor2, judge_acc_incor3 = evaluator.eval_subject(
                    subject_name,
                    batch_size,
                    val_df,
                    dev_df,
                    few_shot=few_shot,
                    save_result_dir=save_result_dir,
                    cot=cot,
                    **generate_args
                )
            else:
                choice_cors, judge_cors, judge_incors1, judge_incors2, judge_incors3, choice_acc, judge_acc_cor, judge_acc_incor1, judge_acc_incor2, judge_acc_incor3 = evaluator.eval_subject(
                    subject_name,
                    batch_size,
                    val_df,
                    save_result_dir=save_result_dir,
                    cot=cot,
                    **generate_args
                )
            choice_judge_cors=[]
            choice_judge_incors1=[]
            choice_judge_incors2=[]
            choice_judge_incors3=[]

            for idx,i in enumerate(choice_cors):
                choice_judge_cors.append(i==1 and judge_cors[idx]==1)
                choice_judge_incors1.append(i==1 and judge_incors1[idx]==1)
                choice_judge_incors2.append(i==1 and judge_incors2[idx]==1)
                choice_judge_incors3.append(i==1 and judge_incors3[idx]==1)

            f.write("Average accuracy of select choice {:.3f} - {}\n".format(choice_acc, subject_name))
            f.write("Average accuracy of judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format(judge_acc_cor, judge_acc_incor1, judge_acc_incor2, judge_acc_incor3, subject_name))
            f.write("Average accuracy of choice correct then judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format\
            (np.mean(choice_judge_cors), np.mean(choice_judge_incors1), np.mean(choice_judge_incors2), np.mean(choice_judge_incors3), subject_name))
            # import pdb;pdb.set_trace()
            choice_judge_cors=np.array(choice_judge_cors)
            choice_judge_incors1=np.array(choice_judge_incors1)
            choice_judge_incors2=np.array(choice_judge_incors2)
            choice_judge_incors3=np.array(choice_judge_incors3)

            all_choice_cors.append(choice_cors)  #一个class的

            all_judge_cors.append(judge_cors)  #判断题正确答案的对错情况
            all_judge_incors1.append(judge_incors1)  #判断题错误答案的对错情况
            all_judge_incors2.append(judge_incors2)  #判断题错误答案的对错情况
            all_judge_incors3.append(judge_incors3)  #判断题错误答案的对错情况

            all_choice_judge_cors.append(choice_judge_cors)  #判断题正确答案的对错情况
            all_choice_judge_incors1.append(choice_judge_incors1)  #判断题错误答案的对错情况
            all_choice_judge_incors2.append(choice_judge_incors2)  #判断题错误答案的对错情况
            all_choice_judge_incors3.append(choice_judge_incors3)  #判断题错误答案的对错情况

        weighted_choice_acc = np.mean(np.concatenate(all_choice_cors))
        f.write("Final accuracy of select choice: {:.3f}\n\n".format(weighted_choice_acc))

        weighted_judge_acc = np.mean(np.concatenate(all_judge_cors))
        weighted_judge_acc_wrong1 = np.mean(np.concatenate(all_judge_incors1))
        weighted_judge_acc_wrong2 = np.mean(np.concatenate(all_judge_incors2))
        weighted_judge_acc_wrong3 = np.mean(np.concatenate(all_judge_incors3))
        f.write("Final accuracy of judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n\n".format\
        (weighted_judge_acc, weighted_judge_acc_wrong1, weighted_judge_acc_wrong2, weighted_judge_acc_wrong3))

        weighted_choice_judge_acc = np.mean(np.concatenate(all_choice_judge_cors))
        weighted_choice_judge_acc_wrong1 = np.mean(np.concatenate(all_choice_judge_incors1))
        weighted_choice_judge_acc_wrong2 = np.mean(np.concatenate(all_choice_judge_incors2))
        weighted_choice_judge_acc_wrong3 = np.mean(np.concatenate(all_choice_judge_incors3))
        f.write("Final accuracy of choice correct with judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n\n".format\
        (weighted_choice_judge_acc, weighted_choice_judge_acc_wrong1, weighted_choice_judge_acc_wrong2, weighted_choice_judge_acc_wrong3))



if __name__ == "__main__":
    fire.Fire(main)
