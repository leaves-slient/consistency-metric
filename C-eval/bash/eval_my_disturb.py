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

seed_number=0
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
    
    def format_example_shuffle(self, line, include_answer=True, cot=False):
        example = line['question']
        dic={}
        idx_toword={0:'A', 1:'B', 2:'C', 3:'D'}
        global seed_number
        ss=[0,1,2,3]
        random.seed(seed_number)
        random.shuffle(ss)    #v4每次都随机
        seed_number+=1
        new_order=[idx_toword[i] for i in ss]
        ans_newlabel = idx_toword[new_order.index(line['answer'])]   #新的答案标签

        for j in range(4):
            example += f'\n{choices[j]}. {line[f"{new_order[j]}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{ans_newlabel}。\n\n"
            else:
                example += '\n答案：' + ans_newlabel + '\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example, ans_newlabel

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
        batch_size: int=12,
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

    def eval_subject(self, subject_name, batch_size, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False,**kwargs):
        if 'qwen' in self.model_name:
            batch_size=1
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"

        answers_orig = list(test_df['answer'])
        all_prompts_orig = []
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False)
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
                max_gen_len=kwargs.get("max_gen_len", 200),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        
        score_orig, correct_ratio_orig = self.compute(response_strs_orig,answers_orig,few_shot,cot)

        answers_new = []
        all_prompts_new=[]
        questions=[]
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question, new_answer = self.format_example_shuffle(row, include_answer=False)
            answers_new.append(new_answer)
            questions.append(question)
            full_prompt = few_shot_prompt + question
            # import pdb;pdb.set_trace()
            while len(self.tokenizer.tokenize(full_prompt)) + 1> 2048: # bos token
                prompt_split = full_prompt.split("\n\n")
                prompt_split.pop(1)
                full_prompt = '\n\n'.join(prompt_split)
            all_prompts_new.append(full_prompt)

        response_strs_new = self.generate_batch(
                all_prompts_new,
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 200),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
        
        score_new, correct_ratio_new = self.compute(response_strs_new,answers_new,few_shot,cot)

        with open(os.path.join(save_result_dir, f'{subject_name}'),'w',encoding="utf-8") as f:
            for idx,temp in enumerate(questions):
                f.write('Q: {}\nA_model: {}\nA_correct: {}\nA_model_orig: {}\nA_correct_orig: {}\n\n'.format(temp, response_strs_new[idx], answers_new[idx], response_strs_orig[idx], answers_orig[idx]))
        return correct_ratio_orig, score_orig, correct_ratio_new, score_new

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
        all_cors_orig = []
        all_cors_new = []
        all_both = []
        for subject_name in tqdm(subjects):
            val_file_path = os.path.join(data_dir+'/val', f'{subject_name}_val.csv')
            val_df = pd.read_csv(val_file_path)
            if few_shot:
                dev_file_path = os.path.join(data_dir+'/dev', f'{subject_name}_dev.csv')
                dev_df = pd.read_csv(dev_file_path)
                correct_ratio_orig, score_orig, correct_ratio_new, score_new = evaluator.eval_subject(
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
                correct_ratio_orig, score_orig, correct_ratio_new, score_new = evaluator.eval_subject(
                    subject_name,
                    batch_size,
                    val_df,
                    save_result_dir=save_result_dir,
                    cot=cot,
                    **generate_args
                )
            all_cors_orig.append(score_orig)
            all_cors_new.append(score_new)
            both=[]
            for i in range(len(score_orig)):
                if score_orig[i]==1 and score_new[i]==1:
                    both.append(1)
                else:
                    both.append(0)
            both = np.array(both)
            all_both.append(both)
            f.write("{} : original acc is {} shuffle acc is {} all acc is {}\n".format(subject_name,correct_ratio_orig,correct_ratio_new,np.mean(both)))
            print("{} : original acc is {} shuffle acc is {} all acc is {}\n".format(subject_name,correct_ratio_orig,correct_ratio_new,np.mean(both)))
        weighted_acc_orig = np.mean(np.concatenate(all_cors_orig))
        weighted_acc_new = np.mean(np.concatenate(all_cors_new))
        both_acc = np.mean(np.concatenate(all_both))
        f.write("Final original acc is {}, shuffle acc is {}, both acc is {}\n".format(weighted_acc_orig, weighted_acc_new, both_acc))
        print("Final original acc is {}, shuffle acc is {}, both acc is {}\n".format(weighted_acc_orig, weighted_acc_new, both_acc))


if __name__ == "__main__":
    fire.Fire(main)
