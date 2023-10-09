import json
import os
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import fire
import pandas as pd
import torch
import tensor_parallel as tp
from tqdm import tqdm
from evaluator import Evaluator
# from fairscale.nn.model_parallel.initialize import initialize_model_parallel
# from llama import ModelArgs, Tokenizer, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from vllm import LLM, SamplingParams

choices = ["A", "B", "C", "D"]

generate_args = {
    "max_gen_length": 256,
    "temperature": 0.8,
    "top_p": 0.95,
}

class Mytest_Evaluator_vllm(Evaluator):
    def __init__(self, llm, tokenizer, choices, k, model_name="Any"):
        self.llm = llm
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
        batch_size: int=8,
        temperature: float = 0.8,
        top_p: float = 0.95
    ) -> List[str]:
        answers = []
        
        if cot:
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, max_tokens=512, stop='\n\n')
        else:
            sampling_params = SamplingParams(temperature=0, top_k=-1, max_tokens=1)
        outputs = self.llm.generate(prompts, sampling_params)
        answers = [output.outputs[0].text for output in outputs]
            # import pdb;pdb.set_trace()
        print('此任务测试完成')
        return answers

    def eval_subject(self, subject_name, batch_size, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False,**kwargs):
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            few_shot_prompt=f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"

        answers = list(test_df['answer'])
        all_prompts=[]
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + question
            # import pdb;pdb.set_trace()
            while len(self.tokenizer.tokenize(full_prompt)) + 1> 2048: # bos token
                prompt_split = full_prompt.split("\n\n")
                prompt_split.pop(1)
                full_prompt = '\n\n'.join(prompt_split)
            all_prompts.append(full_prompt)

        response_strs = self.generate_batch(
                all_prompts,
                cot=cot,
                batch_size=batch_size,
                max_gen_len=kwargs.get("max_gen_len", 200),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95)
            )
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
            if save_result_dir:
                score.append(correct)
        correct_ratio = correct_num/len(answers)
        # import pdb;pdb.set_trace()
        if save_result_dir:
            test_df['model_output']=response_strs
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio, score

    def extract_ans(self,response_str):
        pattern=[
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list

def load(
    ckpt_dir,
    token_dir,
    ntrain,
    model_type
) -> Mytest_Evaluator_vllm:
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

    llm = LLM(model=ckpt_dir, tokenizer=token_dir, tokenizer_mode="slow", dtype="float16", gpu_memory_utilization=0.95, trust_remote_code=True)
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, top_k=1, max_tokens=1)

    evaluator = Mytest_Evaluator_vllm(
        llm=llm,
        tokenizer=tokenizer,
        choices=choices,
        k=ntrain,
    )
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return evaluator


def main(
        ckpt_dir: str,
        token_dir: str,
        batch_size: int = 8,
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
    run_date = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    save_result_dir = os.path.join(
        temp_dir, f"{'_CoT' if cot else ''}{'_few-shot' if few_shot else ''}")
    os.makedirs(save_result_dir, exist_ok=True)
    with open(out_file,'w')as f:
        all_cors = []
        for subject_name in tqdm(subjects):
            val_file_path = os.path.join(data_dir+'/val', f'{subject_name}_val.csv')
            val_df = pd.read_csv(val_file_path)
            if few_shot:
                dev_file_path = os.path.join(data_dir+'/dev', f'{subject_name}_dev.csv')
                dev_df = pd.read_csv(dev_file_path)
                correct_ratio, cors = evaluator.eval_subject(
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
                correct_ratio, cors = evaluator.eval_subject(
                    subject_name,
                    batch_size,
                    val_df,
                    save_result_dir=save_result_dir,
                    cot=cot,
                    **generate_args
                )
            all_cors.append(cors)
            f.write("{} : Acc is {}\n".format(subject_name,correct_ratio))
            print("{} : Acc is {}\n".format(subject_name,correct_ratio))
        weighted_acc = np.mean(np.concatenate(all_cors))
        f.write("Final acc is {}\n".format(weighted_acc))
        print("Final acc is {}\n".format(weighted_acc))


if __name__ == "__main__":
    fire.Fire(main)
