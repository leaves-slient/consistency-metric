import json
import os
import time
from evaluators.my_model_test import Mytest_Evaluator_vllm
from pathlib import Path
from typing import Tuple
import numpy as np
import fire
import pandas as pd
import torch
import tensor_parallel as tp
from tqdm import tqdm
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
