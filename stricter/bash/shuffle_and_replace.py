import txt_file_operate
import numpy as np
import argparse
from tqdm import tqdm

def extract(file_name):
    f=open(file_name).read()
    cors=[]
    cors_orig=[]
    A_model=[]
    A_correct=[]
    A_model_orig=[]
    A_correct_orig=[]
    lines=txt_file_operate.load(file_name)
    for idx,i in enumerate(lines):
        if 'A_model:' in i:
            aa=i
            count=idx+1
            while 'A_correct:' not in lines[count]:
                aa += lines[count]
                count += 1
            aa=aa.split('A_model:')[1].strip()
            A_model.append(aa)
        if 'A_correct:' in i:
            A_correct.append(i.split('A_correct:')[1].strip())
        if 'A_model_orig:' in i:
            A_model_orig.append(i.split('A_model_orig:')[1].strip())
        if 'A_correct_orig:' in i:
            A_correct_orig.append(i.split('A_correct_orig:')[1].strip())   
    for idx,i in enumerate(A_model):
        # cors.append(A_model[idx]==A_correct[idx])
        # cors_orig.append(A_model_orig[idx]==A_correct_orig[idx])
        cors.append(A_correct[idx] in A_model[idx])
        cors_orig.append(A_correct_orig[idx] in A_model_orig[idx])
    acc = np.mean(cors)
    acc_orig = np.mean(cors_orig)
    return cors, acc, cors_orig, acc_orig



def main():
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
    shuffle_temp_dir = args.shuffle_temp_dir
    replace_temp_dir = args.replace_temp_dir
    with open(args.output_filename, 'w', encoding='utf-8') as f:
        f.write("下面开始是原始情况，换位置和换符号都对的情况\n")
        all_cors_shuffle = []
        all_cors_replace = []
        all_cors_orig = []
        all_both=[]
        for subject in tqdm(subjects):
            cors_shuffle, acc_shuffle, cors_orig, acc_orig = extract(shuffle_temp_dir + '/%s'%subject)
            cors_replace, acc_replace, cors_orig, acc_orig = extract(replace_temp_dir + '/%s'%subject)
            both=[]
            for i in range(len(cors_shuffle)):
                if cors_shuffle[i]==True and cors_replace[i]==True and cors_orig[i]==True:
                    both.append(True)
                else:
                    both.append(False)
            both_acc = np.mean(both)
            both = np.array(both)
            f.write("Average shuffle accuracy {:.3f} , replace accuracy {:.3f} , orig accuracy {:.3f} , both accuracy {:.3f} - {}\n".format(acc_shuffle, acc_replace, acc_orig, both_acc, subject))
            all_cors_shuffle.append(cors_shuffle)  #一个class的
            all_cors_replace.append(cors_replace)  #一个class的
            all_cors_orig.append(cors_orig)  #一个class的
            all_both.append(both)
        weighted_acc_shuffle = np.mean(np.concatenate(all_cors_shuffle))
        weighted_acc_replace = np.mean(np.concatenate(all_cors_replace))
        weighted_acc_orig = np.mean(np.concatenate(all_cors_orig))
        weighted_acc_both = np.mean(np.concatenate(all_both))
        f.write("Final Average shuffle accuracy: {:.3f}, replace accuracy: {:.3f}, orig accuracy: {:.3f}, both accuracy: {:.3f} \n\n".format(weighted_acc_shuffle, weighted_acc_replace, weighted_acc_orig, weighted_acc_both))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle_temp_dir', type=str, required=True)
    parser.add_argument('--replace_temp_dir', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True) 
    args = parser.parse_args()
    
    main()

