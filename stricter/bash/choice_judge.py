import txt_file_operate
import numpy as np
import argparse
from tqdm import tqdm

def extract(file_name):
    cors=[]
    A_model=[]
    A_correct=[]
    lines=txt_file_operate.load(file_name)
    for i in lines:
        if 'A_model:' in i:
            A_model.append(i.split('A_model:')[1].strip())
        if 'A_correct:' in i:
            A_correct.append(i.split('A_correct:')[1].strip())  
    for idx,i in enumerate(A_model):
        cors.append(A_model[idx]==A_correct[idx])
    # import pdb;pdb.set_trace()
    acc = np.mean(cors)
    return cors, acc

def extract_judge(file_name):
    cors=[]
    A_model=[]
    A_correct=[]
    lines=txt_file_operate.load(file_name)
    for i in lines:
        if 'A_model:' in i:
            A_model.append(i.split('A_model:')[1].strip())
        if 'A_correct:' in i:
            A_correct.append(i.split('A_correct:')[1].strip())  
    for idx,i in enumerate(A_model):
        cors.append(A_correct[idx] in A_model[idx])
    # import pdb;pdb.set_trace()
    acc = np.mean(cors)
    return cors, acc

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
    temp_dir = args.temp_dir
    with open(args.output_filename, 'w', encoding='utf-8') as f:
        f.write("下面开始是选择题和判断题都对的情况，之前没有做改动的版本的判断题在选择题对了的情况的基础上对不对的情况\n")
        all_choice_cors = []   #选择题一个类的答案

        all_judge_cors = []  #判断题正确答案的对错情况
        all_judge_incors1 = []  #判断题错误答案的对错情况
        all_judge_incors2 = []
        all_judge_incors3 = []

        all_choice_judge_cors = []  #选择题对了后，判断题正确答案的对错情况
        all_choice_judge_incors1 = []  #选择题对了后，判断题错误答案的对错情况
        all_choice_judge_incors2 = []
        all_choice_judge_incors3 = []

        All_All_correct = []
        for subject in tqdm(subjects):
            choice_cors, choice_acc = extract(temp_dir + '/%s'%subject)
            judge_cors, judge_acc_cor = extract_judge(temp_dir + '/%s'%subject + '-correct')
            judge_incors1, judge_acc_incor1 = extract_judge(temp_dir + '/%s'%subject + '-incorrect1')
            judge_incors2, judge_acc_incor2 = extract_judge(temp_dir + '/%s'%subject + '-incorrect2')
            judge_incors3, judge_acc_incor3 = extract_judge(temp_dir + '/%s'%subject + '-incorrect3')

            choice_judge_cors=[]
            choice_judge_incors1=[]
            choice_judge_incors2=[]
            choice_judge_incors3=[]

            All_correct=[]

            for i in range(len(choice_cors)):
                if choice_cors[i]==True and judge_cors[i]==True and judge_incors1[i]==True and judge_incors2[i]==True and judge_incors3[i]==True:
                    All_correct.append(True)
                else:
                    All_correct.append(False)
                if choice_cors[i]==True and judge_cors[i]==True:
                    choice_judge_cors.append(True)
                else:
                    choice_judge_cors.append(False)
                if choice_cors[i]==True and judge_incors1[i]==True:
                    choice_judge_incors1.append(True)
                else:
                    choice_judge_incors1.append(False)
                if choice_cors[i]==True and judge_incors2[i]==True:
                    choice_judge_incors2.append(True)
                else:
                    choice_judge_incors2.append(False)
                if choice_cors[i]==True and judge_incors3[i]==True:
                    choice_judge_incors3.append(True)
                else:
                    choice_judge_incors3.append(False)
            # import pdb;pdb.set_trace()

            f.write("Average accuracy of select choice {:.3f} - {}\n".format(choice_acc, subject))
            f.write("Average accuracy of judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format(judge_acc_cor, judge_acc_incor1, judge_acc_incor2, judge_acc_incor3, subject))
            f.write("Average accuracy of choice correct with judge correct {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f} - {}\n".format\
            (np.mean(choice_judge_cors), np.mean(choice_judge_incors1), np.mean(choice_judge_incors2), np.mean(choice_judge_incors3), subject))
            
            choice_judge_cors=np.array(choice_judge_cors)
            choice_judge_incors1=np.array(choice_judge_incors1)
            choice_judge_incors2=np.array(choice_judge_incors2)
            choice_judge_incors3=np.array(choice_judge_incors3)
            All_correct=np.array(All_correct)

            all_choice_cors.append(choice_cors)  #一个class的

            all_judge_cors.append(judge_cors)  #判断题正确答案的对错情况
            all_judge_incors1.append(judge_incors1)  #判断题错误答案的对错情况
            all_judge_incors2.append(judge_incors2)  #判断题错误答案的对错情况
            all_judge_incors3.append(judge_incors3)  #判断题错误答案的对错情况

            all_choice_judge_cors.append(choice_judge_cors)  #判断题正确答案的对错情况
            all_choice_judge_incors1.append(choice_judge_incors1)  #判断题错误答案的对错情况
            all_choice_judge_incors2.append(choice_judge_incors2)  #判断题错误答案的对错情况
            all_choice_judge_incors3.append(choice_judge_incors3)  #判断题错误答案的对错情况

            All_All_correct.append(All_correct)
            
        weighted_choice_acc = np.mean(np.concatenate(all_choice_cors))
        f.write("Final Average accuracy of select choice: {:.3f}\n\n".format(weighted_choice_acc))

        weighted_judge_acc = np.mean(np.concatenate(all_judge_cors))
        weighted_judge_acc_wrong1 = np.mean(np.concatenate(all_judge_incors1))
        weighted_judge_acc_wrong2 = np.mean(np.concatenate(all_judge_incors2))
        weighted_judge_acc_wrong3 = np.mean(np.concatenate(all_judge_incors3))
        f.write("Final Average accuracy of judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n\n".format\
        (weighted_judge_acc, weighted_judge_acc_wrong1, weighted_judge_acc_wrong2, weighted_judge_acc_wrong3))

        weighted_choice_judge_acc = np.mean(np.concatenate(all_choice_judge_cors))
        weighted_choice_judge_acc_wrong1 = np.mean(np.concatenate(all_choice_judge_incors1))
        weighted_choice_judge_acc_wrong2 = np.mean(np.concatenate(all_choice_judge_incors2))
        weighted_choice_judge_acc_wrong3 = np.mean(np.concatenate(all_choice_judge_incors3))
        f.write("Final Average accuracy of choice correct with judge correct: {:.3f} incorrect1 {:.3f} incorrect2 {:.3f} incorrect3 {:.3f}\n选择且四个选项都判对的概率是 {:.3f}\n\n".format\
        (weighted_choice_judge_acc, weighted_choice_judge_acc_wrong1, weighted_choice_judge_acc_wrong2, weighted_choice_judge_acc_wrong3, np.mean(np.concatenate(All_All_correct))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_dir', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True)
    parser.add_argument("--classes", "-c", choices=["Humanities", "Social sciences", "STEM", "Other"],
                        default=["Humanities", "Social sciences", "STEM", "Other"], nargs="+")    
    args = parser.parse_args()
    
    main()

