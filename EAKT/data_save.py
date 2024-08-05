import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split

main_table = pd.read_csv('data/MainTable.csv')
early = pd.read_csv('data/early.csv')
late = pd.read_csv('data/late.csv')

diff = pd.read_csv('data/difficulty_attr.csv')
group = pd.read_csv('data/ability_attr.csv')
time = pd.read_csv('data/time_attr.csv')

test = main_table.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first')
test = test.reset_index(drop = True)

def genr_response(bool_type):
    if bool_type == True:
        return 1
    else:
        return 0 

early = early.merge(diff, how='left', left_on='ProblemID', right_on='ID')
early = early.merge(group, how='left', left_on='ProblemID', right_on='ID')
early = early.merge(time, how='left', left_on='ProblemID', right_on='ID')
early['Label'] = early['Label'].apply(genr_response)
early_time = early.merge(test, how='left', on = ['SubjectID', 'ProblemID'])

late = late.merge(diff, how='left', left_on='ProblemID', right_on='ID')
late = late.merge(group, how='left', left_on='ProblemID', right_on='ID')
late = late.merge(time, how='left', left_on='ProblemID', right_on='ID')
late['Label'] = late['Label'].apply(genr_response)
late_time = late.merge(test, how='left', on = ['SubjectID', 'ProblemID'])

all_data = pd.concat([early_time,late_time]).reset_index(drop=True)

length = 50



order = ['SubjectID','AssignmentID_x','ProblemID','Label','difficult','ServerTimestamp','group','time']

all_data = all_data[order]
early_data = early_time[order]
late_data = late_time[order]

with open('data/difficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        difficult2id = eval(line)

with open('data/sdifficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        sdifficult2id = eval(line)

with open('data/gpt_difficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        gpt_difficult2id = eval(line)

with open('data/gpt_sdifficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        gpt_sdifficult2id = eval(line)

with open('data/user2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        user2id = eval(line)
with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)
with open('data/skill2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        skill2id = eval(line)
with open('data/group2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        group2id = eval(line)


diff_temp_list = []
for key,value in gpt_difficult2id.items():
    diff_temp_list.append(value)


diff_recode = set(diff_temp_list)
time_recode = set(time['time'].unique())

diff_recode_dict = {}
time_recode_dict = {}

count = 1
for i in diff_recode:
    diff_recode_dict[i] = count 
    count += 1
count = 1
for i in time_recode:
    time_recode_dict[i] = count 
    count += 1


with open('data/diff_recode_dict', 'w', encoding = 'utf-8') as fo:
    fo.write(str(diff_recode_dict))
    fo.close()


def genr_new_qid(QuestionId):
    return problem2id[QuestionId]

def genr_new_sid(SubjectId):
    return skill2id[SubjectId]

temp_all_data = all_data
temp_all_data['new_qid'] = temp_all_data['ProblemID'].apply(genr_new_qid)
temp_all_data['new_sid'] = temp_all_data['AssignmentID_x'].apply(genr_new_sid)

def genr_edge_index():
    in_vec = []
    out_vec = []
    for i in set(np.array(temp_all_data['new_sid'])):
        index = temp_all_data[temp_all_data['new_sid'] == i].index.tolist()
        temp = temp_all_data.iloc[index]
        problem_id = np.array(temp['new_qid'])
        problems = set(problem_id)
        for i in problems:
            for j in problems:
                if i != j:
                    in_vec.append(i)
                    out_vec.append(j)
    final_list = []
    for i in range(1,51):
        in_vec.append(i)
        out_vec.append(i)
    final_list.append(in_vec)
    final_list.append(out_vec)
    return final_list


temp_list = genr_edge_index()
temp_list
np.save('data/edge_index.npy', np.array(temp_list))


def genr_adj_matrix():
    matrix = [[0]*50 for i in range(50)]
    for i in range(1,51):
        for j in range(len(temp_list[0])):
            if temp_list[0][j] == i:
                matrix[i-1][temp_list[1][j] - 1] = 1
    return matrix
temp_matrix = genr_adj_matrix()
np.save('data/adj_matrix.npy', np.array(temp_matrix))

all_user = np.array(all_data['SubjectID'])
user = sorted(list(set(all_user)))
np.random.seed(100)
np.random.shuffle(user)

train_all_id, temp_id = train_test_split(user,test_size=0.2,random_state=5)
train_id = np.array(train_all_id)
test_id, valid_id = train_test_split(temp_id,test_size=0.5,random_state=5)
test_id = np.array(test_id)
valid_id = np.array(valid_id)
nones = np.load('data/nones.npy', allow_pickle = True)
nonesk = np.load('data/nonesk.npy', allow_pickle = True)
none_attr_diff = np.load('data/none_attr_diff.npy', allow_pickle = True)
none_attr_sdiff = np.load('data/none_attr_sdiff.npy', allow_pickle = True)


q_a_train = []
for item in tqdm(train_id):

    idx = all_data[(all_data.SubjectID==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['ServerTimestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    quiz = temp

    train_q = []
    train_d = []
    train_a = []
    train_skill = []
    train_sd = []
    train_gpt_diff = []
    train_gpt_sdiff = []
    train_gpt_group = []
    train_gpt_time = []

    for one in range(0,len(quiz)):
        if True:
            train_q.append(problem2id[quiz[one][2]])
            train_d.append(difficult2id[quiz[one][2]])
            train_a.append(int(quiz[one][3]))
            train_skill.append(skill2id[quiz[one][1]])
            train_sd.append(sdifficult2id[quiz[one][1]])
            train_gpt_diff.append(diff_recode_dict[gpt_difficult2id[quiz[one][2]]])
            train_gpt_sdiff.append(gpt_sdifficult2id[quiz[one][1]])
            train_gpt_group.append(group2id[quiz[one][6]])
            train_gpt_time.append(time_recode_dict[quiz[one][7]])
            
        if len(train_q) >=length :
            q_a_train.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd, train_gpt_diff, train_gpt_sdiff, train_gpt_group, train_gpt_time])
            train_q = []
            train_d = []
            train_a = []
            train_skill = []
            train_sd = []
            train_gpt_diff = []
            train_gpt_sdiff = []
            train_gpt_group = []
            train_gpt_time = []
    if len(train_q)>=2 and len(train_q) < length:
        q_a_train.append([train_q, train_d, train_a, train_skill, len(train_q), train_sd, train_gpt_diff, train_gpt_sdiff, train_gpt_group, train_gpt_time])

np.save("data/train.npy",np.array(q_a_train, dtype=object))

q_a_test = []
for item in tqdm(test_id):

    idx = all_data[(all_data.SubjectID==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['ServerTimestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    quiz = temp

    test_q = []
    test_d = []
    test_a = []
    test_skill = []
    test_sd = []
    test_gpt_diff = []
    test_gpt_sdiff = []
    test_gpt_group = []
    test_gpt_time = []

    for one in range(0,len(quiz)):
        if True:
            test_q.append(problem2id[quiz[one][2]])
            test_d.append(difficult2id[quiz[one][2]])
            test_a.append(int(quiz[one][3]))
            test_skill.append(skill2id[quiz[one][1]])
            test_sd.append(sdifficult2id[quiz[one][1]])
            test_gpt_diff.append(diff_recode_dict[gpt_difficult2id[quiz[one][2]]])
            test_gpt_sdiff.append(gpt_sdifficult2id[quiz[one][1]])
            test_gpt_group.append(group2id[quiz[one][6]])
            test_gpt_time.append(time_recode_dict[quiz[one][7]])
            
        if len(test_q) >=length :
            q_a_test.append([test_q, test_d, test_a, test_skill, len(test_q), test_sd, test_gpt_diff, test_gpt_sdiff, test_gpt_group, test_gpt_time])
            test_q = []
            test_d = []
            test_a = []
            test_skill = []
            test_sd = []
            test_gpt_diff = []
            test_gpt_sdiff = []
            test_gpt_group = []
            test_gpt_time = []
    if len(test_q)>=2 and len(test_q) < length:
        q_a_test.append([test_q, test_d, test_a, test_skill, len(test_q), test_sd, test_gpt_diff, test_gpt_sdiff, test_gpt_group, test_gpt_time])

np.save("data/test.npy",np.array(q_a_test, dtype=object))


q_a_valid = []
for item in tqdm(valid_id):

    idx = all_data[(all_data.SubjectID==item)].index.tolist() 
    temp1 = all_data.iloc[idx]
    temp1 = temp1.sort_values(by=['ServerTimestamp']) 
    temp = np.array(temp1)
    if len(temp) < 2:
        continue
    quiz = temp

    valid_q = []
    valid_d = []
    valid_a = []
    valid_skill = []
    valid_sd = []
    valid_gpt_diff = []
    valid_gpt_sdiff = []
    valid_gpt_group = []
    valid_gpt_time = []

    for one in range(0,len(quiz)):
        if True:
            valid_q.append(problem2id[quiz[one][2]])
            valid_d.append(difficult2id[quiz[one][2]])
            valid_a.append(int(quiz[one][3]))
            valid_skill.append(skill2id[quiz[one][1]])
            valid_sd.append(sdifficult2id[quiz[one][1]])
            valid_gpt_diff.append(diff_recode_dict[gpt_difficult2id[quiz[one][2]]])
            valid_gpt_sdiff.append(gpt_sdifficult2id[quiz[one][1]])
            valid_gpt_group.append(group2id[quiz[one][6]])
            valid_gpt_time.append(time_recode_dict[quiz[one][7]])
            
        if len(valid_q) >=length :
            q_a_valid.append([valid_q, valid_d, valid_a, valid_skill, len(valid_q), valid_sd, valid_gpt_diff, valid_gpt_sdiff, valid_gpt_group, valid_gpt_time])
            valid_q = []
            valid_d = []
            valid_a = []
            valid_skill = []
            valid_sd = []
            valid_gpt_diff = []
            valid_gpt_sdiff = []
            valid_gpt_group = []
            valid_gpt_time = []
    if len(valid_q)>=2 and len(valid_q) < length:
        q_a_valid.append([valid_q, valid_d, valid_a, valid_skill, len(valid_q), valid_sd, valid_gpt_diff, valid_gpt_sdiff, valid_gpt_group, valid_gpt_time])


np.save("data/valid.npy",np.array(q_a_valid, dtype=object))
print('complete')

