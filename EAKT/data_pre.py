import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import time, datetime
from sklearn.model_selection import train_test_split
import os

main_table = pd.read_csv('data/MainTable.csv')
early = pd.read_csv('data/early.csv')
late = pd.read_csv('data/late.csv')

diff = pd.read_csv('data/difficulty_attr.csv')
group = pd.read_csv('data/ability_attr.csv')

test = main_table.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first')
test = test.reset_index(drop = True)

early = early.merge(diff, how='left', left_on='ProblemID', right_on='ID')
early = early.merge(group, how='left', left_on='ProblemID', right_on='ID')
early_time = early.merge(test, how='left', on = ['SubjectID', 'ProblemID'])

late = late.merge(diff, how='left', left_on='ProblemID', right_on='ID')
late = late.merge(group, how='left', left_on='ProblemID', right_on='ID')
late_time = late.merge(test, how='left', on = ['SubjectID', 'ProblemID'])

all_data = pd.concat([early_time,late_time]).reset_index(drop=True)
order = ['SubjectID','AssignmentID_x','ProblemID','Label','difficult','group']
all_data = all_data[order]

skill_list = np.array(all_data['AssignmentID_x'])
skills = set(skill_list)
print('# of skills:',  len(skills))

user_id = np.array(all_data['SubjectID'])
problem_id = np.array(all_data['ProblemID'])
user = set(user_id)
problem = set(problem_id)
group_list = np.array(all_data['group'])
groups = set(group_list)

print('# of users:', len(user))
print('# of questions:',  len(problem))

user2id ={}
problem2id = {}
skill2id = {}
group2id = {}

count = 1
for i in user:
    user2id[i] = count 
    count += 1
count = 1
for i in problem:
    problem2id[i] = count 
    count += 1
count = 1
for i in skills:
    skill2id[i] = count 
    count += 1
count = 1
for i in groups:
    group2id[i] = count 
    count += 1

with open('data/user2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(user2id))
    fo.close()
with open('data/problem2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(problem2id))
    fo.close()
with open('data/skill2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(skill2id))
    fo.close()
with open('data/group2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(group2id))
    fo.close()

sdifficult2id = {}
count = []
nonesk = []  
for i in tqdm(skills):
    tttt = []
    idx = all_data[(all_data.AssignmentID_x==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    if len(idx) < 30:
        sdifficult2id[i] = 1.02
        nonesk.append(i)
        continue
    for xxx in np.array(temp1):
        tttt.append(xxx[3])
    if tttt == []:

        sdifficult2id[i] = 1.02
        nonesk.append(i)
        continue
    avg = int(np.mean(tttt)*100)+1
    count.append(avg)
    sdifficult2id[i] = avg 


difficult2id = {}
count = []
nones = []
for i in tqdm(problem):
    tttt = []
    idx = all_data[(all_data.ProblemID==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    if len(idx) < 30:
        difficult2id[i] = 1.02
        nones.append(i)
        continue
    for xxx in np.array(temp1):
        tttt.append(xxx[3])
    if tttt == []:
        difficult2id[i] = 1.02
        nones.append(i)
        continue
    avg = int(np.mean(tttt)*100)+1
    count.append(avg)

    difficult2id[i] = avg 


gpt_difficult2id = {}
count = []
none_attr_diff = []
for i in tqdm(problem):
    tttt = []
    idx = all_data[(all_data.ProblemID==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    for xxx in np.array(temp1):
        tttt.append(xxx[4])
    if tttt == []:
        gpt_difficult2id[i] = 1.02
        none_attr_diff.append(i)
        continue
    avg = int(np.mean(tttt))
    if avg <= 0:
        gpt_difficult2id[i] = 1.02
        none_attr_diff.append(i)
        continue
    count.append(avg)

    gpt_difficult2id[i] = avg 


gpt_sdifficult2id = {}
count = []
none_attr_sdiff = []
for i in tqdm(skills):
    tttt = []
    idx = all_data[(all_data.AssignmentID_x==i)].index.tolist() 
    temp1 = all_data.iloc[idx]
    for xxx in np.array(temp1):
        tttt.append(xxx[4])
    if tttt == []:
        gpt_sdifficult2id[i] = 1.02
        none_attr_sdiff.append(i)
        continue
    avg = int(np.mean(tttt))
    if avg <= 0:
        gpt_sdifficult2id[i] = 1.02
        none_attr_sdiff.append(i)
        continue
    count.append(avg)

    gpt_sdifficult2id[i] = avg



with open('data/difficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(difficult2id))
    fo.close()
with open('data/sdifficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(sdifficult2id))
    fo.close()
with open('data/gpt_difficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(gpt_difficult2id))
    fo.close()
with open('data/gpt_sdifficult2id', 'w', encoding = 'utf-8') as fo:
    fo.write(str(gpt_sdifficult2id))
    fo.close()

np.save('data/nones.npy', np.array(nones))
np.save('data/nonesk.npy', np.array(nonesk))
np.save('data/none_attr_diff.npy', np.array(none_attr_diff))
np.save('data/none_attr_sdiff.npy', np.array(none_attr_sdiff))


print('complete')
