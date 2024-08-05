import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
from EduKTM import KTM
import sys
import time
import logging
import random
import numpy as np
from math import sqrt
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
import torch.utils.data as Data
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

print("Loading data...")

print("Training data processing...")
train_students = np.load("data/train.npy", allow_pickle=True)

print("Training data processing...")
valid_students = np.load("data/valid.npy", allow_pickle=True)

print("Validation data processing...")
test_students = np.load("data/test.npy", allow_pickle=True)

NUM_QUESTIONS = 50
EMBED_SIZE = 128
BATCH_SIZE = 64

def pad_left(target_list):
    max_len = 50
    if len(target_list) < max_len:
        return [0] * (max_len - len(target_list)) + target_list
    else:
         return target_list


def find_key(value_to_find):
    keys_with_value = [key for key, value in problem2id.items() if value == value_to_find]
    return keys_with_value


def genr_dataloader(seq, number):
    temp_attr = []
    for i in seq:
        temp_attr.append(pad_left(i[number]))
    data = torch.IntTensor(temp_attr)
    data_loader = Data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader

train_question_loader = genr_dataloader(train_students,0)
test_question_loader = genr_dataloader(test_students,0)
valid_question_loader = genr_dataloader(valid_students,0)

train_answer_loader = genr_dataloader(train_students,2)
test_answer_loader = genr_dataloader(test_students,2)
valid_answer_loader = genr_dataloader(valid_students,2)

train_skill_loader = genr_dataloader(train_students,3)
test_skill_loader = genr_dataloader(test_students,3)
valid_skill_loader = genr_dataloader(valid_students,3)

train_gptdiff_loader = genr_dataloader(train_students,6)
test_gptdiff_loader = genr_dataloader(test_students,6)
valid_gptdiff_loader = genr_dataloader(valid_students,6)

train_group_loader = genr_dataloader(train_students,8)
test_group_loader = genr_dataloader(test_students,8)
valid_group_loader = genr_dataloader(valid_students,8)

train_time_loader = genr_dataloader(train_students,9)
test_time_loader = genr_dataloader(test_students,9)
valid_time_loader = genr_dataloader(valid_students,9)

adj_matrix = np.load('data/adj_matrix.npy', allow_pickle = True)
index_list = np.load('data/edge_index.npy', allow_pickle = True)
with open('data/gpt_difficult2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        gpt_difficult2id = eval(line)


with open('data/problem2id', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        problem2id = eval(line)


with open('data/diff_recode_dict', 'r', encoding = 'utf-8') as fi:
    for line in fi:
        diff_recode_dict = eval(line)

class EAKT_base(Module):
    def __init__(self, num_q, emb_size, dropout=0.1):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lstm_layer = LSTM(self.emb_size * 6, self.hidden_size, batch_first=True).to(device)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.embedding_question = nn.Embedding(num_q + 1, emb_size).to(device)
        self.answer_embedding = nn.Embedding(2, emb_size).to(device)
        self.diff_embedding = nn.Embedding(6, emb_size).to(device)
        self.skill_embedding = nn.Embedding(6, emb_size).to(device)
        self.group_embedding = nn.Embedding(20, emb_size).to(device)
        self.time_embedding = nn.Embedding(6, emb_size).to(device)
        self.GAT_layer = GAT(emb_size, 64, emb_size, 0.2, 0.2, 4).to(device)


    def forward(self, q, r, d, s, g, t):
        d_all = torch.IntTensor([diff_recode_dict[gpt_difficult2id[find_key(i)[0]]] for i in range(1,51)]).to(device)
        d_all = self.diff_embedding(d_all)
        q_emb_all = self.GAT_layer(d_all, torch.tensor(adj_matrix).to(device)).to(device)

        embedding_dict = {node: q_emb_all[node-1] for node in range(1,51)}
        embedding_dict[0] = q_emb_all[0]
        embedding_sequences = [[embedding_dict[int(node)] for node in batch] for batch in q]
        embedding_sequences_tensor = torch.stack([torch.stack(batch) for batch in embedding_sequences]).to(device)

        q_emb = embedding_sequences_tensor
        q_emb_raw = self.embedding_question(q).to(device)
        s_emb = self.skill_embedding(s).to(device)
        r_emb = self.answer_embedding(r).to(device)
        g_emb = self.group_embedding(g).to(device)
        t_emb = self.time_embedding(t).to(device)
        concatenated_embeddings = torch.cat((q_emb_raw, q_emb, g_emb, t_emb, s_emb, r_emb), dim=-1)
        h, _ = self.lstm_layer(concatenated_embeddings)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        

        y_trimmed = y[:, :-10, :]
        element_40 = y[:, 39, :]

        replicated_elements = element_40.unsqueeze(1).repeat(1, 10, 1)
        y_modified = torch.cat((y_trimmed, replicated_elements), dim=1)

        return y_modified


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W).to(device) 
        e = self._prepare_attentional_mechanism_input(Wh).to(device)

        zero_vec = -9e15*torch.ones_like(e).to(device)
        attention = torch.where(adj > 0, e, zero_vec).to(device)
        attention = F.softmax(attention, dim=1).to(device)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh).to(device)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]).to(device)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]).to(device)
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = nn.Linear(nhid * nheads, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x)
        return self.leakyrelu(x)


def process_raw_pred(question, true_answer, answer):
    m = torch.zeros_like(answer, dtype=torch.bool).to(device)
    for i in range(1, len(question)):
        m[i - 1, question[i] - 1] = True
    temp_answer = torch.masked_select(answer, m).to(device)
    temp_answer = torch.concat([torch.tensor([0.5]).to(device),temp_answer])
    mask = torch.zeros_like(question, dtype=torch.bool).to(device)
    mask[question != 0] = True
    final_true_answer = torch.masked_select(true_answer, mask).to(device)
    final_answer = torch.masked_select(temp_answer, mask).to(device)
    return final_answer, final_true_answer


def process_raw_pred_new_method(question, true_answer, answer):
    m = torch.zeros_like(answer, dtype=torch.bool).to(device)
    for i in range(1, len(question)):
        m[i - 1, question[i] - 1] = True
    temp_answer = torch.masked_select(answer, m).to(device)
    temp_answer = torch.concat([torch.tensor([0.5]).to(device),temp_answer])
    mask = torch.zeros_like(question, dtype=torch.bool).to(device)
    mask[question != 0] = True
    final_true_answer = torch.masked_select(true_answer, mask).to(device)
    final_answer = torch.masked_select(temp_answer, mask).to(device)
    final_question = torch.masked_select(question, mask).to(device)
    return final_answer, final_true_answer, final_question


class EAKT(KTM):
    def __init__(self, num_questions, emb_size):
        super(EAKT, self).__init__()
        self.num_questions = num_questions
        self.EAKT_model = EAKT_base(num_questions, emb_size).to(device)

    def train(self, q, a, d, s, g, test_data=None, *, epoch: int, lr=0.005) -> ...:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.EAKT_model.parameters(), lr)

        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch_q, batch_a, batch_d, batch_s in tqdm.tqdm(zip(q, a, d, s), "Epoch %s" % e):
                
                pred_y = self.EAKT_model(batch_q.to(device), batch_a.to(device), batch_d.to(device), batch_s.to(device))
                batch_size = batch_q.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch_q[student].to(device), batch_a[student].to(device), pred_y[student].to(device))
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float().to(device)])
            loss = loss_function(all_pred, all_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))

    def eval(self, q, a, d, s, g, t) -> float:
        self.EAKT_model.eval()
        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)
        for batch_q, batch_a, batch_d, batch_s, batch_g, batch_t in tqdm.tqdm(zip(q,a,d,s,g,t)):
            pred_y = self.EAKT_model(batch_q.to(device), batch_a.to(device), batch_d.to(device), batch_s.to(device), batch_g.to(device), batch_t.to(device))
            batch_size = batch_q.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch_q[student].to(device), batch_a[student].to(device), pred_y[student].to(device))
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        return roc_auc_score(y_truth.cpu().detach().numpy(), y_pred.cpu().detach().numpy())



    def eval_new_method(self, q, a, d, s, g, t) -> float:
        self.EAKT_model.eval()
        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)
        y_pred_new_method = torch.Tensor([]).to(device)
        y_truth_new_method = torch.Tensor([]).to(device)
        ques_cold_start = torch.Tensor([]).to(device)
        for batch_q, batch_a, batch_d, batch_s, batch_g, batch_t in tqdm.tqdm(zip(q,a,d,s,g,t)):
            pred_y = self.EAKT_model(batch_q.to(device), batch_a.to(device), batch_d.to(device), batch_s.to(device), batch_g.to(device), batch_t.to(device))
            batch_size = batch_q.shape[0]
            for student in range(batch_size):
                pred, truth, ques = process_raw_pred_new_method(batch_q[student].to(device), batch_a[student].to(device), pred_y[student].to(device))
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])
                if len(pred) >= 10:
                    y_pred_new_method = torch.cat([y_pred_new_method, pred[-10:]])
                    y_truth_new_method = torch.cat([y_truth_new_method, truth[-10:]])
                    ques_cold_start = torch.cat([ques_cold_start, ques[-10:]])
        
        y_pred_new_method_for_ACC = [1 if p >= 0.5 else 0 for p in y_pred_new_method]

        return roc_auc_score(y_truth_new_method.cpu().detach().numpy(), y_pred_new_method.cpu().detach().numpy()), accuracy_score(y_truth_new_method.cpu().detach().numpy(), y_pred_new_method_for_ACC)


    def save(self, filepath):
        torch.save(self.EAKT_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.EAKT_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)

    def train_and_save(self, q, a, d, s, g, t, test_data=None, *, epoch: int, lr=0.002) -> ...:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.EAKT_model.parameters(), lr)

        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch_q, batch_a, batch_d, batch_s, batch_g, batch_t in tqdm.tqdm(zip(q,a,d,s,g,t), "Epoch %s" % e):
                pred_y = self.EAKT_model(batch_q.to(device), batch_a.to(device), batch_d.to(device), batch_s.to(device), batch_g.to(device), batch_t.to(device))
                batch_size = batch_q.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch_q[student].to(device), batch_a[student].to(device), pred_y[student].to(device))
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float().to(device)])
            loss = loss_function(all_pred, all_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if e % 1 == 0:
                torch.save(self.EAKT_model.state_dict(), 'result/epoch'+str(e)+".params")
                logging.info("save parameters to %s" % 'result/epoch'+str(e)+".params")

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))


model = EAKT(NUM_QUESTIONS, EMBED_SIZE)

model.train_and_save(train_question_loader, train_answer_loader, train_gptdiff_loader, train_skill_loader, train_group_loader, train_time_loader, epoch=51)

def print_result():
    best_value = 0
    best_one = 0
    auc_list = []
    for i in range(0,50):
        model.load("result/epoch" + str(i) + ".params")
        auc, ACC = model.eval_new_method(test_question_loader, test_answer_loader, test_gptdiff_loader, test_skill_loader, test_group_loader, test_time_loader)
        if auc >= best_value:
            best_value = auc
            best_one = i
        auc_list.append(auc)
        print(i)
    return auc_list, best_one

temp_list, best_one = print_result()
for i in temp_list:
    print(i)


model.load('result/epoch'+str(best_one)+".params")
valid_auc_new_method, valid_ACC = model.eval_new_method(valid_question_loader, valid_answer_loader, valid_gptdiff_loader, valid_skill_loader, valid_group_loader, valid_time_loader)
print("[Epoch %d] auc: %.6f" % (best_one, valid_auc_new_method))
print("[Epoch %d] acc: %.6f" % (best_one, valid_ACC))