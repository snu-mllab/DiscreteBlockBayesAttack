import torch
import copy
import numpy as np
import time 
def greedy_attack(x, y, syndict, BBM):
    init_time = time.time()
    x_ = x.cpu().detach()
    y_ = y.cpu().detach()

    BBM.initialize_num_queries()
    BBM.set_xy(x_, y_)

    # Skipped
    if BBM.get_score(x_,require_transform=False) >= 0:
        elapsed_time = time.time() - init_time
        return copy.deepcopy(x), None, None, -1, elapsed_time, None
    # Success or Fail
    else:
        BBM.initialize_num_queries()
        x_att, attack_logs = greedy_attack_(x_, syndict, BBM)
        
        num_queries = BBM.num_queries
        modif_rate = (torch.sum(x_att!=x_) / x_.shape[1]).item()
        succ = 1 if BBM.get_score(x_att, require_transform=False) >= 0 else 0 # 1 if Success else 0.
        elapsed_time = time.time() - init_time
        return x_att.cuda(), num_queries, modif_rate, succ, elapsed_time, attack_logs

def greedy_attack_(x, syndict, BBM):
    hb = HistoryBoardGreedy()
    L = x.shape[1]
    ind_scores = []
    for ind in range(L):
        del_seq = torch.cat([x[:,:ind],x[:,ind+1:]],dim=1)
        score = BBM.get_score(del_seq, require_transform=False)
        ind_scores.append(score)
    ind_order = (-np.array(ind_scores)).argsort()

    best_x = x
    best_score = BBM.get_score(x, require_transform=False)
    for ind in ind_order:
        cur_word = x[0][ind].item()

        best_tmp_x, best_tmp_score = copy.deepcopy(best_x), copy.deepcopy(best_score)
        for new_word in syndict[cur_word]:
            if new_word == cur_word:
                continue
            new_seq = swap_seq_word(best_x,ind,new_word)
            new_score = BBM.get_score(new_seq, require_transform=False)
            if new_score >= 0:
                return new_seq, [hb.time_list, hb.eval_X, hb.eval_Y]
            if new_score > best_tmp_score:
                best_tmp_x = new_seq
                best_tmp_score = new_score
        
        if best_tmp_score > best_score:
            best_x = best_tmp_x
            best_score = best_tmp_score
    return best_x, [hb.time_list, hb.eval_X, hb.eval_Y]

def eval_and_add_datum(seq, hb, BBM):
    score = BBM.get_score(seq, require_transform=False)
    hb.add_datum(seq, score)
    return score 

class HistoryBoardGreedy(object):
    def __init__(self):
        self.eval_X = None
        self.eval_Y = None 
        self.time_list = [time.time()]
        
    def add_datum(self, new_X, new_Y):
        if type(new_Y) == float: new_Y = torch.Tensor([new_Y])
        if self.eval_X == None:
            self.eval_X = new_X.view(1,-1)
            self.eval_Y = new_Y.view(1,1)
            self.time_list.append(time.time())
        else:
            self.eval_X = torch.cat([self.eval_X, new_X.view(1,-1)])
            self.eval_Y = torch.cat([self.eval_Y, new_Y.view(1,1)])
            self.time_list.append(time.time())
        
def swap_seq_word(seq, ind, word):
    assert len(seq.shape)==2 and seq.shape[0] == 1, "something wrong"
    new_seq = copy.deepcopy(seq)
    new_seq[0][ind] = word 
    return new_seq