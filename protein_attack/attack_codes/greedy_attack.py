import torch
import copy
import numpy as np

def greedy_attack(x, y, syndict, BBM):
    x_ = x.cpu().detach()
    y_ = y.cpu().detach()
    # Skipped
    if BBM.get_score(x_,y_) >= 0:
        return copy.deepcopy(x), None, None, -1
    # Success or Fail
    else:
        BBM.initialize_num_queries()
        x_att = greedy_attack_(x_, y_, syndict, BBM)
        num_queries = BBM.num_queries
        modif_rate = (torch.sum(x_att!=x_) / x_.shape[1]).item()
        succ = 1 if BBM.get_score(x_att,y_) >= 0 else 0 # 1 if Success else 0.
        return x_att.cuda(), num_queries, modif_rate, succ

def greedy_attack_(x, y, syndict, BBM):
    L = x.shape[1]
    ind_scores = []
    for ind in range(L):
        del_seq = torch.cat([x[:,:ind],x[:,ind+1:]],dim=1)
        score = BBM.get_score(del_seq, y)
        ind_scores.append(score)
    ind_order = (-np.array(ind_scores)).argsort()

    best_x = x
    best_score = BBM.get_score(x, y)
    for ind in ind_order:
        cur_word = x[0][ind].item()

        best_tmp_x, best_tmp_score = copy.deepcopy(best_x), copy.deepcopy(best_score)
        for new_word in syndict[cur_word]:
            if new_word == cur_word:
                continue
            new_seq = swap_seq_word(best_x,ind,new_word)
            new_score = BBM.get_score(new_seq, y)
            if new_score >= 0:
                return new_seq
            if new_score > best_tmp_score:
                best_tmp_x = new_seq
                best_tmp_score = new_score
        
        if best_tmp_score > best_score:
            best_x = best_tmp_x
            best_score = best_tmp_score
    return best_x


def swap_seq_word(seq, ind, word):
    assert len(seq.shape)==2 and seq.shape[0] == 1, "something wrong"
    new_seq = copy.deepcopy(seq)
    new_seq[0][ind] = word 
    return new_seq