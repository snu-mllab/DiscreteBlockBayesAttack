import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from algorithms import BlockBayesAttack
import copy
import torch
import time
def get_query_budget(x, syndict, baseline='textfooler'):
    if baseline == 'textfooler':
        query_budget_count = []
        for ind in range(x.shape[1]):
            num = x[0][ind].cpu().item() 
            candid = syndict[num]
            query_budget_count.append(len(candid))
        print(sum(query_budget_count))
        query_budget = sum(query_budget_count)
    return query_budget

def get_nv(BBM, x_):
    nv = []
    for ind in range(BBM.len_seq):
        nv.append(len(BBM.syndict[x_[0][ind].cpu().item()]))
    return nv


def bayesian_attack(x, y, syndict, BBM, dpp_type='dpp_posterior', block_size=40, max_loop=5, max_patience=20, post_opt='v3', fit_iter=3):
    '''
        x : size 1 x L tensor
        y : size 1 tensor
    '''
    init_time = time.time()
    x_ = x.cpu().detach()
    y_ = y.cpu().detach()
        
    BBM.initialize_num_queries()
    BBM.set_xy(x_, y_)
    n_vertices = get_nv(BBM, x_)

    # Skipped
    if BBM.get_score(x_, require_transform=False) >= 0:
        elapsed_time = time.time() - init_time
        return copy.deepcopy(x), None, None, -1, elapsed_time, None
    # Success or Fail
    else:
        query_budget = get_query_budget(x_, syndict, baseline='textfooler')
        BBM.set_query_budget(query_budget)
        attacker = BlockBayesAttack(block_size=block_size, max_patience=max_patience, post_opt=post_opt, dpp_type=dpp_type, max_loop=max_loop, fit_iter=fit_iter)  
        
        attacker_input = torch.zeros(1, x_.numel())
        x_att, attack_logs = attacker.perform_search(attacker_input, n_vertices, BBM) 
        x_att_transformed = BBM.seq2input(x_att)
    
        num_queries = BBM.num_queries
        modif_rate = (torch.sum(x_att_transformed!=x_) / x_.shape[1]).item()
        
        score = BBM.get_score(x_att_transformed,require_transform=False) 
        
        succ = 1 if score >= 0 else 0 # 1 if Success else 0.
        elapsed_time = time.time() - init_time
        return x_att_transformed.cuda(), num_queries, modif_rate, succ, elapsed_time, attack_logs