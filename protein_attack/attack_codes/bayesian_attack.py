import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from algorithms import BlockBayesOpt
import copy

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


def bayesian_attack(x, y, syndict, BBM, dpp_type='dpp_posterior', block_size=40, max_loop=5, max_patience=20, post_opt='v3'):
    '''
        x : size 1 x L tensor
        y : size 1 tensor
    '''
    x_ = x.cpu().detach()
    y_ = y.cpu().detach()
    
    BBM.initialize_num_queries()
    BBM.set_xy(x_, y_)
    n_vertices = get_nv(syndict, x_, y_)

    # Skipped
    if BBM.get_score(x_) >= 0:
        return copy.deepcopy(x), None, None, -1
    # Success or Fail
    else:
        query_budget = get_query_budget(x_, syndict, baseline='textfooler')
        BBM.set_query_budget(query_budget)
        attacker = BlockBayesAttack(block_size=block_size, max_patience=max_patience, post_opt=post_opt, dpp_type=dpp_type, max_loop=max_loop, niter=1)  
        
        x_att = attacker.perform_search(x_, n_vertices, BBM)
        num_queries = BBM.num_queries
        modif_rate = (torch.sum(x_att!=x_) / x_.shape[1]).item()
        succ = 1 if BBM.get_score(x_att,y_) >= 0 else 0 # 1 if Success else 0.
        return x_att.cuda(), num_queries, modif_rate, succ