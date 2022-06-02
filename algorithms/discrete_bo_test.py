import numpy as np
import torch

# pest control code is adapted from COMBO paper 
# Combinatorial Bayesian Optimization using the Graph Cartesian Product, Chanyong Oh et al.
# https://github.com/QUVA-Lab/COMBO/blob/master/COMBO/experiments/test_functions/multiple_categorical.py

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25

def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac

def _pest_control_score(x):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold

class PestControl(object):
    """
    Ising Sparsification Problem with the simplest graph
    """
    def __init__(self, random_seed=None):
        self.n_vertices = np.array([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES)
        self.random_seed_info = str(random_seed).zfill(4)
    
    def evaluate(self, x):
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy())
        return float(evaluation)

class BlackBoxModel():
    def __init__(self):
        self.eval_cache = dict()
        self.initialize_num_queries()
        self.pc = PestControl()

    def initialize_num_queries(self):
        self.clean_cache()
        self.num_queries = 0
    
    def clean_cache(self):
        del self.eval_cache
        self.eval_cache = dict()
    
    def set_query_budget(self, query_budget):
        self.query_budget = query_budget
    
    def set_random_queries(self, random_queries):
        self.random_queries = random_queries
    
    def get_score(self,x):
        '''
            torch tensor [0,2,1,3,2,...]
            -> input to pest control
            -> return score
        '''
        if x in self.eval_cache:
            score = -self.pc.evaluate(x)
            self.num_queries += 1
            self.eval_cache[x] = score
        else:
            score = self.eval_cache[x]
        return score

import time
from discrete_bayesian_opt import BayesOpt


def get_nv():
    nv = []
    for i in range(25):
        nv.append(5)
    return nv

def bayesian_opt(x, nv, BBM, query_budget=300, random_queries=50, dpp_type='dpp_posterior', niter=1):
    '''
        x : size 1 x L tensor
    '''
    x_ = x
    # Success or Fail
    BBM.initialize_num_queries()
    BBM.set_query_budget(query_budget)
    BBM.set_random_queries(random_queries)
    attacker = BayesOpt(dpp_type, niter)
    x_att, y_att = attacker.perform_search(x_, nv, BBM)
    num_queries = BBM.num_queries
    return x_att, y_att, num_queries

if __name__ == '__main__':
    BBM = BlackBoxModel()
    nv = get_nv()
    x = torch.LongTensor([[0 for i in range(25)]])

    sl = []
    for i in range(5):
        tt0 = time.time()
        x_att, y_att, num_queries = bayesian_opt(x,nv,BBM,random_queries=20,niter=1)
        tt1 = time.time()
        print(f"{i}-th trial")
        print(f"{tt1-tt0} sec")
        print(x_att)
        print("score = ",-y_att) # COMBO's result was 12.0012 += 0.0033
        print(num_queries)
        sl.append(-y_att)
    print("result: ", sum(sl)/len(sl))