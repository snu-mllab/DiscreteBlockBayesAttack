"""
Search via Bayesian Optimization
===============

"""
from collections import defaultdict
import numpy as np
import torch
import random

from algorithms.bayesopt.acquisition.algorithm import acquisition_maximization_with_indices
from algorithms.bayesopt.historyboard import HistoryBoard
from algorithms.bayesopt.surrogate_model.gp_model import MyGPModel

import time
from copy import deepcopy
import gc

def list_by_ind(l, ind):
    return [l[i] for i in ind]

class BayesOpt:
    """An attack based on Bayesian Optimization

    Args:
        dpp_type : dpp type. one of ['no','no_one','dpp_posterior]
    """

    def __init__(self, dpp_type='dpp_posterior', niter=20):
        
        self.dpp_type = dpp_type
        self.niter = niter

        self.batch_size = 4
        self.memory_count = 0
        if dpp_type == 'no_one':
            self.batch_size = 1

    def perform_search(self, x, n_vertices, BBM):
        self.orig_X = x
        self.hb = HistoryBoard(orig_X=x, n_vertices=n_vertices)
        self.BBM = BBM

        if self.BBM.num_queries >= self.BBM.query_budget or len(self.hb.target_indices)==0:
            return self.orig_X

        self.eval_and_add_datum(self.orig_X)
        
        # Initialize surrogate model wrapper.
        self.surrogate_model = MyGPModel(self.niter)

        # Bayesian Optimization Loop
        # set best evaluated seq as initial seq.
        best_ind =  self.hb.best_in_history()[3][0]
        initial_seq = self.hb.eval_X[best_ind]
        eff_len = len(self.hb.target_indices)
        
        D_0 = []
        index_order = list(range(len(self.hb.target_indices)))
        
        assert len(index_order) == eff_len, "something wrong"

        center_seq = initial_seq
        center_ind = 0

        # init INDEX_DICT, HISTORY_DICT
        LOCAL_OPTIMUM = defaultdict(list)
            
        opt_indices = deepcopy(index_order)
        
        while True:
            if not opt_indices: 
                break
            stage_init_ind = len(self.hb.eval_Y)
            stage_iter = self.BBM.query_budget - self.BBM.num_queries
            ex_ball_size = 10000
            if hasattr(self.BBM, 'random_queries'):
                n_samples = self.BBM.random_queries 
            else:
                n_samples = int(stage_iter / len(opt_indices)) if len(opt_indices)<=3 else int(stage_iter / len(opt_indices)) * 2
 
            # Exploration.
            stage_call = 0 
            if self.BBM.num_queries >= self.BBM.query_budget: break
            stage_call, best_X = self.exploration_ball_with_indices(center_seq=center_seq,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices)
            if stage_call == -1:
                return best_X
            if self.BBM.num_queries >= self.BBM.query_budget: break

            # Exploitation.
            num_candids = 10
            while stage_call < stage_iter:
                self.clean_memory_cache()

                self.surrogate_model.fit_partial(self.hb, list(range(eff_len)), stage_init_ind, prev_indices=[0]) 

                best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3]       
                for best_ind in best_inds:
                    if not (best_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                        break
                best_val = self.hb.eval_Y[best_inds[0]][0].item()
                best_seq = self.hb.eval_X[best_ind].view(1,-1)
                reference = best_val
                best_indiced = self.hb.reduce_seqs(best_seq)
                best_seqs = self.find_greedy_init_with_indices(cand_indices=best_indiced, max_radius=eff_len, num_candids=num_candids, reference=reference)
                best_candidates = acquisition_maximization_with_indices(best_seqs, opt_indices=opt_indices, batch_size=self.batch_size, stage=eff_len, hb=self.hb, surrogate_model=self.surrogate_model, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False)

                if type(best_candidates) == type(None):
                    LOCAL_OPTIMUM[tuple(opt_indices)].append(best_ind)
                    rand_indices = self.hb.nbd_sampler(best_indiced, self.batch_size, 2, 1)
                    best_candidates = [self.hb.seq_by_indices(inds) for inds in random.sample(list(rand_indices), self.batch_size)]                        

                if stage_call >= stage_iter or self.BBM.num_queries >= self.BBM.query_budget: break

                for best_candidate in best_candidates:
                    self.eval_and_add_datum(best_candidate)
                    stage_call += 1
                    if stage_call >= stage_iter or self.BBM.num_queries >= self.BBM.query_budget: break
            best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3] 
            
            for center_ind in best_inds:
                if not (center_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                    center_ind = int(center_ind)
                    break
            center_seq = self.hb.eval_X[center_ind]
            if self.BBM.num_queries >= self.BBM.query_budget: break
            break
            
        best_ind = torch.argmax(self.hb.eval_Y.view(-1)).item()
        best_X = self.hb.eval_X[best_ind].view(1,-1)
        best_Y = float(self.hb.eval_Y[best_ind][0].item())
        return best_X, best_Y

    def exploration_ball_with_indices(self, center_seq, n_samples, ball_size, stage_call, opt_indices):
        if n_samples == 0:
            return stage_call, None
        fix_indices = list(set(list(range(len(self.hb.target_indices)))) - set(opt_indices))

        prev_len = self.hb.eval_Y.shape[0]

        for i in range(n_samples):
            rand_candidate = self.hb.sample_ball_candidates_from_seq(center_seq, n_samples=1, ball_size=ball_size, fix_indices=fix_indices)[0]
            patience = 100
            assert type(rand_candidate) == torch.Tensor and rand_candidate.shape == self.orig_X.shape, "something wrong"
            while rand_candidate.cpu().detach() in self.BBM.eval_cache and patience:
                rand_candidate = self.hb.sample_ball_candidates_from_seq(center_seq, n_samples=1, ball_size=ball_size, fix_indices=fix_indices)[0]
                patience -= 1
            self.eval_and_add_datum(rand_candidate)
            if self.BBM.num_queries >= self.BBM.query_budget:
                stage_call += self.hb.eval_Y.shape[0] - prev_len
                return stage_call, None

        # If any query were not evaluated, hardly sample non orig examples.
        if self.hb.eval_Y.shape[0] == prev_len:
            center_indiced = self.hb.numbering_seq(center_seq)
            rand_indiced = copy.deepcopy(center_indiced)
            for ind in opt_indices:
                rand_indiced[ind] = int(random.sample(list(range(self.hb.reduced_n_vertices[ind]-1)), 1)[0] + 1)
            rand_candidate = self.hb.seq_by_indices(rand_indiced)
            self.eval_and_add_datum(rand_candidate)
        stage_call += self.hb.eval_Y.shape[0] - prev_len
        return stage_call, None

    def find_greedy_init_with_indices(self, cand_indices, max_radius, num_candids, reference=None):
        ### Before Greedy Ascent Step ###
        # calculate acquisition
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming(self.hb.orig_seq, max_radius)
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference)
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        center_candidates = [self.hb.seq_by_indices(cand_indices[idx]) for idx in topk_indices]
        return center_candidates

    def eval_and_add_datum(self, seq):
        if not seq in self.BBM.eval_cache:
            score = self.BBM.get_score(seq)
            self.hb.add_datum(seq, score)

    def clean_memory_cache(self):
        if self.memory_count == 1:
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_count = 0
        else:
            self.memory_count += 1

    _perform_search = perform_search
