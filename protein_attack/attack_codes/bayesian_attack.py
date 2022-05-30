
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


def bayesian_attack(x, y, syndict, BBM, kernel_name='categorical', block_policy='straight', dpp_type='dpp_posterior', block_size=40, max_loop=5, max_patience=20):
    '''
        x : size 1 x L tensor
        y : size 1 tensor
    '''
    x_ = x.cpu().detach()
    y_ = y.cpu().detach()
    # Skipped
    if BBM.get_score(x_,y_) >= 0:
        return copy.deepcopy(x), None, None, -1
    # Success or Fail
    else:
        BBM.initialize_num_queries()
        query_budget = get_query_budget(x_, syndict, baseline='textfooler')
        BBM.set_query_budget(query_budget)
        attacker = BayesAttack(kernel_name, block_policy, dpp_type, block_size, max_loop, max_patience)
        x_att = attacker.perform_search(x_, y_, syndict, BBM)
        num_queries = BBM.num_queries
        modif_rate = (torch.sum(x_att!=x_) / x_.shape[1]).item()
        succ = 1 if BBM.get_score(x_att,y_) >= 0 else 0 # 1 if Success else 0.
        return x_att.cuda(), num_queries, modif_rate, succ


"""
Search via Bayesian Optimization
===============

"""
from collections import defaultdict
import numpy as np
import torch
import random

from attack_codes.BayesOpt.acquisition.algorithm import kmeans_pp, acquisition_maximization_with_indices
from attack_codes.BayesOpt.historyboard import HistoryBoardProtein
from attack_codes.BayesOpt.surrogate_model.gp_model import MyGPModel

import copy
import time
from copy import deepcopy
import gc

def list_by_ind(l, ind):
    return [l[i] for i in ind]

class BayesAttack:
    """An attack based on Bayesian Optimization

    Args:
        kernel_name : kernel name. one of ['categorical']
        dpp_type : dpp type. one of ['no','no_one','dpp_posterior]
    """

    def __init__(self, kernel_name='categorical', block_policy='straight', dpp_type='dpp_posterior', block_size=40, max_loop=5, max_patience=20, print_option=False):
        
        self.kernel_name = kernel_name
        self.block_policy = block_policy
        self.dpp_type = dpp_type
        self.IBS = block_size
        self.max_loop = max_loop
        self.max_patience = max_patience
        self.print_option = print_option

        self.use_sod = True
        self.reg_coef = 0.0
        self.batch_size = 4
        self.memory_count = 0

        print('self.IBS :', self.IBS,  'max_loop :', self.max_loop)
        if dpp_type == 'no_one':
            self.batch_size = 1

    def perform_search(self, x, y, syndict, BBM):
        self.orig_X = x
        self.orig_Y = y
        self.hb = HistoryBoardProtein(orig_X=x, syndict=syndict)
        self.BBM = BBM

        if self.BBM.num_queries == self.BBM.query_budget or len(self.hb.target_indices)==0:
            return self.orig_X

        self.eval_and_add_datum(self.orig_X)
        if self.print_option:
            print("orig score : ",self.hb.eval_Y[0])
        
        # Initialize surrogate model wrapper.
        self.surrogate_model = MyGPModel(self.kernel_name)

        # Bayesian Optimization Loop
        # set best evaluated seq as initial seq.
        best_ind =  self.hb.best_in_history()[3][0]
        initial_seq = self.hb.eval_X[best_ind]
        eff_len = len(self.hb.target_indices)
        
        D_0 = []
        index_order = self.get_index_order_for_block_decomposition(initial_seq, self.block_policy)
        
        assert len(index_order) == eff_len, "something wrong"
        self.NB_INIT = int(np.ceil(eff_len / self.IBS))
        self.INDEX_DICT = defaultdict(list)
        self.HISTORY_DICT = defaultdict(list)
        self.BLOCK_QUEUE = [(0,int(i)) for i in range(self.NB_INIT)]

        center_seq = initial_seq
        center_ind = 0

        # init INDEX_DICT, HISTORY_DICT
        ALL_IND = index_order

        for KEY in self.BLOCK_QUEUE:
            self.INDEX_DICT[KEY] = deepcopy(ALL_IND[self.IBS*KEY[1]:self.IBS*(KEY[1]+1)])

        LOCAL_OPTIMUM = defaultdict(list)
        stage = -1
        while self.BLOCK_QUEUE:
            if self.BLOCK_QUEUE[0][0] != stage:
                self.BLOCK_QUEUE = self.update_queue(self.BLOCK_QUEUE, self.INDEX_DICT)
                stage += 1
            if not self.BLOCK_QUEUE: break

            KEY = self.BLOCK_QUEUE.pop(0)

            opt_indices = deepcopy(self.INDEX_DICT[KEY])
            fix_indices = list( set(list(range(eff_len))) - set(opt_indices) )

            self.HISTORY_DICT[KEY].append(center_ind)

            if not opt_indices: continue

            stage_init_ind = len(self.hb.eval_Y)
            stage_iter = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            ex_ball_size = 10000
            n_samples = int(stage_iter / len(opt_indices)) if len(opt_indices)<=3 else int(stage_iter / len(opt_indices)) * 2
            next_len = self.IBS                   

            if self.print_option:
                print("KEY : ", KEY, 'in', self.BLOCK_QUEUE)
                print("opt_indices : ", opt_indices)
                print("cur best")
                print(self.hb.best_in_history()[1][0][0].item(), self.hb.best_in_history()[2][0])

            # Exploration.
            prev_qr = len(self.hb.eval_Y)
            stage_call = 0 
            if self.BBM.num_queries >= self.BBM.query_budget: break
            stage_call, best_X = self.exploration_ball_with_indices(center_seq=center_seq,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices)
            if stage_call == -1:
                return best_X
            if len(self.hb.eval_Y) == prev_qr:
                if KEY[0] < self.max_loop: 
                    new = (KEY[0]+1, KEY[1])
                    self.BLOCK_QUEUE.append(new)
                    self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                continue
            if self.BBM.num_queries >= self.BBM.query_budget: break

            # union parent histories and prev best ind (center ind)
            parent_history = set(D_0)
            for n in range(KEY[0]):
                key = (int(n),KEY[1])
                parent_history |= set(self.HISTORY_DICT[key])

                inds = self.HISTORY_DICT[key]
                loc_fix_indices = list( set(list(range(eff_len))) - set(deepcopy(self.INDEX_DICT[key])) )
                if loc_fix_indices:
                    history = self.hb.eval_X_num[inds]
                    uniq = torch.unique(history[:,loc_fix_indices],dim=0)
                    assert uniq.shape[0] == 1, f'{uniq.shape},{uniq[:,:5]}'

            parent_history.add(center_ind)
            parent_history = list(parent_history)

            if self.use_sod:
                parent_history = self.subset_of_dataset(parent_history, stage_iter)
                assert len(parent_history) <= stage_iter, f'something wrong {stage_iter}, {len(parent_history)}'
            # Exploitation.
            num_candids = 10
            
            init_cent_indiced = deepcopy(self.hb.numbering_seq(center_seq))

            while stage_call < stage_iter:
                self.clean_memory_cache()

                t0 = time.time()
                self.surrogate_model.fit_partial(self.hb, list(range(eff_len)), stage_init_ind, prev_indices=parent_history) 
                t1 = time.time()

                best_inds = self.hb.topk_in_history_with_fixed_indices(len(self.hb.eval_Y), init_cent_indiced, fix_indices)[3]       
                for best_ind in best_inds:
                    if not (best_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                        break
                best_val = self.hb.eval_Y[best_inds[0]][0].item()
                best_seq = self.hb.eval_X[best_ind].view(1,-1)
                reference = best_val - self.reg_coef * torch.count_nonzero(self.hb.eval_X_num[best_inds[0]])
                #print(f"best of current KEY {KEY} :", best_val, self.hb._hamming(self.orig_X, best_seq))
                best_indiced = self.hb.numbering_seqs(best_seq)
                
                best_seqs = self.find_greedy_init_with_indices(cand_indices=best_indiced, max_radius=eff_len, num_candids=num_candids, reference=reference)
                best_candidates = acquisition_maximization_with_indices(best_seqs, opt_indices=opt_indices, batch_size=self.batch_size, stage=eff_len, hb=self.hb, surrogate_model=self.surrogate_model, kernel_name=self.kernel_name, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False, reg_coef=self.reg_coef)

                t2 = time.time()
                if type(best_candidates) == type(None):
                    LOCAL_OPTIMUM[tuple(opt_indices)].append(best_ind)
                    rand_indices = self.hb.nbd_sampler(best_indiced, self.batch_size, 2, 1, fix_indices=fix_indices)
                    best_candidates = [self.hb.seq_by_indices(inds) for inds in random.sample(list(rand_indices), self.batch_size)]                        

                if stage_call >= stage_iter or self.BBM.num_queries >= self.BBM.query_budget: break

                for best_candidate in best_candidates:
                    if self.eval_and_add_datum(best_candidate):
                        return self.final_exploitation(best_candidate, len(self.hb.eval_Y)-1)
                    stage_call += 1
                    if stage_call >= stage_iter or self.BBM.num_queries >= self.BBM.query_budget: break
                t3 = time.time()
                #print(t1-t0,  t2-t1, t3-t2)

            best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3] 
            
            for center_ind in best_inds:
                if not (center_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                    center_ind = int(center_ind)
                    break
            center_seq = self.hb.eval_X[center_ind]
            if self.BBM.num_queries >= self.BBM.query_budget: break

            if KEY[0] < self.max_loop: 
                new = (KEY[0]+1, KEY[1])
                self.BLOCK_QUEUE.append(new)
                self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
            if self.print_option:
                print(f"best of KEY {KEY} : ", self.hb.best_in_recent_history(len(self.hb.eval_X)-stage_init_ind)[1][0][0].item())
            continue
            
        best_ind = torch.argmax(self.hb.eval_Y.view(-1)).item()
        best_X = self.hb.eval_X[best_ind].view(1,-1)

        return best_X
    
    def subset_of_dataset(self, history, num_samples):
        if len(history) <= num_samples:
            return history
        else:
            history_X_num = self.hb.eval_X_num[history].numpy()
            _, selected_indices_ = kmeans_pp(history_X_num, num_samples, dist='hamming')
            history = [history[ind] for ind in selected_indices_]
            return history

    def ind_score(self, ind, beta, stage, ct):
        score = sum([float(beta[i]) for i in ind]) + 1e6 * stage + 1e-6 * ct
        return score

    def update_queue(self, Q, I):
        if Q[0][0] == 0:
            if self.print_option:
                print("first stage.")
            if len(Q)==1:
                return Q
            else:
                order = self.get_block_order(Q,I)
                Q_ = [Q[ind] for ind in order]
                return Q_
        self.fit_surrogate_model_by_block_history()

        if self.kernel_name == 'categorical':
            beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()+1e-6)
        elif self.kernel_name == 'categorical_horseshoe':
            beta = self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()
        for KEY in Q:
            if not I[KEY]:
                Q.remove(KEY)

        def f(KEY):
            return self.ind_score(I[KEY],beta,KEY[0],Q.index(KEY))
        Q_ = sorted(Q,key=f)
        if self.print_option:
            print(Q)
            print('->')
            print(Q_)
        return Q_

    def get_block_order(self, Q, I):
        index_scores = []
        for KEY in Q:
            inds = I[KEY]
            start, end = self.hb.target_indices[inds[0]], self.hb.target_indices[inds[-1]]
            del_seq = deepcopy(self.orig_X)
            del_seq = torch.cat([del_seq[:,:start],del_seq[:,end+1:]],dim=-1)
            score = self.BBM.get_score(del_seq,self.orig_Y)
            index_scores.append(score)
        index_scores = np.array(index_scores)
        index_order = (-index_scores).argsort()
        return index_order

    def get_index_order_for_block_decomposition(self, initial_seq, block_policy):
        if 'rand' == block_policy:
            index_order = list(range(len(self.hb.target_indices)))
            random.shuffle(index_order)
        elif 'straight' in block_policy:
            index_order = list(range(len(self.hb.target_indices)))
        elif 'beta' == block_policy:
            if self.kernel_name == 'categorical':
                beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale.detach().cpu()+1e-6)
            elif self.kernel_name == 'categorical_horseshoe':
                beta = self.surrogate_model.model.covar_module.base_kernel.lengthscale.detach().cpu()
            index_order = torch.argsort(beta)[0]
            if self.print_option:
                print(index_order)
        index_order = [int(i) for i in index_order]
        return index_order

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
            if self.eval_and_add_datum(rand_candidate):
                return -1, self.final_exploitation(rand_candidate, len(self.hb.eval_Y)-1)
            if self.BBM.num_queries >= self.BBM.query_budget:
                stage_call += self.hb.eval_Y.shape[0] - prev_len
                return stage_call, None

        # If any query were not evaluated, hardly sample non orig examples.
        if self.hb.eval_Y.shape[0] == prev_len:
            if self.print_option:
                print("hardsample.")
            center_indiced = self.hb.numbering_seq(center_seq)
            rand_indiced = copy.deepcopy(center_indiced)
            for ind in opt_indices:
                rand_indiced[ind] = int(random.sample(list(range(self.hb.reduced_n_vertices[ind]-1)), 1)[0] + 1)
            rand_candidate = self.hb.seq_by_indices(rand_indiced)
            if self.eval_and_add_datum(rand_candidate):
                return -1, self.final_exploitation(rand_candidate, len(self.hb.eval_Y)-1)
        stage_call += self.hb.eval_Y.shape[0] - prev_len
        return stage_call, None

    def find_greedy_init_with_indices(self, cand_indices, max_radius, num_candids, reference=None):
        ### Before Greedy Ascent Step ###
        # calculate acquisition
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming(self.hb.orig_seq, max_radius)
            reference = reference - self.reg_coef *  torch.count_nonzero(self.hb.eval_X_num[best_ind])
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference,reg_coef=self.reg_coef)
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        center_candidates = [self.hb.seq_by_indices(cand_indices[idx]) for idx in topk_indices]
        return center_candidates

    def eval_and_add_datum(self, seq, return_=True):
        if not seq in self.BBM.eval_cache:
            score = self.BBM.get_score(seq, self.orig_Y)
            self.hb.add_datum(seq, score)
            if return_ and score >= 0:
                return 1
            else:
                return 0
        else:
            return 0
    
    def final_exploitation(self, seq, ind):
        return self.final_exploitation_v3(seq, ind)

    def clean_memory_cache(self):
        if self.memory_count == 10:
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_count = 0
        else:
            self.memory_count += 1

    def final_exploitation_v3(self, seq, ind):
        if self.print_option:
            print("final exploitation")
        cur_seq, best_ind, eff_len, i, nbd_size = seq, ind, self.hb.eff_len_seq, 0, 2
        prev_radius = self.hb._hamming(self.orig_X, cur_seq)
        patience = self.max_patience

        whole_indices = [idx for idx in range(eff_len)] 
        _, sum_history = self.fit_surrogate_model_by_block_history()
        init_idx = len(self.hb.eval_Y)
        expl_info = []
        
        while True:
            self.clean_memory_cache

            max_radius = self.hb._hamming(self.orig_X, cur_seq) - 1
            if prev_radius == max_radius:
                if patience <= 0:
                    break
            else:
                patience = self.max_patience
                prev_radius = max_radius
                nbd_size = 2
            expl_info.append([i, max_radius, patience, self.max_patience, self.BBM.num_queries, self.BBM.query_budget])
            if self.print_option:
                print("final", i, max_radius)
            if max_radius == 0:
                return self.hb.eval_X[best_ind].view(1,-1)
            
            self.surrogate_model.fit_partial(self.hb, whole_indices, init_idx, sum_history)
            best_candidate = cur_seq

            # best in ball seq
            bib_seq, bib_score, _ = self.hb.best_of_hamming(seq=self.orig_X, distance=max_radius)

            best_indiced = self.hb.numbering_seq(best_candidate)
            bib_indiced = self.hb.numbering_seq(bib_seq)
            orig_indiced = self.hb.numbering_seq(self.orig_X)
            rand_indices = self.hb.subset_sampler(best_indiced, 300, nbd_size)

            cand_indices = torch.cat([orig_indiced.view(1,-1), best_indiced.view(1,-1), bib_indiced.view(1,-1), rand_indices], dim=0)
            cand_indices = torch.unique(cand_indices.long(),dim=0).float()
            center_candidates = self.find_greedy_init_with_indices(cand_indices, max_radius, num_candids=self.batch_size, reference=0.0)
  
            reference = self.hb.eval_Y[best_ind].item() - (max_radius + 1) * self.reg_coef
            best_candidates = acquisition_maximization_with_indices(center_candidates, opt_indices=whole_indices, batch_size=self.batch_size, stage=max_radius-1, hb=self.hb, surrogate_model=self.surrogate_model, kernel_name=self.kernel_name, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False, reg_coef=self.reg_coef)
            if best_candidates == None:
                if max_radius + 1 == nbd_size:
                    break
                else:
                    nbd_size += 1
                    continue

            tmp = 0.0
            tmp_modif = eff_len
            if self.BBM.num_queries >= self.BBM.query_budget: break
            for best_candidate in best_candidates:
                modif = self.hb._hamming(self.orig_X, best_candidate)
                if self.eval_and_add_datum(best_candidate):
                    if tmp < self.hb.eval_Y[-1].item() or (tmp == self.hb.eval_Y[-1].item() and tmp_modif > modif):
                        tmp = self.hb.eval_Y[-1].item()
                        tmp_modif = modif
                        cur_seq = best_candidate
                        best_ind = len(self.hb.eval_X) - 1 
                patience -= 1
                if self.BBM.num_queries >= self.BBM.query_budget or patience <= 0: break
            if self.BBM.num_queries >= self.BBM.query_budget or patience <= 0: break
            i += 1
        best_X = self.hb.eval_X[best_ind].view(1,-1)
        return best_X

    def block_history_dict(self):
        bhl = defaultdict(list)
        for KEY, INDEX in self.INDEX_DICT.items():
            HISTORY = self.HISTORY_DICT[KEY]
            bhl[KEY[1]].extend(HISTORY)
        for key in bhl:
            bhl[key] = list(dict.fromkeys(bhl[key]))
            opt_indices = list(range(key*self.IBS,min((key+1)*self.IBS,len(self.hb.reduced_n_vertices))))
            num_samples = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            bhl[key] = self.subset_of_dataset(bhl[key],num_samples)
            if self.print_option:
                print(f"num samples in {key} : {len(bhl[key])}")
        sum_history = []
        for key, l in bhl.items():
            sum_history.extend(l)
        return bhl, sum_history
    
    def fit_surrogate_model_by_block_history(self):
        eff_len = len(self.hb.target_indices)
        bhl, sum_history = self.block_history_dict()
        whole_indices = [idx for idx in range(eff_len)] # nonzero indices
        self.surrogate_model.fit_partial(self.hb, whole_indices, len(self.hb.eval_Y), sum_history)
        return bhl, sum_history

    _perform_search = perform_search
