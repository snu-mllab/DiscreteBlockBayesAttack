"""
Search via Bayesian Optimization
===============

"""
from collections import defaultdict
import numpy as np
import torch
import random

from bayesopt.acquisition.algorithm import kmeans_pp, acquisition_maximization_with_indices
from bayesopt.historyboard import HistoryBoard
from bayesopt.surrogate_model.gp_model import MyGPModel

import copy
import time
from copy import deepcopy
import gc

def list_by_ind(l, ind):
    return [l[i] for i in ind]

class BlockBayesAttack:
    """An attack based on Bayesian Optimization

    Args:
        dpp_type : dpp type. one of ['no','dpp_posterior]
        post_opt : ['', 'v2', 'v3', 'v4']
    """

    def __init__(self, block_size=40, batch_size=4, update_step=1, max_patience=50, post_opt='', use_sod=True, dpp_type='dpp_posterior', max_loop=5, niter=1):
        
        self.block_size = block_size
        self.batch_size = batch_size
        self.update_step = update_step
        self.max_patience = max_patience
        self.post_opt = post_opt 
        self.use_sod = use_sod
        self.dpp_type = dpp_type
        self.max_loop = max_loop
        self.batch_size = batch_size
        self.niter = niter

        self.memory_count = 0
    
    def check_query_const(self):
        if self.BBM.num_queries == self.BBM.query_budget:
            return True
        else:
            return False

    def init_before_loop(self):
        # Bayesian Optimization Loop
        # set best evaluated seq as initial seq.
        best_ind = self.hb.best_in_history()[3][0]
        initial_seq = self.hb.eval_X[best_ind]
        self.eff_len = self.hb.eff_len_seq
        
        D_0 = []
        index_order = self.get_index_order_for_block_decomposition()

        self.NB_INIT = int(np.ceil(self.eff_len / self.block_size))
        self.INDEX_DICT = defaultdict(list)
        self.HISTORY_DICT = defaultdict(list)
        self.BLOCK_QUEUE = [(0,int(i)) for i in range(self.NB_INIT)]

        center_seq = initial_seq
        center_ind = 0

        # init INDEX_DICT, HISTORY_DICT
        ALL_IND = index_order
        for KEY in self.BLOCK_QUEUE:
            self.INDEX_DICT[KEY] = deepcopy(ALL_IND[self.block_size*KEY[1]:self.block_size*(KEY[1]+1)])

        LOCAL_OPTIMUM = defaultdict(list)
        stage = -1

        return D_0, center_seq, center_ind, LOCAL_OPTIMUM, stage

    def init_in_loop(self, KEY, center_ind):
        self.clean_memory_cache()
        opt_indices = deepcopy(self.INDEX_DICT[KEY])
        fix_indices = list( set(list(range(self.eff_len))) - set(opt_indices) )

        self.HISTORY_DICT[KEY].append(int(center_ind))

        if not opt_indices: cont = True
        else: cont = False

        stage_init_ind = len(self.hb.eval_Y)
        stage_iter = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
        ex_ball_size = 10000
        n_samples = int(stage_iter / len(opt_indices)) if len(opt_indices)<=3 else int(stage_iter / len(opt_indices)) * 2
        next_len = self.block_size

        return opt_indices, fix_indices, stage_init_ind, stage_iter, ex_ball_size, n_samples, next_len, cont

    def perform_search(self, x, n_vertices, BBM):
        self.orig_X = x
        self.hb = HistoryBoard(orig_X=x, n_vertices=n_vertices)
        self.BBM = BBM
        self.eff_len = self.hb.eff_len_seq

        print("query budget is ", self.BBM.query_budget)    
        if self.check_query_const() or len(self.hb.target_indices)==0: return x

        self.eval_and_add_datum(self.orig_X)
        
        # Initialize surrogate model wrapper.
        self.surrogate_model = MyGPModel(niter=self.niter)

        D_0, center_seq, center_ind, LOCAL_OPTIMUM, stage = self.init_before_loop()

        while self.BLOCK_QUEUE:
            if self.BLOCK_QUEUE[0][0] != stage:
                self.BLOCK_QUEUE = self.update_queue(self.BLOCK_QUEUE, self.INDEX_DICT)
                stage += 1
            if not self.BLOCK_QUEUE: break

            KEY = self.BLOCK_QUEUE.pop(0)
            opt_indices, fix_indices, stage_init_ind, stage_iter, ex_ball_size, n_samples, next_len, cont = self.init_in_loop(KEY,center_ind)
            if cont: continue

            # Exploration.
            prev_qr = len(self.hb.eval_Y)
            stage_call = 0 

            if self.check_query_const(): break
            stage_call, fX, X = self.exploration_ball_with_indices(center_seq=center_seq,n_samples=n_samples,ball_size=ex_ball_size,stage_call=stage_call, opt_indices=opt_indices, KEY=KEY, stage_init_ind=stage_init_ind)

            if stage_call == -1:
                return X
            if len(self.hb.eval_Y) == prev_qr:
                if KEY[0] < self.max_loop: 
                    new = (KEY[0]+1, KEY[1])
                    self.BLOCK_QUEUE.append(new)
                    self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                continue
            if self.check_query_const(): break

            # union parent histories and prev best ind (center ind)
            parent_history = set(D_0)
            for n in range(KEY[0]):
                key = (int(n),KEY[1])
                parent_history |= set(self.HISTORY_DICT[key])

                inds = self.HISTORY_DICT[key]
                loc_fix_indices = list( set(list(range(self.eff_len))) - set(deepcopy(self.INDEX_DICT[key])) )
                if loc_fix_indices:
                    history = self.hb.eval_X_reduced[inds]
                    uniq = torch.unique(history[:,loc_fix_indices],dim=0)
                    assert uniq.shape[0] == 1, f'{uniq.shape},{uniq[:,:5]}'
                    print(uniq[:,:5], fix_indices)

            parent_history.add(center_ind)
            parent_history = list(parent_history)

            if self.use_sod:
                parent_history = self.subset_of_dataset(parent_history, stage_iter)
                assert len(parent_history) <= stage_iter, f'something wrong {stage_iter}, {len(parent_history)}'
            
            # Exploitation.
            num_candids = 10
            init_cent_indiced = deepcopy(self.hb.reduce_seq(center_seq))

            print("before loop")
            count = 0
            prev_size = len(self.hb.eval_Y)
            iter_patience = 5
            while stage_call < stage_iter and iter_patience:
                self.clean_memory_cache()
                if prev_size == len(self.hb.eval_Y):
                    iter_patience -= 1
                else:
                    iter_patience = 5
                    prev_size = len(self.hb.eval_Y)
                self.surrogate_model.fit_partial(self.hb, list(range(self.eff_len)), stage_init_ind, prev_indices=parent_history) 

                if count  % self.update_step == 0:
                    best_inds = self.hb.topk_in_history_with_fixed_indices(len(self.hb.eval_Y), init_cent_indiced, fix_indices)[3]          
                    for best_ind in best_inds:
                        if not (best_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                            break

                    best_val = self.hb.eval_Y[best_inds[0]][0].item()
                    best_seq = [self.hb.eval_X[best_ind]]
                    reference = best_val
                    best_indiced = self.hb.reduce_seqs(best_seq)
                    
                    best_seqs = self.find_greedy_init_with_indices_v2(cand_indices=best_indiced, max_radius=self.eff_len, num_candids=num_candids, reference=reference)
                best_candidates = acquisition_maximization_with_indices(best_seqs, opt_indices=opt_indices, batch_size=self.batch_size, stage=self.eff_len, hb=self.hb, surrogate_model=self.surrogate_model, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False)

                if type(best_candidates) == type(None):
                    LOCAL_OPTIMUM[tuple(opt_indices)].append(best_ind)
                    rand_indices = self.hb.nbd_sampler(best_indiced, self.batch_size, 2, 1, fix_indices=fix_indices)
                    best_candidates = [self.hb.seq_by_indices(inds) for inds in random.sample(list(rand_indices), self.batch_size)]                        

                if stage_call >= stage_iter or self.check_query_const(): break

                prev_len = len(self.hb.eval_Y) 
                if self.eval_and_add_data(best_candidates):
                    self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                    X = self.final_exploitation(self.hb.eval_X[-1], len(self.hb.eval_Y)-1)
                    return X
                stage_call += len(self.hb.eval_Y) - prev_len
                if stage_call >= stage_iter or self.check_query_const(): break
                count += 1

            best_inds = self.hb.topk_in_history(len(self.hb.eval_Y))[3] 
            
            for center_ind in best_inds:
                if not (center_ind in LOCAL_OPTIMUM[tuple(opt_indices)]):
                    center_ind = int(center_ind)
                    break
            center_seq = self.hb.eval_X[center_ind]
            if self.check_query_const(): break

            if KEY[0] < self.max_loop: 
                print("280 line.")
                print(KEY[0], self.max_loop)
                new = (KEY[0]+1, KEY[1])
                self.BLOCK_QUEUE.append(new)
                self.INDEX_DICT[new] = deepcopy(opt_indices[:next_len])
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
            print(f"best of KEY {KEY} : ", self.hb.best_in_recent_history(len(self.hb.eval_X)-stage_init_ind)[1][0][0].item())
            continue
        
        if (self.BBM.num_queries < self.BBM.query_budget) and (not 'nogreed' in self.post_opt):
            # Greedy step.
            print("greedy step !!!")
            best_ind = self.hb.topk_in_history(1)[3][0]
            best_score = self.hb.eval_Y[best_ind][0].item()
            self.fit_surrogate_model_by_block_history()
            index_order = self.get_index_order_for_block_decomposition('beta')
            print("before greedy step!!")
            for i in range(10):
                if i>0:
                    orig_indiced = self.hb.reduce_seq(self.orig_X)
                    rand_indices = self.hb.nbd_sampler(orig_indiced, 1, self.eff_len, 1, fix_indices=[])
                    rand_seqs = [self.hb.seq_by_indices(indices) for indices in rand_indices]
                    scores = self.BBM.get_scores(rand_seqs)
                    if self.check_query_const(): break
                    if scores:
                        max_ind = torch.argmax(scores)
                        center_seq = rand_seqs[max_ind]
                        if scores[max_ind] >= 0: return center_seq
                        search_over = False
                    else:
                        search_over = True
                else:
                    center_seq = self.hb.eval_X[best_ind]
                    search_over = False
                if not search_over:
                    best_X, best_score, is_nice = self.greedy_step(center_seq, index_order)
                    print('best score in final greedy loop', best_score)
                    if is_nice:
                        return best_X
                else:
                    break
            print("after greedy step!!")
            
        if self.BBM.query_budget < float('inf'):
            Ys = self.hb.eval_Y[:self.BBM.query_budget]
        else:
            Ys = self.hb.eval_Y
        max_ind = torch.argmax(Ys)
        return self.hb.eval_X[max_ind]
    
    def subset_of_dataset(self, history, num_samples):
        if len(history) <= num_samples:
            return history
        else:
            history_X_reduced = self.hb.eval_X_reduced[history].numpy()
            _, selected_indices_ = kmeans_pp(history_X_reduced, num_samples, dist='hamming')
            history = [history[ind] for ind in selected_indices_]
            return history

    def greedy_step(self, seq, index_order, is_shuffle=True):
        best_seq = seq
        best_indiced = self.hb.reduce_seq(seq)
        best_score = self.BBM.get_score([best_seq])

        order = deepcopy(index_order)
        print('initial_score ', best_score)
        while True:
            prev_best_seq = best_seq
            if is_shuffle:
                random.shuffle(order)
            for ind in order:
                nv = self.hb.reduced_n_vertices[ind]
                candids = [deepcopy(best_indiced) for i in range(nv)] 
                for i, cand in enumerate(candids):
                    cand[ind] = i 
                candid_seqs = [self.hb.seq_by_indices(cand) for cand in candids]
                candid_scores = self.BBM.get_scores(candid_seqs)
                max_ind = torch.argmax(candid_scores)
                if self.check_query_const(): break
                if candid_scores[max_ind]>=0: return candid_seqs[max_ind], candid_scores[max_ind], True
                if best_score < candid_scores[max_ind]:
                    best_seq = candid_seqs[max_ind]
                    best_indiced = self.hb.reduce_seq(best_seq)
                    best_score = candid_scores[max_ind]
                print(ind, best_score)
            if torch.all(prev_best_seq == best_seq):
                break 
        return best_seq, best_score, False

    def ind_score(self, ind, beta, stage):
        score = sum([float(beta[i]) for i in ind]) + 1e6 * stage
        return score

    def update_queue(self, Q, I):
        if Q[0][0] == 0:
            print("first stage.")
            if len(Q)==1:
                return Q
            else:
                order = self.BBM.get_initial_block_order()
                Q_ = [Q[ind] for ind in order]
                return Q_
        if len(Q)==1:
            return Q
        self.fit_surrogate_model_by_block_history()

        beta = 1/(self.surrogate_model.model.covar_module.base_kernel.lengthscale[0].detach().cpu()+1e-6)
    
        for KEY in Q:
            if not I[KEY]:
                Q.remove(KEY)

        def f(KEY):
            return self.ind_score(I[KEY],beta,KEY[0])

        Q_ = sorted(Q,key=f)
        print(Q)
        print('->')
        print(Q_)
        return Q_

    def get_index_order_for_block_decomposition(self):
        index_order = list(range(len(self.hb.target_indices)))
        index_order = [int(i) for i in index_order]
        return index_order

    def exploration_ball_with_indices(self, center_seq, n_samples, ball_size, stage_call, opt_indices, KEY, stage_init_ind):
        if n_samples == 0:
            print(1)
            return stage_call, None, None
        fix_indices = list(set(list(range(len(self.hb.target_indices)))) - set(opt_indices))
        prev_len = self.hb.eval_Y.shape[0]
        rand_candidates = self.hb.sample_ball_candidates_from_seq(center_seq, n_samples=n_samples, ball_size=ball_size, fix_indices=fix_indices)

        if self.eval_and_add_data(rand_candidates):
            fX = self.hb.eval_X[-1]
            best_candidate = self.hb.eval_X[-1]
            return -1, fX, self.final_exploitation(best_candidate, len(self.hb.eval_Y)-1)

        if self.check_query_const():
            stage_call += self.hb.eval_Y.shape[0] - prev_len
            return stage_call,None, None

        # If any query were not evaluated, hardly sample non orig examples.
        if self.hb.eval_Y.shape[0] == prev_len:
            center_indiced = self.hb.reduce_seq(center_seq)
            rand_indiced = copy.deepcopy(center_indiced)
            for ind in opt_indices:
                rand_indiced[ind] = int(random.sample(list(range(self.hb.reduced_n_vertices[ind]-1)), 1)[0] + 1)
            rand_candidate = self.hb.seq_by_indices(rand_indiced)
            if self.eval_and_add_datum(rand_candidate):
                self.HISTORY_DICT[KEY].extend([ int(i) for i in range(stage_init_ind, len(self.hb.eval_X))])
                return -1, self.final_exploitation(rand_candidate, len(self.hb.eval_Y)-1)
        stage_call += self.hb.eval_Y.shape[0] - prev_len
        return stage_call, None, None

    # CERT
    def find_greedy_init_with_indices_v2(self, cand_indices, max_radius, num_candids, reference=None):
        ### Before Greedy Ascent Step ###
        # calculate acquisition
        if reference is None:
            _, reference, best_ind = self.hb.best_of_hamming_orig(distance=max_radius)
        ei = self.surrogate_model.acquisition(cand_indices, bias=reference)
        topk_values, topk_indices = torch.topk(ei, min(len(ei),num_candids))
        center_indices_list = [cand_indices[idx].view(1,-1) for idx in topk_indices]
        return center_indices_list

    def eval_and_add_datum(self, seq):
        if not self.hb.is_seq_in_hb(seq):
            score = self.BBM.get_score(seq)
            self.hb.add_datum(seq, score)
            if score >= 0:
                return 1
            else:
                return 0

    def eval_and_add_data(self, seqs):
        scores = self.BBM.get_scores(seqs)
        for seq, score in zip(seqs, scores):
            if not self.hb.is_seq_in_hb(seq):
                self.hb.add_datum(seq, score)
            if score >= 0:
                return 1
        return 0

    def eval_and_add_data_best_ind(self, seqs, cur_seq, best_ind, tmp, tmp_modif, patience):
        scores = self.BBM.get_scores(seqs)

        for seq, score in zip(seqs,scores):
            if not self.hb.is_seq_in_hb(seq):
                self.hb.add_datum(seq, score)
                if score >= 0:
                    modif = self.hb._hamming(self.orig_X, seq)
                    if tmp < self.hb.eval_Y[-1].item() or (tmp == self.hb.eval_Y[-1].item() and tmp_modif > modif):
                        tmp = self.hb.eval_Y[-1].item()
                        tmp_modif = modif
                        cur_seq = seq
                        best_ind = len(self.hb.eval_X) - 1 
            patience -= 1
        return cur_seq, best_ind, patience

    def final_exploitation(self, seq, ind):
        if 'v3' in self.post_opt:
            return self.final_exploitation_v3(seq, ind)
        elif 'v2' in self.post_opt:
            return self.final_exploitation_v2(seq, ind)[0]
        elif 'v4' in self.post_opt:
            return self.final_exploitation_v4(seq, ind)
        else:
            return self.hb.eval_X[ind].view(1,-1)

    def final_exploitation_v4(self, seq, ind):
        init_ind = len(self.hb.eval_Y)
        v2_ind = self.final_exploitation_v2(seq, ind)[1]
        v2_seq = self.hb.eval_X[v2_ind].view(1,-1)

        forced_inds = [int(i) for i in range(init_ind, len(self.hb.eval_Y))]
        return self.final_exploitation_v3(v2_seq, v2_ind, forced_inds)

    def final_exploitation_v2(self, seq, ind):
        print("final exploitation")
        cur_seq = seq
        cur_ind = ind
        cur_score = self.hb.eval_Y[cur_ind].item()
        cur_indices = self.hb.reduce_seq(cur_seq)
        nonzero_indices = [ct for ct, ind in enumerate(cur_indices) if ind > 0]

        if len(nonzero_indices)==1:
            print(0)
            return self.hb.eval_X[cur_ind].view(1,-1), cur_ind

        imps = []
        indices = []
        scores = []
        for ct, idx in enumerate(nonzero_indices):
            new_indices = copy.deepcopy(cur_indices)
            new_indices[idx] = 0
            new_seq = self.hb.seq_by_indices(new_indices)
            if self.check_query_const(): break
            self.eval_and_add_datum(new_seq)
            if self.hb.is_seq_in_hb(new_seq):
                new_ind = self.hb.get_seq_ind(new_seq)
            else:
                new_ind = -1
            new_score = self.hb.eval_Y[new_ind].item()
            imps.append(cur_score - new_score)
            scores.append(new_score)
            indices.append(new_indices)
        order = np.argsort(imps)
        if len(order)==0:
            return self.hb.eval_X[cur_ind].view(1,-1), cur_ind
        first_idx = order[0]
        prev_indices = indices[first_idx]
        prev_score = scores[first_idx]
        prev_idx = cur_ind
        if prev_score < 0:
            return self.hb.eval_X[ind].view(1,-1), ind
        for idx in order[1:]:
            if self.check_query_const(): break
            new_indices = copy.deepcopy(prev_indices)
            new_indices[nonzero_indices[idx]] = 0
            new_seq = self.hb.seq_by_indices(new_indices)
            if self.eval_and_add_datum(new_seq):
                prev_indices = new_indices
                prev_score = self.hb.eval_Y[-1].item()  
                prev_idx = len(self.hb.eval_Y)-1
                continue
        return self.hb.eval_X[prev_idx].view(1,-1), prev_idx
    

    def final_exploitation_v3(self, seq, ind, forced_inds=[]):
        print("final exploitation")
        cur_seq = seq
        best_ind = ind
        self.eff_len = self.hb.eff_len_seq
        max_patience = self.max_patience
        prev_radius = self.hb._hamming(self.orig_X, cur_seq)
        patience = max_patience

        i=0
        nbd_size = 2
        opt_indices = [idx for idx in range(self.eff_len)] 
        whole_indices = [idx for idx in range(self.eff_len)] 
        _, sum_history = self.fit_surrogate_model_by_block_history(forced_inds)
        init_idx = len(self.hb.eval_Y)

        while True:
            self.clean_memory_cache()
            max_radius = self.hb._hamming(self.orig_X, cur_seq) - 1
            if prev_radius == max_radius:
                if patience <= 0:
                    break
            else:
                patience = max_patience
                prev_radius = max_radius
                nbd_size = 2
            print("final", i, max_radius)
            if max_radius == 0:
                return self.hb.eval_X[best_ind].view(1,-1)
            
            self.surrogate_model.fit_partial(self.hb, whole_indices, init_idx, sum_history)
            best_candidate = cur_seq

            # best in ball seq
            bib_seq, bib_score, _ = self.hb.best_of_hamming_orig(distance=max_radius)

            best_indiced = self.hb.reduce_seq(best_candidate)
            bib_indiced = self.hb.reduce_seq(bib_seq)  
            orig_indiced = self.hb.reduce_seq(self.orig_X)
            rand_indices = self.hb.subset_sampler(best_indiced, 300, nbd_size)

            cand_indices = torch.cat([orig_indiced.view(1,-1), best_indiced.view(1,-1), bib_indiced.view(1,-1), rand_indices], dim=0)
            cand_indices = torch.unique(cand_indices.long(),dim=0).float()
            center_candidates = self.find_greedy_init_with_indices_v2(cand_indices, max_radius, num_candids=self.batch_size, reference=0.0)  
            reference = self.hb.eval_Y[best_ind].item()
            best_candidates = acquisition_maximization_with_indices(center_candidates, opt_indices=opt_indices, batch_size=self.batch_size, stage=max_radius-1, hb=self.hb, surrogate_model=self.surrogate_model, reference=reference, dpp_type=self.dpp_type, acq_with_opt_indices=False)
            if best_candidates == None:
                if max_radius + 1 == nbd_size:
                    break
                else:
                    nbd_size += 1
                    continue

            tmp = 0.0
            tmp_modif = self.eff_len
            if self.check_query_const(): break

            cur_seq, best_ind, patience = self.eval_and_add_data_best_ind(best_candidates, cur_seq, best_ind, tmp, tmp_modif, patience)
            if self.check_query_const() or patience <= 0: break

            if self.check_query_const() or patience <= 0: break
            i += 1
        return self.hb.eval_X[best_ind].view(1,-1)

    def block_history_dict(self, forced_inds=[]):
        bhl = defaultdict(list)
        print("func block_history_dict")
        print("self.history_dict")
        print(self.HISTORY_DICT)
        for KEY, INDEX in self.INDEX_DICT.items():
            HISTORY = self.HISTORY_DICT[KEY]
            bhl[KEY[1]].extend(HISTORY)
        for key in bhl:
            bhl[key] = list(dict.fromkeys(bhl[key]))
            opt_indices = list(range(key*self.block_size,min((key+1)*self.block_size,len(self.hb.reduced_n_vertices))))
            num_samples = sum([self.hb.reduced_n_vertices[ind]-1 for ind in opt_indices]) 
            bhl[key] = self.subset_of_dataset(bhl[key],num_samples)
            print(f"num samples in {key} : {len(bhl[key])}")
        sum_history = copy.deepcopy(forced_inds)
        for key, l in bhl.items():
            sum_history.extend(l)
        return bhl, sum_history
    
    def fit_surrogate_model_by_block_history(self, forced_inds=[]):
        self.eff_len = len(self.hb.target_indices)
        bhl, sum_history = self.block_history_dict(forced_inds)
        whole_indices = [idx for idx in range(self.eff_len)] # nonzero indices
        self.surrogate_model.fit_partial(self.hb, whole_indices, len(self.hb.eval_Y), sum_history)
        return bhl, sum_history

    def clean_memory_cache(self,debug=False):
        # Clear garbage cache for memory.
        if self.memory_count == 10:
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_count = 0
        else:
            self.memory_count += 1   
        if debug:
            print(torch.cuda.memory_allocated(0))
    @property
    def is_black_box(self):
        return True

    _perform_search = perform_search
