from collections import defaultdict
from math import e
from typing import List
import torch
import copy
import random
import numpy as np
class HistoryBoard(object):
    def __init__(self, orig_X, n_vertices):
        self.orig_X = orig_X
        self.eval_X = None
        self.eval_Y = None 
        self.hamming_with_orig = None
        self.len_seq = orig_X.shape[-1]

        self.n_vertices = n_vertices
        self.target_indices = [ind for ind in range(self.len_seq) if self.n_vertices[ind]>1]
        self.reduced_n_vertices = [self.n_vertices[ind] for ind in self.target_indices]

        self.str2ind = {}

        self.eff_len_seq = len(self.target_indices)

        self.block_ind_cache = defaultdict(list)
        print("n_vertices", self.n_vertices)

    def add_datum(self, new_X, new_Y):
        if type(new_Y) == float: new_Y = torch.Tensor([new_Y])
        new_seq = self.reduce_seq(new_X)
        new_str = self.seq2str(new_X)
        if self.eval_X == None:
            self.eval_X = new_X.view(1,self.len_seq)
            self.eval_X_reduced = new_seq
            self.hamming_with_orig = [0]
            self.eval_Y = new_Y.view(1,1)
        else:
            self.eval_X = torch.cat([self.eval_X, new_X.view(1,self.len_seq)])
            self.eval_X_reduced = torch.cat([self.eval_X_reduced, new_seq])
            self.hamming_with_orig.append(int(torch.sum(new_seq !=0).item()))
            self.eval_Y = torch.cat([self.eval_Y, new_Y.view(1,1)])
        self.str2ind[new_str] = len(self.eval_X) - 1
        
    def add_data(self, new_Xs, new_Ys):
        new_seqs = self.reduce_seqs(new_Xs)
        if self.eval_X == None:
            init_ind = 0
            self.eval_X = new_Xs.view(-1,self.len_seq)
            self.eval_X_reduced = new_seqs
            new_modifs = [int(i) for i in torch.sum(new_seqs!=0,dim=1)]
            self.hamming_with_orig = new_modifs
            self.eval_Y = new_Ys.view(-1,1)
        else:
            init_ind = len(self.eval_X)
            self.eval_X = torch.cat([self.eval_X, new_Xs.view(-1,self.len_seq)])
            self.eval_X_reduced = torch.cat([self.eval_X_reduced, new_seqs])
            new_modifs = [int(i) for i in torch.sum(new_seqs!=0,dim=1)]
            self.hamming_with_orig = new_modifs
            self.eval_Y = torch.cat([self.eval_Y, new_Ys.view(-1,1)])
        for ct, new_X in enumerate(new_Xs):
            new_str = self.seq2str(new_X)
            self.str2ind[new_str] = init_ind + ct

    def seq2str(self, seq):
        seq = seq.view(-1)
        str_ = ''
        for i in seq:
            str_+=f'{i},'
        return str_

    def one_ball_from_orig(self):
        '''
            return - list of seqs of 1-hamming dist ball of orig_seq
        '''
        return self.one_ball_from_seq(self.orig_X)
    
    def one_ball_from_seq(self, x):
        '''
            return - list of seqs of 1-hamming dist ball from given seq
        '''
        x_num = self.reduce_seq(x)
        nbds = self.neighbors(x_num, radius=self.eff_len_seq)
        cands = []
        for i in range(len(nbds)):
            indices = nbds[i]
            cands.append(self.seq_by_indices(indices))
        return cands 

    def sample_ball_candidates_from_orig(self, n_samples, ball_size, fix_indices=None):
        '''
            return - sample n_samples seqs from ball_size-hamming dist ball of orig_seq
        '''
        return self.sample_ball_candidates_from_seq(self.orig_X, n_samples, ball_size, fix_indices=fix_indices)
    
    def sample_ball_candidates_from_seq(self, x, n_samples, ball_size, inner=True, only_indices=False, fix_indices=None):
        '''
            return - sample n_samples seqs from ball_size-hamming dist ball of given seq
        '''
        candidates = []
        x_num = self.reduce_seq(x)
        inradius = 1 if inner else ball_size
        nbds = self.nbd_sampler(x_num, n_samples, radius=ball_size, inradius=inradius, fix_indices=fix_indices)
        if only_indices:
            return nbds
        else:
            for i in range(n_samples):
                cur_indice = nbds[i]
                cur_seq = self.seq_by_indices(cur_indice)
                candidates.append(cur_seq)
            return candidates
    
    def seq_by_indices(self, indices):
        '''
            return - non-reduced version seq corresponding to indices.
        '''
        assert len(indices) == self.eff_len_seq or len(indices) == self.len_seq, "indices length should be one of target indices length or seq length"
        if len(indices) == self.len_seq:
            return torch.FloatTensor(indices).view(1,-1)
        elif len(indices) == self.eff_len_seq:
            new_seq = torch.zeros(self.len_seq)
            eff_new_seq = torch.FloatTensor(indices)
            new_seq[self.target_indices] = eff_new_seq
            return new_seq.view(1,-1)
    
    def reduce_seq(self, seq):
        '''
            return - numbered torch float Tensor of given seq
        '''
        if len(seq.shape) == 2:
            assert seq.shape[0] == 1 and seq.shape[1] == self.len_seq, f"input shape : {seq.shape[0]},{seq.shape[1]}, something wrong"
            seq_ = seq.view(-1)
        elif len(seq.shape) == 1:
            assert seq.shape[0] == self.len_seq, f"input shape : {seq.shape[0]}, something wrong"
            seq_ = seq
        else:
            assert f"input shape : {seq.shape}, something wrong"
        return seq_[self.target_indices].view(1,-1)

    def reduce_seqs(self, seqs):
        '''
            return - numbered torch float Tensor of given seqs
        '''
        numbered_seqs = []
        for seq in seqs:
            numbered_seq = self.reduce_seq(seq)
            numbered_seqs.append(numbered_seq)
        return torch.cat(numbered_seqs, dim=0) 

    def radius_sampler(self, n_samples, radius, inradius=1, fix_indices=None):
        '''
            return - concatenated floattensor sampled from inradius <= r <= radius - sampled ball
        '''
        return self.nbd_sampler(torch.zeros(self.eff_len_seq), n_samples, radius, inradius, fix_indices=fix_indices)
    
    def nbd_sampler(self, center_indices, n_samples, radius, inradius=1, fix_indices=None):
        '''
            return - concatenated floattensor sampled from inradius <= r <= radius - sampled ball
        '''
        if len(center_indices.shape) == 2: center_indices = center_indices.view(-1)
        assert len(center_indices) == self.eff_len_seq or len(center_indices) == self.len_seq, "indices length should be one of target indices length or seq length"
        if len(center_indices) == self.len_seq:
            center_indices = torch.FloatTensor([center_indices[ind] for ind in self.target_indices])

        nmin1 = np.array(self.reduced_n_vertices) - 1
        length = len(self.reduced_n_vertices)
        if inradius < length:
            mask_base = [0] * (length-inradius) + [1] * radius
        else:
            mask_base = [1] * length
        mask = np.stack([random.sample(mask_base,length) for _ in range(n_samples)])
        if not (fix_indices is None):
            mask[:,fix_indices] = 0

        x = np.random.randint(nmin1, size=(n_samples,length)) 
        for i in range(length):
            val = int(center_indices[i])
            x[:,i] = ((x[:,i] + (x[:, i] >= val) * 1.0) - val) * mask[:, i] + val
        y = torch.FloatTensor(x)
        return y
    
    def subset_sampler(self, center_indices, n_samples, radius, opt_indices=None):
        nonzero_indices = [ct for ct, ind in enumerate(center_indices.view(-1)) if ind > 0]
        if type(opt_indices) == list:
            nonzero_indices = [ct for ct in nonzero_indices if ct in opt_indices]

        nL = len(nonzero_indices)
        cands = []
        for _ in range(n_samples):
            if len(nonzero_indices) == 0:
                new_indices = copy.deepcopy(center_indices)
                return new_indices.view(1,-1)
            elif len(nonzero_indices) == 1:
                new_indices = copy.deepcopy(center_indices).view(1,-1)
                new_indices2 = self.reduce_seq(self.orig_X).view(1,-1)

                new_indices = torch.cat([new_indices,new_indices2], dim=0)                
                return new_indices
            else:
                I = random.sample(nonzero_indices, min(radius,nL))
                new_indices = copy.deepcopy(center_indices)
                new_indices[0][I] = 0
                cands.append(new_indices.view(1,-1))
        return torch.cat(cands,dim=0)
    
    def neighbors(self, x, radius, inradius=1, indices=None):
        '''
            returns all neighboring indices of x having l0 norm between inradius and radius
        '''
        if len(x.shape) == 2:
            x = x.view(-1)
        assert len(x) == self.len_seq or len(x) == self.eff_len_seq, "something wrong"
        if len(x) == self.len_seq:
            x = torch.FloatTensor([x[ind] for ind in self.target_indices])
        nbds = x.new_empty((0, x.numel()))
        nbd = self._cartesian_neighbors(x, radius, inradius=inradius, indices=indices)
        nbds = torch.cat([nbds, nbd])
        return nbds

    def _cartesian_neighbors(self, x, radius, inradius=1, indices=None):
        '''
            returns all neighboring vertices on product space of categorical variable.
        '''
        neighbor_list = []
        if type(indices) is type(None):
            traversal_indices = list(range(self.eff_len_seq))
        else:
            traversal_indices = indices
        #for i in range(self.eff_len_seq):
        for i in traversal_indices:
            all_indices = list(range(self.reduced_n_vertices[i]))
            nbd_i_elm = torch.FloatTensor(list(set(all_indices) - set([int(x[i])])))
            nbd_i = x.repeat((nbd_i_elm.numel(), 1))
            nbd_i[:, i] = nbd_i_elm
            neighbor_list.append(nbd_i)
        neighbor_list = torch.cat( neighbor_list, dim=0)
        neighbor_list = self._filter_by_radius_vectorized(neighbor_list, radius, inradius=inradius)
        return neighbor_list
    
    def _filter_by_radius_vectorized(self, xs, radius, inradius=1):
        xs_nonzero_mask = (torch.abs(xs)!=0) * 1.0 
        num_nonzero_vec = torch.sum(xs_nonzero_mask,dim=1)
        _, length = xs.shape
        if inradius==0 or inradius>=length:
            bool_vec = num_nonzero_vec <= radius # False means that index would be filtered.
        else:
            bool_vec = (num_nonzero_vec <= radius) * (num_nonzero_vec >= inradius) 
        filtered_indices = bool_vec.nonzero().view(-1)
        filtered_candidates = xs[filtered_indices]
        return filtered_candidates
    
    def best_of_hamming(self, seq, distance, min_d=0):
        targets = []
        modifs = self.hamming_with_orig
        for idx, eval_seq in enumerate(self.eval_X):
            hamming_distance = modifs[idx]
            if hamming_distance <= distance and hamming_distance >= min_d:
                targets.append([idx, self.eval_Y[idx][0]])
        if len(targets):
            best_ind, best_score = sorted(targets, key = lambda x: -x[1])[0]
            return self.eval_X[best_ind], best_score, best_ind
        else:
            return None, -1, None
    
    def best_of_hamming_orig(self, distance):
        targets = []
        for idx, modif in enumerate(self.hamming_with_orig):
            hamming_distance = modif
            if hamming_distance <= distance:
                targets.append([idx, self.eval_Y[idx][0]])
        if len(targets):
            best_ind, best_score = sorted(targets, key = lambda x: -x[1])[0]
            return self.eval_texts[best_ind], best_score, best_ind
        else:
            return None, -1, None

    def best_in_history(self):
        return self.topk_in_history(1)
            
    def topk_in_history_with_fixed_indices(self, k, cur_indices, fix_indices):
        import time
        cur_inds = cur_indices.view(-1)
        hist_inds = self.eval_X_reduced
        target_inds = (cur_inds[fix_indices] == hist_inds[:,fix_indices]).all(dim=1).nonzero(as_tuple=True)[0]
        
        y = self.eval_Y[target_inds].view(-1)
        _, indices_ = torch.topk(y, min(y.shape[0],k))
        indices = [int(target_inds[ind].item()) for ind in indices_]
        topk_X = self.eval_X[indices]
        topk_Y = self.eval_Y[indices]
        topk_seqs = self.eval_X[indices]
        modif = [self.hamming_with_orig[ind] for ind in indices]
        return topk_X, topk_Y, modif, indices

    def topk_in_history(self, k):
        _, indices = torch.topk(self.eval_Y.view(-1), min(self.eval_Y.shape[0],k))
        topk_X = self.eval_X[indices]
        topk_Y = self.eval_Y[indices]
        topk_seqs = self.eval_X[indices]
        modif = [self.hamming_with_orig[ind] for ind in indices]
        return topk_X, topk_Y, modif, indices
    
    def best_in_recent_history(self, num):
        return self.topk_in_recent_history(1, num)

    def topk_in_recent_history(self, k, num):
        _, indices = torch.topk(self.eval_Y[-num:].view(-1), min(self.eval_Y.shape[0],k))
        topk_X = self.eval_X[-num:][indices]
        topk_Y = self.eval_Y[-num:][indices]
        topk_seqs = self.eval_X[-num:][indices]
        modif = [self.hamming_with_orig[ind] for ind in indices]
        indices = [-num + ind for ind in indices]
        return topk_X, topk_Y, modif, indices

    def _hamming(self, seq1, seq2):
        return sum([w1!=w2 for w1, w2 in zip(seq1[0], seq2[0])])

    def _hammings_from_orig(self, seqs):
        modifs = torch.sum(self.orig_X!=seqs,dim=1)
        modifs = [m.item() for m in modifs]
        return modifs
    
    def is_seq_in_hb(self, seq):
        seq_ = seq.view(1,-1)
        bv = torch.all(seq_ == self.eval_X,dim=1)
        return torch.any(bv)
    
    def get_seq_ind(self, seq):
        str_ = self.seq2str(seq)
        ind = self.str2ind[str_]   
        return ind