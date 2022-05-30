from collections import defaultdict
from math import e
from typing import List
import torch
import copy
import random
import time
import numpy as np
from attack_codes.BayesOpt.historyboard.hb import HistoryBoard
class HistoryBoardFixed(HistoryBoard):
    def __init__(self, orig_text, transformer, enc_model):
        self.orig_text = orig_text
        self.orig_text.attack_attrs['modified_indices'] = set()
        self.transformer = transformer
        self.enc_model = enc_model

        self.eval_texts = []
        self.eval_texts_str = []
        self.eval_results = []
        self.eval_X = None
        self.eval_X_num = None
        self.eval_Y = None 
        self.numbering_dict = dict()
        self.len_text = len(orig_text.words)

        self.word_substitution_cache = [[] for _ in range(self.len_text)]
        self.investigate_fixed_word_candids()
        self.n_vertices = [len(w_candids) for w_candids in self.word_substitution_cache]
        self.target_indices = [ind for ind in range(self.len_text) if self.n_vertices[ind]>1]
        self.reduced_n_vertices = [self.n_vertices[ind] for ind in self.target_indices]
        print("n_vertices", self.n_vertices)
    
    def investigate_fixed_word_candids(self):
        for ind in range(self.len_text):
            transformed_texts = self.transformer(self.orig_text, original_text=self.orig_text, indices_to_modify=[ind])
            self.word_substitution_cache[ind].append(self.orig_text.words[ind])
            for txt in transformed_texts:
                if txt.words[ind] in self.word_substitution_cache[ind]:
                    continue
                else:
                    self.word_substitution_cache[ind].append(txt.words[ind])

    def add_datum(self, new_text, result, discrete=False):
        if self.eval_X == None:
            self.eval_texts = [new_text]
            self.eval_texts_str = [new_text.text]
            self.eval_results = [result]
            self.eval_X = torch.Tensor(self.enc_model.encode([new_text.text]))
            self.eval_Y = torch.Tensor([[result.score]])
            self.eval_X_num = self.numbering_text(new_text).view(1,-1)
        else:
            self.eval_texts.append(new_text)
            self.eval_texts_str.append(new_text.text)
            self.eval_results.append(result)
            self.eval_X = torch.cat([self.eval_X, torch.Tensor(self.enc_model.encode([new_text.text]))])
            self.eval_Y = torch.cat([self.eval_Y, torch.Tensor([[result.score]])])
            self.eval_X_num = torch.cat([self.eval_X_num, self.numbering_text(new_text).view(1,-1)])
        
    def add_data(self, new_texts, results, discrete=False):
        if self.eval_X == None:
            self.eval_texts = [new_text for new_text in new_texts]
            self.eval_texts_str = [new_text.text for new_text in new_texts]
            self.eval_results = [result for result in results]
            self.eval_X = torch.Tensor(self.enc_model.encode([new_text.text for new_text in new_texts]))
            self.eval_Y = torch.Tensor([[result.score for result in results]])
            self.eval_X_num = self.numbering_texts(new_texts)
        else:
            self.eval_texts.extend(new_texts)
            self.eval_texts_str.extend([new_text.text for new_text in new_texts])
            self.eval_results.extend(results)
            self.eval_X = torch.cat([self.eval_X, torch.Tensor(self.enc_model.encode([new_text.text for new_text in new_texts]))])
            self.eval_Y = torch.cat([self.eval_Y, torch.Tensor([[result.score for result in results]])])
            self.eval_X_num = torch.cat([self.eval_X_num, self.numbering_texts(new_texts)])

    def one_ball_from_orig(self):
        '''
            return - list of texts of 1-hamming dist ball of orig_text
        '''
        return self.one_ball_from_text(self.orig_text)
    
    def one_ball_from_text(self, text):
        '''
            return - list of texts of 1-hamming dist ball from given text
        '''
        x = self.numbering_text(text)
        nbds = self.neighbors(x, radius=len(self.target_indices))
        cands = []
        for i in range(len(nbds)):
            indices = nbds[i]
            cands.append(self.text_by_indices(indices))
        return cands 

    def sample_ball_candidates_from_orig(self, n_samples, ball_size, fix_indices=None):
        '''
            return - sample n_samples texts from ball_size-hamming dist ball of orig_text
        '''
        return self.sample_ball_candidates_from_text(self.orig_text, n_samples, ball_size, fix_indices=fix_indices)
    
    def sample_ball_candidates_from_text(self, text, n_samples, ball_size, inner=True, only_indices=False, fix_indices=None):
        '''
            return - sample n_samples texts from ball_size-hamming dist ball of given text
        '''
        candidates = []
        indiced_text = self.numbering_text(text)
        inradius = 1 if inner else ball_size
        nbds = self.nbd_sampler(indiced_text, n_samples, radius=ball_size, inradius=inradius, fix_indices=fix_indices)
        if only_indices:
            return nbds
        else:
            for i in range(n_samples):
                cur_indice = nbds[i]
                cur_text = self.text_by_indices(cur_indice)
                candidates.append(cur_text)
            return candidates
    
    def text_by_indices(self, indices):
        assert len(indices) == len(self.target_indices) or len(indices) == self.len_text, "indices length should be one of target indices length or text length"
        if len(indices) == self.len_text:
            cur_text = self.orig_text
            modified_indices = [ct for ct, ind in enumerate(indices) if ind > 0 and ct in self.target_indices]
            words = [self.word_substitution_cache[ct][int(ind)] for ct, ind in enumerate(indices) if ind > 0 and ct in self.target_indices]
        elif len(indices) == len(self.target_indices):
            cur_text = self.orig_text
            modified_indices = [self.target_indices[ct] for ct, ind in enumerate(indices) if ind > 0]
            words = [self.word_substitution_cache[self.target_indices[ct]][int(ind)] for ct, ind in enumerate(indices) if ind > 0]
        #print(modified_indices, words)
        new_text = cur_text.replace_words_at_indices(modified_indices, words)
        new_text.attack_attrs['modified_indices'] = set(modified_indices)
        return new_text
    
    def numbering_text(self, text):
        '''
            return - numbered torch float Tensor of given text
        '''
        if type(text) == list:
            words = text
        else:
            words = text.words
        numbered_text = []
        #print(text)
        #print(text.words)

        for ct in self.target_indices:
            #print(words[ct], self.word_substitution_cache[ct])
            idx = self.word_substitution_cache[ct].index(words[ct])
            numbered_text.append(idx)
        return torch.FloatTensor(numbered_text)

    def numbering_texts(self, texts):
        '''
            return - numbered torch float Tensor of given texts
        '''
        numbered_texts = []
        for text in texts:
            numbered_text = self.numbering_text(text)
            numbered_texts.append(numbered_text.view(1,-1))
        return torch.cat(numbered_texts, dim=0) 

    def radius_sampler(self, n_samples, radius, inradius=1, fix_indices=None):
        '''
            return - concatenated floattensor sampled from inradius <= r <= radius - sampled ball
        '''
        return self.nbd_sampler(torch.zeros(len(self.target_indices)), n_samples, radius, inradius, fix_indices=fix_indices)
    
    def nbd_sampler(self, center_indices, n_samples, radius, inradius=1, fix_indices=None):
        '''
            return - concatenated floattensor sampled from inradius <= r <= radius - sampled ball
        '''
        if len(center_indices.shape) == 2: center_indices = center_indices.view(-1)
        assert len(center_indices) == len(self.target_indices) or len(center_indices) == self.len_text, "indices length should be one of target indices length or text length"
        if len(center_indices) == self.len_text:
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
        nonzero_indices = [ct for ct, ind in enumerate(center_indices) if ind > 0]
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
                new_indices2 = self.numbering_text(self.orig_text).view(1,-1)

                new_indices = torch.cat([new_indices,new_indices2], dim=0)                
                return new_indices
            else:
                I = random.sample(nonzero_indices, min(radius,nL))
                new_indices = copy.deepcopy(center_indices)
                new_indices[I] = 0
                cands.append(new_indices.view(1,-1))
        return torch.cat(cands,dim=0)
    
    def neighbors(self, x, radius, inradius=1, indices=None):
        """

        :param x: 1D Tensor
        :param partition_samples:
        :param edge_mat_samples:
        :param n_vertices:
        :param uniquely:
        :return:
        """
        if len(x.shape) == 2:
            x = x.view(-1)
        assert len(x) == self.len_text or len(x) == len(self.target_indices), "something wrong"
        if len(x) == self.len_text:
            x = torch.FloatTensor([x[ind] for ind in self.target_indices])
        nbds = x.new_empty((0, x.numel()))
        nbd = self._cartesian_neighbors(x, radius, inradius=inradius, indices=indices)
        nbds = torch.cat([nbds, nbd])
        return nbds

    def _cartesian_neighbors(self, x, radius, inradius=1, indices=None):
        """
        For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by edge_mat_list
        :param grouped_x: 1D Tensor
        :param edge_mat_list:
        :return: 2d tensor in which each row is 1-hamming distance far from x
        """
        neighbor_list = []
        if type(indices) is type(None):
            traversal_indices = list(range(len(self.target_indices)))
        else:
            traversal_indices = indices
        #for i in range(len(self.target_indices)):
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

    def update_center_text(self, center_text, best_candidates):
        '''
            return center_text if center_text is better than all best_candidates. 
            return best of best_candidates otherwise.
        '''
        center_idx = self.eval_texts_str.index(center_text.text)
        best_idxs = [self.eval_texts_str.index(best_candid.text) for best_candid in best_candidates if best_candid.text in self.eval_texts_str]

        center_y = self.eval_Y[center_idx]
        best_ys = self.eval_Y[best_idxs]
        best_y = torch.max(best_ys)
        
        if best_y > center_y:
            best_idx = best_idxs[torch.argmax(best_ys).item()]
            center_idx = best_idx
            center_text = self.eval_texts[center_idx]
        return center_text, center_idx

