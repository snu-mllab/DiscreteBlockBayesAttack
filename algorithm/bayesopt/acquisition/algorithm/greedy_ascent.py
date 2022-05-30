import torch
import numpy as np
from bayesopt.acquisition.acquisition_function.acquisition_functions import expected_improvement
from bayesopt.dpp.dpp import dpp_sample
import copy 
def find_remained_indices(candidates, history_X, N):
    if len(candidates)==0:
        return []
    values, indices = torch.topk(((candidates.long().t() == history_X.long().unsqueeze(-1)).all(dim=1)).int(),1,1 )
    rm_ids = set([int(ind) for ind in indices[values!=0]])
    remained_indices = list(set(list(range(N))) - rm_ids)
    return remained_indices

def find_removed_indices(candidates, history_X):
    values, indices = torch.topk(((candidates.long().t() == history_X.long().unsqueeze(-1)).all(dim=1)).int(),1,1 )
    rm_ids = set([int(ind) for ind in indices[values!=0]])
    return list(rm_ids)

def greedy_ascent_with_indices(center_indices, opt_indices, stage, hb, surrogate_model, kernel_name, batch_size, reference=None, filter=True, return_ei=False, acq_with_opt_indices=True):
    
    candidates = hb.neighbors(center_indices, stage+1, 0, indices=opt_indices)
    N, L = candidates.shape

    # filtering observed candidates
    if filter:
        remained_indices = find_remained_indices(candidates, hb.eval_X_num, N)
    else:
        remained_indices = list(set(list(range(N))))

    testX_cate = candidates[remained_indices]

    # calculate acquisition
    if acq_with_opt_indices:
        testX = testX_cate[:,opt_indices]
        centerX = center_indices.view(1,-1)[:,opt_indices]
    else:
        testX = testX_cate
        centerX = center_indices.view(1,-1)

    if reference == None:
        _, reference, best_ind = hb.best_of_hamming(hb.orig_X, stage+1)

    if len(remained_indices)==0:
        "something wrong, do larger space in greedy ascent with indices"
        if return_ei:
            center_ei = surrogate_model.acquisition(centerX, bias=reference)
            return center_indices.view(1,-1), torch.Tensor([center_ei])
        else:
            return center_indices.view(1,-1)

    ei = surrogate_model.acquisition(testX, bias=reference)
    #center_ei = surrogate_model.acquisition(centerX, bias=reference)

    topk_values, topk_indices = torch.topk(ei, min(len(ei),batch_size))
    best_candidates_indices = torch.cat([testX_cate[idx].view(1,-1) for idx in topk_indices],dim=0)
    if return_ei:
        return best_candidates_indices, topk_values
    else:
        return best_candidates_indices

def acquisition_maximization_with_indices(cur_seqs, opt_indices, batch_size, stage, hb, surrogate_model, kernel_name, reference=None, patience=5, dpp_type='no', acq_with_opt_indices=True):
    global_candidates_, global_eis_ = [], []

    for cur_seq in cur_seqs:
        cur_indices = hb.numbering_seq(cur_seq).view(1,-1)
        if acq_with_opt_indices:
            cur_ei = surrogate_model.acquisition(cur_indices[:,opt_indices], bias=reference)
        else:
            cur_ei = surrogate_model.acquisition(cur_indices, bias=reference)
        global_candidates_.append(cur_indices)
        global_eis_.append(cur_ei)

        num_next = int(np.ceil(100 / len(cur_seqs)))
        filtering = True
        new_candidates_ = []
        new_eis_ = []
        new_candidates, new_eis = greedy_ascent_with_indices(cur_indices, opt_indices, stage, hb, surrogate_model, kernel_name=kernel_name, batch_size=num_next, reference=reference, filter=filtering, return_ei=True, acq_with_opt_indices=acq_with_opt_indices)
        new_candidates_.append(new_candidates)
        new_eis_.extend(new_eis)
        N = len(new_candidates_)
        new_candidates_ = torch.cat(new_candidates_, dim=0)
        
        candidates, indices = unique(new_candidates_, dim=0)
        eis = [new_eis_[ind] for ind in indices]
        assert len(candidates) == len(eis), f'something wrong {len(candidates)}, {len(eis)}'

        global_candidates_.append(candidates)
        global_eis_.extend(eis)

    global_candidates, indices = unique(torch.cat(global_candidates_, dim=0), dim=0)
    global_eis = [global_eis_[ind] for ind in indices]
    N, L = global_candidates.shape
    remained_indices = find_remained_indices(global_candidates, hb.eval_X_num, N)

    global_candidates = global_candidates[remained_indices]
    global_eis = [global_eis[ind] for ind in remained_indices]
    assert len(global_candidates) == len(global_eis), f'something wrong {len(global_candidates)}, {len(global_eis)}'
    
    global_eis = torch.Tensor(global_eis)

    if len(global_candidates) == 0:
        return None

    if dpp_type == 'no' or dpp_type == 'no_one':
        topk_values, topk_indices = torch.topk(global_eis, min(len(global_eis),batch_size))
        candidates = [hb.seq_by_indices(global_candidates[ind]) for ind in topk_indices]
    elif dpp_type == 'dpp_posterior':
        topk_values, topk_indices = torch.topk(global_eis, min(len(global_eis),100))
        global_candidates = global_candidates[topk_indices]

        num = min(len(global_candidates), batch_size)
        if acq_with_opt_indices:
            Lmat = surrogate_model.get_covar(global_candidates[:,opt_indices].cuda()).cpu().detach().numpy()
        else:
            Lmat = surrogate_model.get_covar(global_candidates.cuda()).cpu().detach().numpy()
        Lmat = Lmat / np.mean(np.abs(Lmat))
        if Lmat.shape[0] == num:
            best_indices = list(range(num))
        else:
            best_indices = dpp_sample(Lmat, k=num, T=0)
        candidates = [hb.seq_by_indices(global_candidates[ind]) for ind in best_indices]
    
    if len(candidates):
        return candidates
    else:
        "something wrong, do larger space"
        return None

def unique(x, dim=None):
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)