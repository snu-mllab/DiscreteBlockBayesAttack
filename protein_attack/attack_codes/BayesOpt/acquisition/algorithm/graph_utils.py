import random as rnd
import numpy as np
import torch

def filter_by_radius(xs, radius):
    filtered_candidates = torch.empty(0)
    for i in range(len(xs)):
        if xs[i].nonzero().numel() <= radius:
            filtered_candidates = torch.cat([filtered_candidates, xs[i:i+1]], dim=0)

    return filtered_candidates.long()

def filter_by_radius_vectorized(xs, radius, inradius=1):
    xs_nonzero_mask = (torch.abs(xs)!=0) * 1.0 
    num_nonzero_vec = torch.sum(xs_nonzero_mask,dim=1)
    _, length = xs.shape
    if inradius==0 or inradius>=length:
        bool_vec = num_nonzero_vec <= radius # False means that index would be filtered.
    else:
        bool_vec = (num_nonzero_vec <= radius) * (num_nonzero_vec >= inradius) 
    filtered_indices = bool_vec.nonzero().view(-1)
    filtered_candidates = xs[filtered_indices]

    return filtered_candidates.long()

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def radius_sampler(n_vertices, radius, n_samples, inradius=1):
    nmin1 = n_vertices - 1
    length = len(n_vertices)
    if inradius < length:
        mask_base = [0] * (length-inradius) + [1] * radius
    else:
        mask_base = [1] * length
    mask = np.stack([rnd.sample(mask_base,length) for _ in range(n_samples)])
    x = np.random.randint(nmin1, size=(n_samples,length)) + 1
    y = torch.Tensor(x*mask).long()
    
    return y
            
def neighbors(x, partition_samples, edge_mat_samples, n_vertices, radius, uniquely=False, inradius=1):
    """

    :param x: 1D Tensor
    :param partition_samples:
    :param edge_mat_samples:
    :param n_vertices:
    :param uniquely:
    :return:
    """
    nbds = x.new_empty((0, x.numel()))
    # nlp에서는 edge mat sample이 모두 동일해서 하나에 대해서만 함.
    for i in range(len(partition_samples[:1])):
        nbd = _cartesian_neighbors(x, edge_mat_samples[i], radius, inradius=inradius)
        added_ind = []
        if uniquely:
            for j in range(nbd.size(0)):
                if not torch.any(torch.all(nbds == nbd[j], dim=1)):
                    added_ind.append(j)
            if len(added_ind) > 0:
                nbds = torch.cat([nbds, nbd[added_ind]])
        else:
            nbds = torch.cat([nbds, nbd])
    return nbds

def _cartesian_neighbors(grouped_x, edge_mat_list, radius, inradius=1):
    """
    For given vertices, it returns all neighboring vertices on cartesian product of the graphs given by edge_mat_list
    :param grouped_x: 1D Tensor
    :param edge_mat_list:
    :return: 2d tensor in which each row is 1-hamming distance far from x
    """
    neighbor_list = []
    for i in range(len(edge_mat_list)):
        nbd_i_elm = edge_mat_list[i][grouped_x[i]].nonzero(as_tuple=False).squeeze(1)
        nbd_i = grouped_x.repeat((nbd_i_elm.numel(), 1))
        nbd_i[:, i] = nbd_i_elm
        neighbor_list.append(nbd_i)
    neighbor_list = torch.cat( neighbor_list, dim=0)
    neighbor_list = filter_by_radius_vectorized(neighbor_list, radius, inradius=inradius)
    return neighbor_list

if __name__ == "__main__":
    n_vertices = [2,5,3,5,6,7,3]
    radius = 3
    n_samples = 5
    x = radius_sampler(n_vertices, radius, n_samples)
    print(x)