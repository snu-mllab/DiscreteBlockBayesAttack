import numpy as np
from dppy.finite_dpps import FiniteDPP

def dpp_init(L,k):
    n, m = L.shape
    assert n == m, "L should be square numpy matrix"
    assert n >= k, "candidate pool should be greater or equal than k"

    S = [0]
    cur_det = L[S][:,S]
    while len(S) < k:
        det_best = -1e9
        S_best = None
        for i in range(n):
            if i in S:
                continue
            S_tmp = S + [i]
            submat = L[S_tmp][:,S_tmp]
            det = np.linalg.det(submat)
            if det > det_best:
                S_best = S_tmp
                det_best = det 
        S = S_best
        cur_det = det_best
    return S, cur_det

def dpp_sample(L, k, T):
    n, m = L.shape
    assert n == m, "L should be square numpy matrix"
    assert n >= k, "candidate pool should be greater or equal than k"

    # greedy insertion
    S, cur_det = dpp_init(L, k)
    if T == 0:
        return S
    try:
        DPP = FiniteDPP('likelihood', **{'L': L})
        S = DPP.sample_mcmc_k_dpp(size=k, s_init=S, nb_iter=T)
        return S
    except:
        L_ = L + 1e-8 * np.eye(n)
        DPP = FiniteDPP('likelihood', **{'L': L_})
        S = DPP.sample_mcmc_k_dpp(size=k, s_init=S, nb_iter=T)
        return S