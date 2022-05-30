from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from attack_codes.BayesOpt.surrogate_model.gp_regression_mixed import NewMixedSingleTaskGP
import torch
from attack_codes.BayesOpt.historyboard.hb import HistoryBoard
from gpytorch.constraints import GreaterThan

import gpytorch

MAX_ITER = 20
print_freq = 20

from typing import List

def fit_model_partial(
        hb : HistoryBoard = None,
        kernel_name : str = None,
        opt_indices : list = None,
        init_ind : int = None,
        prev_indices : int = None,
        params : dict = {},
    ):
    surrogate_model, train_X, train_Y = get_data_and_model_partial(hb, kernel_name, opt_indices, init_ind, prev_indices)
    if params:
        surrogate_model.load_state_dict(params)

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
    {'params': surrogate_model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    n_iter = MAX_ITER
    mll = ExactMarginalLogLikelihood(surrogate_model.likelihood, surrogate_model).cuda()

    while True:
        output = surrogate_model(train_X)
        loss = -mll(output, surrogate_model.train_targets)
        loss.backward()
        optimizer.zero_grad()
        del output, loss
        break

    for i in range(n_iter):
        optimizer.zero_grad()
        output = surrogate_model(train_X)

        loss = -mll(output, surrogate_model.train_targets)
        
        if (i+1) % print_freq == 0 or i == 0:
            pass
            #print_iter(i, loss, surrogate_model, kernel_name, n_iter)
        loss.backward()
        optimizer.step()
    return surrogate_model

def get_data_and_model_partial(hb, kernel_name, opt_indices, init_ind, prev_indices):
    train_Y = torch.cat([hb.eval_Y[prev_indices].view(len(prev_indices),1), hb.eval_Y[init_ind:]], dim=0)
    train_Y = train_Y.to(dtype=torch.double)
    if kernel_name in ['categorical' , 'categorical_horseshoe']:
        train_X_center = hb.eval_X_num[prev_indices][:,opt_indices].view(len(prev_indices),-1).cuda()
        train_X = hb.eval_X_num[init_ind:, opt_indices].cuda()
        train_X = torch.cat([train_X_center, train_X], dim=0)
        train_X = train_X.to(dtype=torch.double)
        _, L = train_X.shape
        #print("get data model parital", train_X.shape)
        if kernel_name == 'categorical':
            surrogate_model = NewMixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=list(range(L))).cuda()
        elif kernel_name == 'categorical_horseshoe':
            surrogate_model = NewMixedSingleTaskGP(train_X=train_X, train_Y=train_Y, cat_dims=list(range(L)), use_horseshoe=True).cuda()

    surrogate_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
    surrogate_model.mean_module.initialize(constant=-1.0)
    surrogate_model.train()
    return surrogate_model, train_X, train_Y

def print_iter(i, loss, surrogate_model, kernel_name, n_iter):
    print_str = 'Iter %d/%d - Loss: %.3f   kernel_name : %s\n' % (i+1, n_iter, loss.item(), kernel_name)   
    if kernel_name == 'categorical':
        print_str += '   lengthscale[:,:5]: {}\n'.format(surrogate_model.covar_module.base_kernel.lengthscale[:,:5])
    elif kernel_name == 'categorical_horeshoe':
        print_str += '   lengthscale[:,:5]: {}\n'.format(1./surrogate_model.covar_module.base_kernel.lengthscale[:,:5])

    print_str += '   noise: %.3f' % (surrogate_model.likelihood.noise.item())
    print_str += '   mean: %.3f' % (surrogate_model.mean_module.constant.item())
    print(print_str)

