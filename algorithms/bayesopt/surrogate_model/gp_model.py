from algorithms.bayesopt.fitting.fitting import fit_model_partial
from algorithms.bayesopt.surrogate_model.model_wrapper import BaseModel
import torch
import numpy as np
import gc
from algorithms.bayesopt.acquisition.acquisition_function.acquisition_functions import expected_improvement
import gpytorch
import copy
class MyGPModel(BaseModel):
    def __init__(self, niter : int = 20):
        self.model = None 
        self.partition_size = 512
        self.niter = niter

    def fit_partial(self, hb, opt_indices, init_ind, prev_indices):
        if type(self.model) == type(None):
            params = None
        else:
            params = copy.deepcopy(self.model.state_dict())
        del self.model
        self.model = fit_model_partial(hb, opt_indices, init_ind, prev_indices, params=params, niter=self.niter)
    
    def predict(self, eval_X):
        self.model.eval()
        self.model.likelihood.eval()
        
        with gpytorch.settings.fast_pred_var():
            pred = self.model(eval_X)
            pred = self.model.likelihood(pred)
            mean, variance = pred.mean.detach(), pred.variance.clamp_min(1e-9).detach()
        return mean, variance
    
    def acquisition(self, eval_X, bias=None):
        with torch.no_grad():
            eval_X_cuda = eval_X.to(dtype=torch.double).cuda()
            mean, var = self.predict(eval_X_cuda)
            ei = expected_improvement(mean, var, bias).cpu().detach()
        del mean, var, eval_X_cuda
        return ei

    def get_covar(self, eval_X):
        with torch.no_grad():
            covar = self.model.posterior(eval_X).mvn.covariance_matrix
        return covar 