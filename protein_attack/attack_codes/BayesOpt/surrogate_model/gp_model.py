from attack_codes.BayesOpt.fitting.fitting import fit_model_partial
from attack_codes.BayesOpt.surrogate_model.model_wrapper import BaseModel
import torch
import numpy as np
import gc
from attack_codes.BayesOpt.acquisition.acquisition_function.acquisition_functions_reg import ExpectedImprovementReg
from attack_codes.BayesOpt.acquisition.acquisition_function.acquisition_functions import expected_improvement
import gpytorch

class MyGPModel(BaseModel):
    def __init__(self, kernel_name : str):
        self.model = None 
        self.kernel_name = kernel_name
        self.partition_size = 512
 
    def fit_partial(self, hb, opt_indices, init_ind, prev_indices):
        del self.model
        self.model = fit_model_partial(hb, self.kernel_name, opt_indices, init_ind, prev_indices)
    
    def predict(self, eval_X):
        self.model.eval()
        self.model.likelihood.eval()
        
        with gpytorch.settings.fast_pred_var():
            pred = self.model(eval_X)
            pred = self.model.likelihood(pred)
            mean, variance = pred.mean.detach(), pred.variance.clamp_min(1e-9).detach()
        return mean, variance
    
    def acquisition(self, eval_X, bias=None, reg_coef=1e-5):
        with torch.no_grad():
            eval_X_cuda = eval_X.to(dtype=torch.double).cuda()
            mean, var = self.predict(eval_X_cuda)
            mean = mean - reg_coef * torch.count_nonzero(eval_X_cuda, dim=-1).view(-1)
            ei = expected_improvement(mean, var, bias).cpu().detach()
        del mean, var, eval_X_cuda
        return ei

    def get_covar(self, eval_X):
        with torch.no_grad():
            covar = self.model.posterior(eval_X).mvn.covariance_matrix
        return covar 