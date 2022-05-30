
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from typing import Dict, Optional, Union
from torch import Tensor
from botorch.acquisition.objective import ScalarizedObjective
from torch.distributions import Normal
from botorch.utils.transforms import t_batch_mode_transform
import torch
class ExpectedImprovementReg(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective: Optional[ScalarizedObjective] = None,
        reg_coef: float = 1e-5,
        maximize: bool = True,
    ) -> None:
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)
        self.reg_coef = reg_coef

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        self.best_f = self.best_f.to(X)
        posterior = self._get_posterior(X=X)
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(view_shape)
        mean = mean - self.reg_coef * torch.count_nonzero(X, dim=-1).view(view_shape)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei