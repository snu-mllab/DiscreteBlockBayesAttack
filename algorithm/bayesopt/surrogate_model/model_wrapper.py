from abc import abstractmethod
import torch
from bayesopt.historyboard.hb import HistoryBoard

class BaseModel:
    @abstractmethod
    def fit(self, hb:HistoryBoard):
        raise NotImplementedError

    @abstractmethod
    def update(self, hb:HistoryBoard):
        raise NotImplementedError

    @abstractmethod
    def predict(self, eval_X: torch.Tensor, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def acquisition(self, eval_X: torch.Tensor, acq_func='expected_improvement', bias=None, **kwargs):
        raise NotImplementedError
