import torch
from torch.distributions.normal import Normal

def expected_improvement(mean, var, reference):
	predictive_normal = Normal(mean.new_zeros(mean.size()), mean.new_ones(mean.size()))
	std = torch.sqrt(var)
	standardized = (mean - reference) / std

	ucdf = predictive_normal.cdf(standardized)
	updf = torch.exp(predictive_normal.log_prob(standardized))
	ei = std * (updf + standardized * ucdf)
	return ei