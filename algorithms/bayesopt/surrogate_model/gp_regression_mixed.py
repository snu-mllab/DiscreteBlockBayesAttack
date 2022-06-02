#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable
from typing import Dict, List, Optional, Any

import torch
from botorch.models.gp_regression import SingleTaskGP
from algorithms.bayesopt.kernel_function.new_categorical_kernel import CategoricalKernel2 # Memory efficient re-implementation of the categorical kernel
from botorch.utils.containers import TrainingData
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors import GammaPrior, UniformPrior, HorseshoePrior
from torch import Tensor
from gpytorch.distributions.multivariate_normal import MultivariateNormal


class NewMixedSingleTaskGP(SingleTaskGP):
    r"""A single-task exact GP model for categorical search spaces.
    """
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: List[int],
        likelihood: Optional[Likelihood] = None,
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            cat_dims: A list of indices corresponding to the columns of
                the input `X` that should be considered categorical features.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
        """
        if len(cat_dims) == 0:
            raise ValueError(
                "Must specify categorical dimensions for MixedSingleTaskGP"
            )
        self._ignore_X_dims_scaling_check = cat_dims
        input_batch_shape, aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

        if likelihood is None:
            # This Gamma prior is quite close to the Horseshoe prior
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)

        lengthscale_prior = GammaPrior(3.0,6.0)
        outputscale_prior = UniformPrior(0,1,validate_args=False)
        outputscale_constraint = Interval(0,1, initial_value=0.1)
        outputscale_prior.low = outputscale_prior.low.cuda()
        outputscale_prior.high = outputscale_prior.high.cuda()
        covar_module = ScaleKernel(
            CategoricalKernel2(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                lengthscale_constraint=GreaterThan(1e-06),
                lengthscale_prior=lengthscale_prior
            ),
            batch_shape=aug_batch_shape,
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=outputscale_prior,
        ) 
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
        )
    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @classmethod
    def construct_inputs(
        cls, training_data: TrainingData, **kwargs: Any
    ) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: None expected for this class.
        """
        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "cat_dims": kwargs["categorical_features"],
            "likelihood": kwargs.get("likelihood"),
        }
