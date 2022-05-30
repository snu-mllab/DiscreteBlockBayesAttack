#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor
import numpy as np

class NewCategoricalKernel(Kernel):
    r"""A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: list,
        x2: list,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs
    ) -> Tensor:
        x1_ = [x.words for x in x1]
        x2_ = [x.words for x in x2]
        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        dists = delta * self.lengthscale.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res

class CategoricalKernel2(Kernel):
    r"""A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        key: int = 0,
        **kwargs
    ) -> Tensor:
        if key == 0:
            dists = BinaryGradientFunction2.apply(self.lengthscale.unsqueeze(-2), x1, x2)
        elif key == 1:
            delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
            dists = BinaryGradientFunction.apply(self.lengthscale.unsqueeze(-2), delta)
        elif key == 2:
            delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
            dists = delta / self.lengthscale.unsqueeze(-2)
            if last_dim_is_batch:
                dists = dists.transpose(-3, -1)
            else:
                dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res

class NewCategoricalKernel2(Kernel):
    r"""A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) * lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        key: int = 0,
        **kwargs
    ) -> Tensor:
        if key == 0:
            dists = NewBinaryGradientFunction2.apply(self.lengthscale.unsqueeze(-2), x1, x2)
        elif key == 2:
            delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
            dists = delta * self.lengthscale.unsqueeze(-2)
            if last_dim_is_batch:
                dists = dists.transpose(-3, -1)
            else:
                dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

class BinaryGradientFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lengthscale, binary):
        nbatch = 64
        detached_ls = detach_variable(lengthscale)
        N = binary.shape[0]
        dists = []
        for i in range(int(np.ceil(N / nbatch))):
            tmp = binary[i*nbatch:(i+1)*nbatch] / detached_ls
            dists.append(tmp.mean(-1))
        dists_final = torch.cat(dists,0)
        ctx.save_for_backward(binary, detached_ls)
        return dists_final

    @staticmethod
    def backward(ctx, grad_output):
        nbatch = 64
        binary, detached_ls = ctx.saved_tensors
        N, L = binary.shape[0], binary.shape[-1]
        m = len(binary.shape)
        if m == 3:
            sumdim = [0,1]
        elif m ==  4:
            sumdim = [0,1,2]

        grad = 0
        for i in range(int(np.ceil(N / nbatch))):
            tmp = -grad_output[i*nbatch:(i+1)*nbatch].unsqueeze(-1) * binary[i*nbatch:(i+1)*nbatch]
            grad += torch.sum(tmp / torch.square(detached_ls), dim=sumdim) / L
        return grad.view(1,1,-1), None

class BinaryGradientFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lengthscale, x1, x2):
        nbatch = 256
        detached_ls = detach_variable(lengthscale)
        m = len(x1.shape)
        if m == 2:
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]
        dists = []
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp_pt = binary_pt_pt / detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt
                    del tmp_pt
                dists.append(torch.cat(dists_pt,dim=1))
            dists_final = torch.cat(dists,dim=0)

        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch,j*nbatch:(j+1)*nbatch].unsqueeze(-3)

                    tmp_pt = binary_pt_pt / detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt, tmp_pt
                dists.append(torch.cat(dists_pt,dim=2))
            dists_final = torch.cat(dists,dim=0)
        del dists, dists_pt
        ctx.save_for_backward(x1, x2, detached_ls)
        return dists_final

    @staticmethod
    def backward(ctx, grad_output):
        nbatch = 256
        x1, x2, detached_ls = ctx.saved_tensors
        m = len(x1.shape)
        if m == 2:
            sumdim = [0,1]
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            sumdim = [0,1,2]
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]

        grad = 0
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = -grad_output[i*nbatch:(i+1)*nbatch, j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp / torch.square(detached_ls), dim=sumdim) / L
                    del binary_pt_pt, tmp
        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch,j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = -grad_output[i*nbatch:(i+1)*nbatch,:,j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp / torch.square(detached_ls), dim=sumdim) / L
                    del binary_pt_pt, tmp
        return grad.view(1,1,-1), None, None


class NewBinaryGradientFunction2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, lengthscale, x1, x2):
        nbatch = 64
        detached_ls = detach_variable(lengthscale)
        m = len(x1.shape)
        if m == 2:
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]
        dists = []
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp_pt = binary_pt_pt * detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt
                    del tmp_pt
                dists.append(torch.cat(dists_pt,dim=1))
            dists_final = torch.cat(dists,dim=0)

        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                dists_pt = []
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch,j*nbatch:(j+1)*nbatch].unsqueeze(-3)

                    tmp_pt = binary_pt_pt * detached_ls
                    dists_pt.append(tmp_pt.mean(-1))
                    del binary_pt_pt, tmp_pt
                dists.append(torch.cat(dists_pt,dim=2))
            dists_final = torch.cat(dists,dim=0)
        ctx.save_for_backward(x1, x2, detached_ls)
        return dists_final

    @staticmethod
    def backward(ctx, grad_output):
        nbatch = 64
        x1, x2, detached_ls = ctx.saved_tensors
        m = len(x1.shape)
        if m == 2:
            sumdim = [0,1]
            N, M, L = x1.shape[0], x2.shape[0], x1.shape[-1]
        elif m ==  3:
            sumdim = [0,1,2]
            N, M, L = x1.shape[0], x2.shape[1], x1.shape[-1]

        grad = 0
        if m == 2:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = grad_output[i*nbatch:(i+1)*nbatch, j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp, dim=sumdim) / L
                    del binary_pt_pt, tmp
        elif m == 3:
            for i in range(int(np.ceil(N / nbatch))):
                for j in range(int(np.ceil(M / nbatch))):
                    binary_pt_pt = x1[i*nbatch:(i+1)*nbatch].unsqueeze(-2) != x2[i*nbatch:(i+1)*nbatch,j*nbatch:(j+1)*nbatch].unsqueeze(-3)
                    tmp = grad_output[i*nbatch:(i+1)*nbatch,:,j*nbatch:(j+1)*nbatch].unsqueeze(-1) * binary_pt_pt
                    grad += torch.sum(tmp, dim=sumdim) / L
                    del binary_pt_pt, tmp
        return grad.view(1,1,-1), None, None


if __name__=='__main__':
    from gpytorch.priors import GammaPrior
    from gpytorch.constraints import GreaterThan
    N, M, L = 10, 5, 32
    lengthscale_prior = GammaPrior(3.0,6.0)#None
    m = NewCategoricalKernel2(ard_num_dims=L, 
                            lengthscale_prior=lengthscale_prior, 
                            active_dims=L,
                            lengthscale_constraint=GreaterThan(1e-06)
                        ).cuda()

    x1 = torch.randint(5,[N,1,L]).cuda()
    x2 = torch.randint(5,[N,M,L]).cuda()

    res0 = m.forward(x1, x2, key=0)
    print(res0)

    l0 = torch.sum(res0)
    l0.backward()
    print(m.raw_lengthscale.grad)
    print(m.raw_lengthscale.grad.shape)
    
    #m.raw_lengthscale.grad.zero_()

    #res1 = m.forward(x1, x2, key=1)
    #print(res1)

    #l1 = torch.sum(res1)
    #l1.backward()
    #print(m.raw_lengthscale.grad)
    #print(m.raw_lengthscale.grad.shape)
    
    m.raw_lengthscale.grad.zero_()

    res2 = m.forward(x1, x2, key=2)
    print(res2)

    l2 = torch.sum(res2)
    l2.backward()
    print(m.raw_lengthscale.grad)
    print(m.raw_lengthscale.grad.shape)
