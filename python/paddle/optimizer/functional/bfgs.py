# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import paddle
from ...autograd import vjp
from .linesearch import hager_zhang as linesearch

def verify_symmetric_positive_definite_matrix(H):
    r"""
    Verifies the input matrix is symmetric positive-definite matrix.

    Args:
        H (Tensor): a square matrix, of type float32 or float64.

    Returns:
         
    """
    is_positive_definite = True
    try:
        L = paddle.cholesky(H)
        del L
    except RuntimeError:
        is_positive_definite = False
    
    assert is_positive_definite, (
        f"(Batched) matrix {H} is not positive definite."
    )

    perm = list(range(len(H.shape)))
    perm[-2:] = perm[-1], perm[-2]
    batch_deltas = paddle.norm(H - H.transpose(perm), axis=perm[-2:])
    is_symmetic = paddle.sum(batch_deltas) == 0.0
    
    assert is_symmetic, (
        f"(Batched) matrix {H} is not symmetric."
    )

class BFGS_State(object):
    r"""
    BFFS_State is used to represent intermediate and final result of
    the BFGS minimization.

    Pulic instance members:    
        k: the iteration number.
        converged (Tensor): a boolean typed tensor of shape [...].
            For each element, True indicates the minimum is found
            up to the gradient tolerance. False if otherwise.
        linesearch_status (Tensor): a boolean typed tensor of shape
            [...]. For each element, True indicates line search
            succeeds to find a step size. False if fails to find
            a step size when the maximum number of trials is reached.
        xk (Tensor): the iterate point.
        fk (Tensor): the ``func``'s output.
        gk (Tensor): the ``func``'s gradients. 
        Hk (Tensor): the approximated inverse hessian of ``func``.
    """
    def __init__(self, k, xk, fk, gk, Hk, converged=None):
        self.k = k
        self.xk = xk
        self.fk = fk
        self.gk = gk
        self.Hk = Hk   
        self.ls_status = paddle.expand(paddle.to_tensor([True]),
                                       shape=fk.shape)
        if converged is not None:
            self.converged = converged
        else:
            converged = paddle.to_tensor([False])
            self.converged = converged.broadcast_to(fk.shape)


def iterates(func, x0, H0=None, 
             grad_tolerance=1e-8,
             max_iters=50,
             max_ls_iters=50):
    r"""
    Returns the iterates on the minimization path of a differentiable
    function by applying the BFGS quasi-Newton method. 

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization,
        Second Edition, 2006.

    Args:
        func: the objective function to minimize. ``func`` accepts
            a multivariate input and returns a scalar, both allowed
            batching into tensors in shape [..., n] and [...].
        x0 (Tensor): the starting point of the iterates. The batching
            form has the shape [..., n].
        H0 (Tensor): the initial inverse hessian approximation. The
            batching form has the shape [..., n, n], where the
            batching dimensions have the same shape with ``x0``.
            The default value is None.
        grad_tolerance (Tensor): the termination condition. If the
            inf-norm of the gradients is smaller than ``grad_tolerance``
            then end the minimization path. The default value is 1e-8.
        max_iters (Scalar): the maximum number minimization iterations.
            The default value is 50.
        max_ls_iters (Scalar): the maximum number of line search
            iterations. The default value is 50.
    
    Returns:
        A generator which returns a BFGS_State per iteration.
        BFGS_State (Dict):
            k: the iteration number.
            converged (Tensor): a boolean typed tensor of shape [...].
                For each element, True indicates the minimum is found
                up to the gradient tolerance. False if otherwise.
            linesearch_status (Tensor): a boolean typed tensor of shape
                [...]. For each element, True indicates line search
                succeeds to find a step size. False if fails to find
                a step size when the maximum number of trials is reached.
            xk (Tensor): the iterate point.
            fk (Tensor): the ``func``'s output.
            gk (Tensor): the ``func``'s gradients. 
            Hk (Tensor): the approximated inverse hessian of ``func``.
    """
    def cond(iterate, max_iters=max_iters):
        return (iterate.k < max_iters
                & paddle.logical_not(iterate.converged)
                & iterate.ls_status)

    assert isinstance(x0, paddle.Tensor), (
        f"{x0} is expected to be paddle.Tensor but found {type(x0).} "
    )

    # Starting point
    f0, g0 = vjp(func, x0)
    
    input_shape = x0.shape
    hessian_shape = input_shape + input_shape[-1:]
    
    if H0 is None:
        H0 = paddle.expand(paddle.eye(input_shape[-1]), hessian_shape)
    else:
        verify_symmetric_positive_definite_matrix(H0)        

    # The minimizing loop starts.
    iterate = BFGS_State(0, x0, f0, g0, H0)

    while cond(iterate):

        k, xk, fk = iterate.k, iterate.xk, iterate.fk
        gk, Hk = iterate.gk, iterate.Hk

        # Calculate the line search direction pk.
        pk = - paddle.dot(Hk, gk)
        
        # Perform line search and get alpha_k
        def f(ak):
            r'''
            The induced single variable function of `func` on a particular direction pk.
            Args:
                ak (Tensor): a scalar tensor, or a tensor of shape [...] in batching mode,
                giving the step sizes alpha_k.
            '''
            return func(xk + ak*pk)

        iterate = linesearch(iterate, f, max_iters=max_ls_iters)
        

        