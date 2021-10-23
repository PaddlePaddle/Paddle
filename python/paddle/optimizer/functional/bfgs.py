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
from .linesearch import hz_linesearch as linesearch

def as_float_or_double_tensor(input):
    r"""Generates a float or double typed tensor from `input`. The data
    type of `input` is either float or double.

    Args:
        input(Scalar | List | Tensor): a scalar or a list of floats, or 
        a tensor with dtype float or double.

    Returns:
        A tensor with float or double dtype. 
    """
    assert isinstance(input, (float, list, paddle.Tensor)), (
        f'Input `{input}` should be float, list or paddle.Tensor but found 
        f'{type(input)}.'
    )
    try:
        output = paddle.to_tensor(input)
        if output.dtype not in paddle.float, paddle.float64:
            raise TypeError
    except TypeError:
        raise TypeError(f'The data type of {input} is {type(input)}, which is not supported.')
    
    return output

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

def pnorm(x, p=2):
    return paddle.norm(x, p=p, axis=-1)

def infnorm(x):
    return paddle.norm(x, p='inf', axis=-1)

def matnorm(x):
    return paddle.norm(x, 'fro')

class BFGS_State(object):
    r"""
    BFFS_State is used to represent intermediate and final result of
    the BFGS minimization.

    Pulic instance members:    
        k: the iteration number.
        converged (Tensor): a boolean typed tensor of shape [...].
            For each element, True indicates the minimum is found
            up to the gradient tolerance. False if otherwise.
        ls (Namedtuple): a summary of the line search result
            for the next iterate.
        nf (Tensor): scalar valued tensor holding the number of
            function calls made.
        ng (Tensor): scalar valued tensor holding the number of
            gradient calls made.
        xk (Tensor): the iterate point.
        fk (Tensor): the ``func``'s output.
        gk (Tensor): the ``func``'s gradients. 
        Hk (Tensor): the approximated inverse hessian of ``func``.
    """
    def __init__(self, xk, fk, gk, Hk, k=0, nf=1, ng=1):
        self.xk = xk
        self.fk = fk
        self.gk = gk
        self.Hk = Hk
        self.k = k
        self.nf = nf
        self.ng = ng
        self.ls_status = paddle.expand(paddle.to_tensor([True]),
                                       shape=fk.shape)
        converged = paddle.to_tensor([False])
        self.converged = converged.broadcast_to(fk.shape)


def iterates(func, x0, H0=None, gtol=1e-8, iters=50, ls_iters=50):
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
        gtol (Scalar): terminates if the gradient norm is smaller than
            this `gtol`. Currently gradient norm uses inf norm.
            The default value is 1e-8.
        iters (Scalar): the maximum number minimization iterations.
            The default value is 50.
        ls_iters (Scalar): the maximum number of line search iterations.
            The default value is 50.
    
    Returns:
        A generator which returns a BFGS_State per iteration.
    """
    # Evaluates the starting points
    x0 = as_float_or_double_tensor(x0)
    f0, g0 = vjp(func, x0)
    
    input_shape = x0.shape
    hessian_shape = input_shape + input_shape[-1:]
    
    # The initial approximation of the inverse Hessian.
    if H0 is None:
        H0 = paddle.expand(paddle.eye(input_shape[-1]), hessian_shape)
    else:
        verify_symmetric_positive_definite_matrix(H0)        

    # Puts the starting points in the initial state and kicks off minimization 
    # process.
    state = BFGS_State(x0, f0, g0, H0)
    gnorm = infnorm(g0)

    while paddle.any(gnorm > gtol) and k < iters:
        # Performs line search
        ls, alpha_k = linesearch(state, func, gtol=gtol, max_iters=ls_iters)
        

        