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
from .bfgs_utils import as_float_tensor, make_state, update_state
from .bfgs_utils import any_active, active_state, converged_state, failed_state
from .bfgs_utils import vnorm_inf
from .linesearch import hz_linesearch as linesearch


def verify_symmetric_positive_definite_matrix(H):
    r"""
    Verifies the input matrix is symmetric positive-definite matrix.

    Args:
        H (Tensor): a square matrix of dtype float32 or float64.
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


class SearchState(object):
    r"""
    BFFS_State is used to represent intermediate and final result of
    the BFGS minimization.

    Pulic instance members:    
        k: the iteration number.
        state (Tensor): a int8 tensor of shape [...], holding the set of
            searching states for the batch inputs. For each element,
            0 indicates active, 1 converged and 2 failed.
        nf (Tensor): scalar valued tensor holding the number of
            function calls made.
        ng (Tensor): scalar valued tensor holding the number of
            gradient calls made.
        xk (Tensor): the iterate point.
        fk (Tensor): the ``func``'s output.
        gk (Tensor): the ``func``'s gradients. 
        Hk (Tensor): the approximated inverse hessian of ``func``.
    """
    def __init__(self, xk, fk, gk, Hk, ak=None, k=0, nf=1, ng=1):
        self.xk = xk
        self.fk = fk
        self.gk = gk
        self.Hk = Hk
        self.k = k
        self.ak = ak
        self.Ck = fk
        self.nf = nf
        self.ng = ng
        self.state = make_state(fk)
        self._StateConverged = make_state(fk, 'converged')
        self._StateFailed = make_state(fk, 'failed')

def iterates(func,
             x0, 
             dtype='float',
             H0=None,
             gtol=1e-8,
             xtol=0,
             iters=50,
             ls_iters=50):
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
        dtype ('float' | 'float32' | 'float64' | 'double'): the data
            type to be used.
        H0 (Tensor): the initial inverse hessian approximation. The
            batching form has the shape [..., n, n], where the
            batching dimensions have the same shape with ``x0``.
            The default value is None.
        gtol (Scalar): terminates if the gradient norm is smaller than
            this `gtol`. Currently gradient norm uses inf norm.
            The default value is 1e-8.
        xtol (Scalar): terminates if the distance of succesive iterates
            is smaller than this value. The default value is 0.
        iters (Scalar): the maximum number minimization iterations.
            The default value is 50.
        ls_iters (Scalar): the maximum number of line search iterations.
            The default value is 50.
    
    Returns:
        A generator which returns a SearchState per iteration.
    """
    # Proprocesses inputs and control parameters 
    x0 = as_float_tensor(x0, dtype)
    dtype = x0.dtype

    gtol = as_float_tensor(gtol, dtype)
    xtol = as_float_tensor(xtol, dtype)

    # Evaluates the starting points
    f0, g0 = vjp(func, x0)
    
    # If function is applied to batched input, the last axis of the input
    # tensor holds the input dimensions.
    input_dim = x0.shape[-1]
    hessian_shape = x0.shape + [input_dim]
    
    # The initial approximation of the inverse Hessian.
    if H0 is None:
        I = paddle.eye(input_dim, dtype=dtype)
        H0 = paddle.broadcast_to(I, hessian_shape)
    else:
        H0 = as_float_tensor(H0, dtype)        
        verify_symmetric_positive_definite_matrix(H0)

    # Puts the starting points in the initial state and kicks off the
    # minimization process.
    gnorm = vnorm_inf(g0)
    state = SearchState(x0, f0, g0, H0)

    while any_active(state.state) and state.k < iters:
        # Performs line search and updates the state 
        linesearch(state,
                   func, 
                   gtol=gtol,
                   xtol=xtol,
                   max_iters=ls_iters)
        
        # Calculates the gradient norms
        gnorm = vnorm_inf(state.gk)

        # Updates the state tensor on the newly converged elements.
        state.state = update_state(state.state, gnorm < gtol, 'converged')
        

        