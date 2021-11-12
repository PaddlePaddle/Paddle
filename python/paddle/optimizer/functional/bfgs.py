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
import collections
import paddle
from paddle import einsum
from .bfgs_utils import vjp, ternary
from .bfgs_utils import make_state, make_const, update_state
from .bfgs_utils import active_state, any_active, any_active_with_predicates
from .bfgs_utils import converged_state, failed_state
from .bfgs_utils import as_float_tensor, vnorm_inf
from .bfgs_utils import StopCounter, StopCounterException
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


class BfgsResult(collections.namedtuple('BfgsResult', [
                                            'location',
                                            'converged',
                                            'failed',
                                            'gradient',
                                            'function_value',
                                            'inverse_hessian',
                                            'function_evals',
                                            'gradient_evals',
                                            ])):
    def __repr__(self):
        kvs = [(f, getattr(self, f)) for f in self._fields]
        width = max(len(f) for f in self._fields)
        return '\n'.join(f'{k.rjust(width)} : {repr(v)}' for k, v in kvs)


class SearchState(object):
    r"""
    BFFS_State is used to represent intermediate and final result of
    the BFGS minimization.

    Pulic instance members:    
        k: the iteration number.
        state (Tensor): an int tensor of shape [...], holding the set of
            searching states for the batch inputs. For each element,
            0 indicates active, 1 converged and 2 failed.
        nf (Tensor): scalar valued tensor holding the number of
            function calls made.
        ng (Tensor): scalar valued tensor holding the number of
            gradient calls made.
        ak (Tensor): step size.
        pk (Tensor): the minimizer direction.
        Qk (Tensor): weight for averaging function values.
        Ck (Tensor): weighted average of function values.
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
        self.nf = nf
        self.ng = ng
        self.pk = None
        self.state = make_state(fk)
        self.Qk = make_const(fk, 0)
        self.Ck = make_const(fk, 0)
        self.params = None

    def result(self):
        kw = {
            'location' : self.xk,
            'converged' : converged_state(self.state),
            'failed' : failed_state(self.state),
            'gradient' : self.gk,
            'function_value' : self.fk,
            'inverse_hessian' : self.Hk,
            'function_evals' : self.nf,
            'gradient_evals' : self.ng,
        }
        return BfgsResult(**kw)


def update_approx_inverse_hessian(state, Hk, sk, yk, enforce_curvature=False):
    r"""Updates the approximate inverse Hessian.
    
    Given the input displacement s_k and the change of gradients y_k,
    the inverse Hessian at the next iterate is approximated using the following
    formula:
    
        H_k+1 = (I - rho_k * s_k * T(y_k)) * H_k * (I - rho_k * y_k * T(s_k))
                + rho_k * s_k * T(s_k),
    
                            1
        where rho_k = ----------------.
                        T(s_k) * y_k

    Note, the symmetric positive definite property of H_k+1 requires
        
        T(s_k) * y_k > 0.
    
    This is the curvature condition. It's known that a line search result that 
    satisfies the strong Wolfe conditions is guaranteed to meet the curvature 
    condition.

    Args:
        
    """
    rho_k = 1. / einsum('...i, ...i', sk, yk)
    rho_k = ternary(paddle.isinf(rho_k), paddle.zeros_like(rho_k), rho_k)

    # Enforces the curvature condition before updating the inverse Hessian.
    if enforce_curvature:
        assert not any_active_with_predicates(rho_k <= 0)
    else:
        update_state(state.state, rho_k <= 0, 'failed')

    # By expanding the updating formula we obtain a sum of tensor products
    #
    #      H_k+1 = H_k 
    #              - rho * H_k * y_k * T(s_k)    ----- (2)
    #              - rho * s_k * T(y_k) * H_k    ----- (3)
    #              + rho * s_k * T(s_k)                            ----- (4)
    #              + rho * rho * (T(y_k) * H_k * y_k) s_k * T(s_k) ----- (5)
    #
    # Since H_k is symmetric, (3) is (2)'s transpose.
    # H_k * y_k
    Hy = einsum('...ij, ...j', Hk, yk)
    
    # T(y_k) * H_y * y_k
    yHy = einsum('...i, ...i', Hy, yk)

    term23 = einsum('...i, ...j', Hy, sk) + einsum('...i, ...j', sk, Hy)

    # T(s_k) * s_k
    sTs = einsum('...i, ...i', sk, sk)

    term45 = sTs * (1 + rho_k * yHy).unsqueeze(-1).unqueeze(-1)

    Hk_next = Hk + (term45 - term23) * rho_k.unsqueeze(-1).unsqueeze(-1)

    return Hk_next


def iterates(func,
             x0, 
             dtype='float32',
             H0=None,
             gtol=1e-8,
             xtol=0,
             iters=50,
             ls_iters=50):
    r"""minimizes a differentiable function `func` using the BFGS method,
    generating one iterate at a time.

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
        A generator which yields the result `SearchState` for each iteration.
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

    # Calculates the gradient norms
    gnorm = vnorm_inf(state.gk)

    # Updates the state tensor on the newly converged elements.
    state.state = update_state(state.state, gnorm < gtol, 'converged')

    try:
        # Starts to count the number of iterations.
        iter_count = StopCounter(iters)
        iter_count.increment()

        while any_active(state.state):
            k, xk, fk, gk, Hk = state.k, state.xk, state.fk, state.gk, state.Hk
 
            # The negative product of inverse Hessian and gradients - H_k * g_k
            # is used as the line search direction. 
            state.pk = pk = -einsum('...ij, ...j', Hk, gk)

            # Performs line search and updates the state
            linesearch(state,
                       func, 
                       gtol=gtol,
                       xtol=xtol,
                       max_iters=ls_iters)
        
            # Uses the obtained search steps to generate next iterates.
            next_xk = xk + einsum('..., ...i -> ...i', state.ak, pk)
    
            # Calculates displacement s_k = x_k+1 - x_k
            sk = next_xk - xk

            # Obtains the function values and gradients at x_k+1
            next_fk, next_gk = vjp(func, next_xk)
    
            # Calculates the gradient difference y_k = g_k+1 - g_k
            yk = next_gk - gk
    
            # Updates the approximate inverse hessian
            next_Hk = update_approx_inverse_hessian(state, Hk, sk, yk)

            # Finally transitions to the next state
            p = active_state(state.state)
            state.xk = ternary(p, next_xk, xk)
            state.fk = ternary(p, next_fk, fk)
            state.gk = ternary(p, next_gk, gk)
            state.Hk = ternary(p, next_Hk, Hk)
            state.k = k + 1

            # Calculates the gradient norms
            gnorm = vnorm_inf(next_gk)

            # Updates the state on the newly converged elements.
            state.state = update_state(state.state, gnorm < gtol, 'converged')

            # Counts iterations
            iter_count.increment()

            yield state
    except StopCounterException:
        pass
    finally:
        return


def optimize(func,
             x0,
             dtype='float32',
             H0=None,
             gtol=1e-8,
             xtol=0,
             iters=50,
             ls_iters=50,
             summary_only=True):
    r"""minimizes a differentiable function `func` using the BFGS method,
    returning the final result for summary or the list of results including
    all the intermediates.

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
        summary_only (boolean, optional): specifies the result type. If True 
            then returns the final result. Otherwise returns the results of
            all steps.
    
    Returns:
        summary (BfgsResult): The final optimization results if `summary_only`
            is set True.
        results (list[BfgsResult]): the results of all steps if `summary_only`
            is set False.
    """
    states = list(iterates(func, x0, dtype, H0, gtol, xtol, iters, ls_iters))

    if len(states) == 0:
        return None if summary_only else []

    if summary_only:
        return states[-1].result()

    return [s.result() for s in states]
