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
from paddle import dot, einsum
from .bfgs_utils import vjp, ternary
from .bfgs_utils import make_state, make_const
from .bfgs_utils import as_float_tensor, vnorm_inf
from .bfgs_utils import StepCounter, StepCounterException
from .bfgs_utils import SearchState
from .linesearch_new import HagerZhang


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
        f"(Batched) matrix {H} is not positive definite.")

    perm = list(range(len(H.shape)))
    perm[-2:] = perm[-1], perm[-2]
    # batch_deltas = paddle.norm(H - H.transpose(perm), axis=perm[-2:])
    # is_symmetic = paddle.sum(batch_deltas) == 0.0
    is_symmetric = paddle.allclose(H, H.transpose(perm))

    assert is_symmetric, (f"(Batched) matrix {H} is not symmetric.")


def update_approx_inverse_hessian(state, H, s, y, enforce_curvature=False):
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
    # batch = len(s.shape) > 1
    bat = state.bat

    with paddle.no_grad():
        rho = 1. / einsum('...i, ...i', s, y) if bat else 1. / paddle.dot(s, y)
        rho = ternary(paddle.isinf(rho), paddle.zeros_like(rho), rho)

        # Enforces the curvature condition before updating the inverse Hessian.
        if enforce_curvature:
            assert not state.any_active_with_predicates(rho <= 0)
        else:
            state.update_state(~state.stop & (rho <= 0), 'failed')

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
        L = paddle.cholesky(H)
        yL = einsum('...i, ...ij', y, L)

        yLL = einsum('...j, ...ij', yL, L)

        yLLy = einsum('...i, ...i', yL, yL) if bat else paddle.dot(yL, yL)

        ss = einsum('...i, ...j', s, s)

        syLL = einsum('...i, ...j', s, yLL)

        t = einsum('..., ...ij', 1. + rho * yLLy, ss) if bat else (
            1. + rho * yLLy) * ss
        t1 = t - syLL - einsum('...ij->...ji', syLL)
        Hk_next = H + einsum('..., ...ij', rho, t1) if bat else H + rho * t1

        # Hy = einsum('...ij, ...j', H, y)
        # # T(y_k) * H_y * y_k
        # yHy = einsum('...i, ...i', y, Hy) if bat else dot(y, Hy)
        # syH = einsum('...i, ...j -> ...ji', Hy, s)
        # term23 = syH + einsum('...ij->...ji', syH)
        # # T(s_k) * s_k
        # sTs = einsum('...i, ...j', s, s)

        # if bat:
        #     term45 = einsum('...ij, ...', sTs, 1 + rho * yHy)
        #     Hk_next = H + einsum('...ij, ...', term45 - term23, rho)
        # else:
        #     term45 = sTs * (1 + rho * yHy)
        #     Hk_next = H + (term45 - term23) * rho

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
    # tensor holds the input dimensions. However, it's tricky to determine
    # whether a function is actually applied in the batch mode. We assume here
    # that the input being multi-dimensional necessarily implies batching.
    bat = len(x0.shape) > 1
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
    # state = SearchState(bat, x0, f0, g0, H0, gnorm,
    #                     iters=iters, ls_iters=ls_iters)
    HZ = HagerZhang(func, bat, x0, f0, g0, H0, gnorm, ls_iters=ls_iters)

    # Updates the state tensor on the newly converged elements.
    HZ.update_state(gnorm < gtol, 'converged')

    try:
        # Starts to count the number of iterations.
        iter_count = StepCounter(iters)
        iter_count.increment()

        while HZ.any_active():
            k, xk, fk, gk, Hk = HZ.k, HZ.xk, HZ.fk, HZ.gk, HZ.Hk

            # The negative product of inverse Hessian and gradients - H_k * g_k
            # is used as the line search direction. 
            # The negative inner product of approximate inverse hessian and 
            # gradient gives the line search direction p_k. Immediately after 
            # p_k is calculated, the directional derivative on p_k should be 
            # checked to make sure the p_k is a descending direction. If that's 
            # not the case, then sets the line search state as failed for the 
            # corresponding batching element.
            pk = -einsum('...ij, ...j', Hk, gk)

            # Performs line search and updates the state
            ak = HZ.linesearch(pk)

            # Uses the obtained search steps to generate next iterates.
            if bat:
                next_xk = xk + ak.unsqueeze(-1) * pk
            else:
                next_xk = xk + ak * pk
            # Calculates displacement s_k = x_k+1 - x_k

            sk = next_xk - xk

            # Obtains the function values and gradients at x_k+1
            next_fk, next_gk = vjp(func, next_xk)

            # Calculates the gradient difference y_k = g_k+1 - g_k
            yk = next_gk - gk

            # Updates the approximate inverse hessian
            next_Hk = update_approx_inverse_hessian(HZ, Hk, sk, yk)

            # Calculates the gradient norms
            next_gnorm = vnorm_inf(next_gk)

            # Finally transitions to the next state
            p = HZ.active_state()
            HZ.xk = ternary(p, next_xk, xk)
            HZ.fk = ternary(p, next_fk, fk)
            HZ.gk = ternary(p, next_gk, gk)
            HZ.Hk = ternary(p, next_Hk, Hk)
            HZ.gnorm = ternary(p, next_gnorm, HZ.gnorm)

            # Updates the state on the newly converged elements.
            HZ.update_state(HZ.gnorm < gtol, 'converged')
            HZ.update_state(HZ.stop_lowerbound, 'converged')
            HZ.update_state(HZ.stop_blowup, 'blowup')

            HZ.reset_grads()
            HZ.k = k + 1

            # Counts iterations
            iter_count.increment()

            yield HZ
    except StepCounterException:
        pass
    finally:
        return


def minimize(func,
             x0,
             dtype='float32',
             H0=None,
             gtol=1e-8,
             xtol=0,
             iters=50,
             ls_iters=50,
             summary_only=True):
    r"""Minimizes a differentiable function `func` using the BFGS method.

    The BFGS is a quasi-Newton method for solving an unconstrained
    optimization problem over a differentiable function.

    Closely related is the Newton method for minimization. Consider the iterate 
    update formula

    .. math::

        x_{k+1} = x_{k} + H^{-1} \nabla{f},

    If $H$ is the Hessian of $f$ at $x_{k}$, then it's the Newton method.
    If $H$ is positive definite, used as an approximation of the Hessian, then 
    it's a quasi-Newton. In practice, the approximated Hessians are obtained
    by only using the gradients, over either whole or part of the search 
    history.

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
    states = []
    final_state = None
    for state in iterates(func, x0, dtype, H0, gtol, xtol, iters, ls_iters):
        if summary_only:
            final_state = state
        else:
            states.append(state)

    if summary_only:
        if final_state:
            return final_state.result()
        else:
            return {}

    return [s.result() for s in states]
