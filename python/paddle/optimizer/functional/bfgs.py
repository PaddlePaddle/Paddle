# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from .line_search import strong_wolfe
from .utils import _value_and_gradient

import paddle
from paddle.fluid.framework import in_dygraph_mode


def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(
        name=name, dtype=dtype, shape=shape)


def minimize_bfgs(objective_func,
                  initial_position,
                  max_iters=50,
                  tolerance_grad=1e-8,
                  tolerance_change=1e-9,
                  initial_inverse_hessian_estimate=None,
                  line_search_fn='strong_wolfe',
                  max_line_search_iters=50,
                  initial_step_length=1.0,
                  dtype='float32',
                  name=None):
    """
    Minimizes a differentiable function `func` using the BFGS method.
    The BFGS is a quasi-Newton method for solving an unconstrained
    optimization problem over a differentiable function.
    Closely related is the Newton method for minimization. Consider the iterate 
    update formula
    .. math::
        x_{k+1} = x_{k} + H^{-1} \nabla{f},
    If $H$ is the Hessian of $f$ at $x_{k}$, then it's the Newton method.
    If $H$ is symmetric and positive definite, used as an approximation of the Hessian, then 
    it's a quasi-Newton. In practice, the approximated Hessians are obtained
    by only using the gradients, over either whole or part of the search 
    history.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization,
        Second Edition, 2006.

    Args:
        objective_func: the objective function to minimize. ``func`` accepts
            a multivariate input and returns a scalar.
        initial_position (Tensor): the starting point of the iterates. For methods like Newton and quasi-Newton 
        the initial trial step length should always be 1.0.
        max_iters (int): the maximum number of minimization iterations.
        tolerance_grad (float): terminates if the gradient norm is smaller than this. Currently gradient norm uses inf norm.
        tolerance_change (float): terminates if the change of function value/position/parameter between 
            two iterations is smaller than this value.
        initial_inverse_hessian_estimate (Tensor): the initial inverse hessian approximation at initial_position.
        It must be symmetric and positive definite.
        line_search_fn (str): indicate which line search method to use, 'strong wolfe' or 'hager zhang'. 
            only support 'strong wolfe' right now.
        max_line_search_iters (int): the maximum number of line search iterations.
        initial_step_length (float): step length used in first iteration of line search. different initial_step_length 
        may cause different optimal result.
        dtype ('float32' | 'float64'): In static graph, float64 will be convert to float32 due to paddle.assign limit.
    
    Returns:
        is_converge (bool): Indicates whether found the minimum within tolerance.
        num_func_calls (int): number of objective function called.
        position (Tensor): the position of the last iteration. If the search converged, this value is the argmin of 
        the objective function regrading to the initial position.
        objective_value (Tensor): objective function value at the `position`.
        objective_gradient (Tensor): objective function gradient at the `position`.
        inverse_hessian_estimate (Tensor): the estimate of inverse hessian at the `position`.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            def func(x):
                return paddle.dot(x, x)

            x0 = np.random.random(size=[2]).astype('float32')
            results = miminize_bfgs(func, x0)
            print("is_converge: ", results[0])
            print("num_func_calls: ", results[1])
            print("the minimum of func is: ", results[2])
            # is_converge:  True
            # num_func_calls:  Tensor(shape=[1], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                    [9])
            # the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.42310646, 0.98076421])
    """
    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate is None:
        H0 = I
    else:
        if isinstance(initial_inverse_hessian_estimate,
                      np.ndarray) and in_dygraph_mode():
            # paddle.assign will convert numpy array float64 to float32, so when input is numpy, use another way
            # and set_value needs value to be `numpy.ndarray` or `LoDTensor`, so need to keep both ways.
            H0 = paddle.empty(
                initial_inverse_hessian_estimate.shape, dtype=dtype)
            H0.set_value(value=initial_inverse_hessian_estimate)
        else:
            H0 = paddle.assign(initial_inverse_hessian_estimate)

        is_symmetric = paddle.all(paddle.equal(H0, H0.t()))
        # In static mode, raise is not supported, but cholesky will throw preconditionNotMet if 
        # H0 is not symmetric or positive definite.
        if in_dygraph_mode():
            if not is_symmetric:
                raise ValueError(
                    "The initial_inverse_hessian_estimate should be symmetric, but the specified is not.\n{}".
                    format(H0))
            try:
                paddle.linalg.cholesky(H0)
            except RuntimeError as error:
                raise ValueError(
                    "The initial_inverse_hessian_estimate should be positive definite, but the specified is not.\n{}".
                    format(H0))
        else:

            def raise_func():
                raise ValueError(
                    "The initial_inverse_hessian_estimate should be symmetric, but the specified is not.\n{}".
                    format(H0))

            out_var = create_tmp_var(
                paddle.static.default_main_program(),
                name='output',
                dtype='float32',
                shape=[-1])

            def false_fn():
                paddle.static.nn.py_func(
                    func=raise_func, x=is_symmetric, out=out_var)

            paddle.static.nn.cond(is_symmetric, None, false_fn)
            paddle.linalg.cholesky(H0)

    Hk = paddle.assign(H0)
    if isinstance(initial_position, np.ndarray) and in_dygraph_mode():
        xk = paddle.empty(initial_position.shape, dtype=dtype)
        xk.set_value(value=initial_position)
    else:
        xk = paddle.assign(initial_position)

    value, g1 = _value_and_gradient(objective_func, xk)
    num_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')

    # when the dim of x is 1000, it needs more than 30 iters to get all element converge to minimum.
    k = paddle.full(shape=[1], fill_value=0, dtype='int64')
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')
    is_converge = paddle.full(shape=[1], fill_value=False, dtype='bool')

    def cond(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        return (k < max_iters) & ~done

    def body(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        pk = -paddle.matmul(Hk, g1)

        if line_search_fn == 'strong_wolfe':
            alpha, value, g2, ls_func_calls = strong_wolfe(
                f=objective_func,
                xk=xk,
                pk=pk,
                initial_step_length=initial_step_length,
                dtype=dtype)
        else:
            raise NotImplementedError(
                "Currently only support line_search_fn = 'strong_wolfe', but the specified is '{}'".
                format(line_search_fn))
        num_func_calls += ls_func_calls

        sk = alpha * pk
        yk = g2 - g1

        xk = xk + sk
        g1 = g2

        sk = paddle.unsqueeze(sk, 0)
        yk = paddle.unsqueeze(yk, 0)

        rhok_inv = paddle.dot(yk, sk)
        rhok = paddle.static.nn.cond(
            rhok_inv == 0., lambda: paddle.full(shape=[1], fill_value=1000.0, dtype=dtype), lambda: 1. / rhok_inv)

        Vk_transpose = I - rhok * sk * yk.t()
        Vk = I - rhok * yk * sk.t()
        Hk = paddle.matmul(paddle.matmul(Vk_transpose, Hk),
                           Vk) + rhok * sk * sk.t()

        k += 1

        gnorm = paddle.linalg.norm(g1, p=np.inf)
        pk_norm = paddle.linalg.norm(pk, p=np.inf)
        paddle.assign(done | (gnorm < tolerance_grad) |
                      (pk_norm < tolerance_change), done)
        paddle.assign(done, is_converge)

        paddle.assign(done | (alpha == 0.), done)
        return [k, done, is_converge, num_func_calls, xk, value, g1, Hk]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[k, done, is_converge, num_func_calls, xk, value, g1, Hk])
    return is_converge, num_func_calls, xk, value, g1, Hk
