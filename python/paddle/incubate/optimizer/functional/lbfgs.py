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

import paddle

from .line_search import strong_wolfe
from .utils import (
    _value_and_gradient,
    check_initial_inverse_hessian_estimate,
    check_input_type,
)


def minimize_lbfgs(
    objective_func,
    initial_position,
    history_size=100,
    max_iters=50,
    tolerance_grad=1e-8,
    tolerance_change=1e-8,
    initial_inverse_hessian_estimate=None,
    line_search_fn='strong_wolfe',
    max_line_search_iters=50,
    initial_step_length=1.0,
    dtype='float32',
    name=None,
):
    r"""
    Minimizes a differentiable function `func` using the L-BFGS method.
    The L-BFGS is a quasi-Newton method for solving an unconstrained optimization problem over a differentiable function.
    Closely related is the Newton method for minimization. Consider the iterate update formula:

    .. math::
        x_{k+1} = x_{k} + H_k \nabla{f_k}

    If :math:`H_k` is the inverse Hessian of :math:`f` at :math:`x_k`, then it's the Newton method.
    If :math:`H_k` is symmetric and positive definite, used as an approximation of the inverse Hessian, then
    it's a quasi-Newton. In practice, the approximated Hessians are obtained
    by only using the gradients, over either whole or part of the search
    history, the former is BFGS, the latter is L-BFGS.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp179: Algorithm 7.5 (L-BFGS).

    Args:
        objective_func: the objective function to minimize. ``objective_func`` accepts a 1D Tensor and returns a scalar.
        initial_position (Tensor): the starting point of the iterates, has the same shape with the input of ``objective_func`` .
        history_size (Scalar): the number of stored vector pairs {si,yi}. Default value: 100.
        max_iters (int, optional): the maximum number of minimization iterations. Default value: 50.
        tolerance_grad (float, optional): terminates if the gradient norm is smaller than this. Currently gradient norm uses inf norm. Default value: 1e-7.
        tolerance_change (float, optional): terminates if the change of function value/position/parameter between two iterations is smaller than this value. Default value: 1e-9.
        initial_inverse_hessian_estimate (Tensor, optional): the initial inverse hessian approximation at initial_position. It must be symmetric and positive definite. If not given, will use an identity matrix of order N, which is size of ``initial_position`` . Default value: None.
        line_search_fn (str, optional): indicate which line search method to use, only support 'strong wolfe' right now. May support 'Hager Zhang' in the futrue. Default value: 'strong wolfe'.
        max_line_search_iters (int, optional): the maximum number of line search iterations. Default value: 50.
        initial_step_length (float, optional): step length used in first iteration of line search. different initial_step_length may cause different optimal result. For methods like Newton and quasi-Newton the initial trial step length should always be 1.0. Default value: 1.0.
        dtype ('float32' | 'float64', optional): data type used in the algorithm, the data type of the input parameter must be consistent with the dtype. Default value: 'float32'.
        name (str, optional): Name for the operation. For more information, please refer to :ref:`api_guide_Name`. Default value: None.

    Returns:
        output(tuple):

            - is_converge (bool): Indicates whether found the minimum within tolerance.
            - num_func_calls (int): number of objective function called.
            - position (Tensor): the position of the last iteration. If the search converged, this value is the argmin of the objective function regrading to the initial position.
            - objective_value (Tensor): objective function value at the `position`.
            - objective_gradient (Tensor): objective function gradient at the `position`.

    Examples:
        .. code-block:: python

            import paddle

            def func(x):
                return paddle.dot(x, x)

            x0 = paddle.to_tensor([1.3, 2.7])
            results = paddle.incubate.optimizer.functional.minimize_lbfgs(func, x0)
            print("is_converge: ", results[0])
            print("the minimum of func is: ", results[2])
            # is_converge:  is_converge:  Tensor(shape=[1], dtype=bool, place=Place(gpu:0), stop_gradient=True,
            #        [True])
            # the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0., 0.])
    """
    if dtype not in ['float32', 'float64']:
        raise ValueError(
            "The dtype must be 'float32' or 'float64', but the specified is {}.".format(
                dtype
            )
        )

    op_name = 'minimize_lbfgs'
    check_input_type(initial_position, 'initial_position', op_name)

    if initial_inverse_hessian_estimate is None:
        H0 = paddle.eye(initial_position.shape[0], dtype=dtype)
    else:
        check_input_type(
            initial_inverse_hessian_estimate,
            'initial_inverse_hessian_estimate',
            op_name,
        )
        check_initial_inverse_hessian_estimate(initial_inverse_hessian_estimate)
        H0 = initial_inverse_hessian_estimate

    # use detach and assign to create new tensor rather than =, or xk will share memory and grad with initial_position
    xk = paddle.assign(initial_position.detach())
    value, g1 = _value_and_gradient(objective_func, xk)

    k = paddle.full(shape=[1], fill_value=0, dtype='int64')
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')
    is_converge = paddle.full(shape=[1], fill_value=False, dtype='bool')
    num_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')

    history_size = paddle.full(shape=[], fill_value=history_size, dtype='int64')
    head = paddle.full(shape=[1], fill_value=1, dtype='int64')
    tail = paddle.full(shape=[1], fill_value=0, dtype='int64')

    shape = initial_position.shape[0]
    # Use tensor as array of fixed length, rather than flexible tensor array. Because in static graph mode,
    # tensor array will produce tensor of shape[-1], which will cause error when calling jacobian. In this way, can not use append
    # or pop, so we need head and tail to record where is the newest data and where is the oldest.
    # Totally speaking, realized a stack by array.
    sk_vec = paddle.zeros((history_size + 1, shape), dtype=dtype)
    yk_vec = paddle.zeros((history_size + 1, shape), dtype=dtype)
    rhok_vec = paddle.zeros((history_size + 1, 1), dtype=dtype)
    ai_vec = paddle.zeros((history_size + 1, 1), dtype=dtype)

    def cond(
        k,
        done,
        is_converge,
        num_func_calls,
        value,
        xk,
        g1,
        sk_vec,
        yk_vec,
        rhok_vec,
        head,
        tail,
    ):
        return (k < max_iters) & ~done

    def body(
        k,
        done,
        is_converge,
        num_func_calls,
        value,
        xk,
        g1,
        sk_vec,
        yk_vec,
        rhok_vec,
        head,
        tail,
    ):
        # use assign to cut off the relevance between g1 and q, or they will change together.

        # --------------   compute p_k by two-loop recursion    -------------- #
        q = paddle.assign(g1)
        # In a array circle, the index may out of range, so must use mod.
        i = paddle.full(
            shape=[], fill_value=(head - 1).mod(history_size), dtype='int64'
        )

        def cond(i, q):
            return i != tail

        def body(i, q):
            ai_vec[i] = rhok_vec[i] * paddle.dot(sk_vec[i], q)
            q = q - ai_vec[i] * yk_vec[i]
            i = (i - 1).mod(history_size)
            return i, q

        paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, q])

        r = paddle.matmul(H0, q)

        i = paddle.full(shape=[], fill_value=tail + 1, dtype='int64')

        def cond(i, r):
            return i != head

        def body(i, r):
            beta = rhok_vec[i] * paddle.dot(yk_vec[i], r)
            r = r + sk_vec[i] * (ai_vec[i] - beta)
            i = (i + 1).mod(history_size)
            return i, r

        paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[i, r])

        pk = -r

        # --------------   compute alpha by line serach    -------------- #
        if line_search_fn == 'strong_wolfe':
            alpha, value, g2, ls_func_calls = strong_wolfe(
                f=objective_func,
                xk=xk,
                pk=pk,
                max_iters=max_line_search_iters,
                initial_step_length=initial_step_length,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(
                "Currently only support line_search_fn = 'strong_wolfe', but the specified is '{}'".format(
                    line_search_fn
                )
            )
        paddle.assign(num_func_calls + ls_func_calls, num_func_calls)

        # --------------   update sk_vec, yk_vec, rhok_vec    -------------- #
        sk = alpha * pk
        yk = g2 - g1

        rhok_inv = paddle.dot(yk, sk)
        rhok = paddle.static.nn.cond(
            rhok_inv == 0.0,
            lambda: paddle.full(shape=[1], fill_value=1000.0, dtype=dtype),
            lambda: 1.0 / rhok_inv,
        )

        sk_vec[head] = sk
        yk_vec[head] = yk
        rhok_vec[head] = rhok
        head = (head + 1) % history_size

        def true_fn(tail):
            paddle.assign(tail + 1, tail)

        # when array is full, the tail should move forward too.
        paddle.static.nn.cond(head == tail, lambda: true_fn(tail), None)

        xk = xk + sk
        g1 = g2
        k += 1

        # --------------   check convergence    -------------- #
        gnorm = paddle.linalg.norm(g1, p=np.inf)
        pk_norm = paddle.linalg.norm(pk, p=np.inf)
        paddle.assign(
            done | (gnorm < tolerance_grad) | (pk_norm < tolerance_change), done
        )
        paddle.assign(done, is_converge)
        # when alpha=0, there is no chance to get xk change.
        paddle.assign(done | (alpha == 0.0), done)

        return [
            k,
            done,
            is_converge,
            num_func_calls,
            value,
            xk,
            g1,
            sk_vec,
            yk_vec,
            rhok_vec,
            head,
            tail,
        ]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[
            k,
            done,
            is_converge,
            num_func_calls,
            value,
            xk,
            g1,
            sk_vec,
            yk_vec,
            rhok_vec,
            head,
            tail,
        ],
    )
    return is_converge, num_func_calls, xk, value, g1
