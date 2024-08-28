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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

import paddle

from .line_search import strong_wolfe
from .utils import (
    _value_and_gradient,
    check_initial_inverse_hessian_estimate,
    check_input_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor


def minimize_bfgs(
    objective_func: Callable[[Tensor], Tensor],
    initial_position: Tensor,
    max_iters: int = 50,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    initial_inverse_hessian_estimate: Tensor | None = None,
    line_search_fn: Literal['strong_wolfe'] = 'strong_wolfe',
    max_line_search_iters: int = 50,
    initial_step_length: float = 1.0,
    dtype: Literal['float32', 'float64'] = 'float32',
    name: str | None = None,
) -> tuple[bool, int, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Minimizes a differentiable function `func` using the BFGS method.
    The BFGS is a quasi-Newton method for solving an unconstrained optimization problem over a differentiable function.
    Closely related is the Newton method for minimization. Consider the iterate update formula:

    .. math::
        x_{k+1} = x_{k} + H_k \nabla{f_k}

    If :math:`H_k` is the inverse Hessian of :math:`f` at :math:`x_k`, then it's the Newton method.
    If :math:`H_k` is symmetric and positive definite, used as an approximation of the inverse Hessian, then
    it's a quasi-Newton. In practice, the approximated Hessians are obtained
    by only using the gradients, over either whole or part of the search
    history, the former is BFGS, the latter is L-BFGS.

    Reference:
        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp140: Algorithm 6.1 (BFGS Method).

    Args:
        objective_func: the objective function to minimize. ``objective_func`` accepts a 1D Tensor and returns a scalar.
        initial_position (Tensor): the starting point of the iterates, has the same shape with the input of ``objective_func`` .
        max_iters (int, optional): the maximum number of minimization iterations. Default value: 50.
        tolerance_grad (float, optional): terminates if the gradient norm is smaller than this. Currently gradient norm uses inf norm. Default value: 1e-7.
        tolerance_change (float, optional): terminates if the change of function value/position/parameter between two iterations is smaller than this value. Default value: 1e-9.
        initial_inverse_hessian_estimate (Tensor, optional): the initial inverse hessian approximation at initial_position. It must be symmetric and positive definite. If not given, will use an identity matrix of order N, which is size of ``initial_position`` . Default value: None.
        line_search_fn (str, optional): indicate which line search method to use, only support 'strong wolfe' right now. May support 'Hager Zhang' in the future. Default value: 'strong wolfe'.
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
            - inverse_hessian_estimate (Tensor): the estimate of inverse hessian at the `position`.

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # Example1: 1D Grid Parameters
            >>> import paddle
            >>> # Randomly simulate a batch of input data
            >>> inputs = paddle. normal(shape=(100, 1))
            >>> labels = inputs * 2.0
            >>> # define the loss function
            >>> def loss(w):
            ...     y = w * inputs
            ...     return paddle.nn.functional.square_error_cost(y, labels).mean()
            >>> # Initialize weight parameters
            >>> w = paddle.normal(shape=(1,))
            >>> # Call the bfgs method to solve the weight that makes the loss the smallest, and update the parameters
            >>> for epoch in range(0, 10):
            ...     # Call the bfgs method to optimize the loss, note that the third parameter returned represents the weight
            ...     w_update = paddle.incubate.optimizer.functional.minimize_bfgs(loss, w)[2]
            ...     # Use paddle.assign to update parameters in place
            ...     paddle. assign(w_update, w)

        .. code-block:: python
            :name: code-example2

            >>> # Example2: Multidimensional Grid Parameters
            >>> import paddle
            >>> def flatten(x):
            ...     return x. flatten()
            >>> def unflatten(x):
            ...     return x.reshape((2,2))
            >>> # Assume the network parameters are more than one dimension
            >>> def net(x):
            ...     assert len(x.shape) > 1
            ...     return x.square().mean()
            >>> # function to be optimized
            >>> def bfgs_f(flatten_x):
            ...     return net(unflatten(flatten_x))
            >>> x = paddle.rand([2,2])
            >>> for i in range(0, 10):
            ...     # Flatten x before using minimize_bfgs
            ...     x_update = paddle.incubate.optimizer.functional.minimize_bfgs(bfgs_f, flatten(x))[2]
            ...     # unflatten x_update, then update parameters
            ...     paddle.assign(unflatten(x_update), x)
    """

    if dtype not in ['float32', 'float64']:
        raise ValueError(
            f"The dtype must be 'float32' or 'float64', but the specified is {dtype}."
        )

    op_name = 'minimize_bfgs'
    check_input_type(initial_position, 'initial_position', op_name)

    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate is None:
        initial_inverse_hessian_estimate = I
    else:
        check_input_type(
            initial_inverse_hessian_estimate,
            'initial_inverse_hessian_estimate',
            op_name,
        )
        check_initial_inverse_hessian_estimate(initial_inverse_hessian_estimate)

    Hk = paddle.assign(initial_inverse_hessian_estimate)
    # use detach and assign to create new tensor rather than =, or xk will share memory and grad with initial_position
    xk = paddle.assign(initial_position.detach())

    value, g1 = _value_and_gradient(objective_func, xk)
    num_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')

    # when the dim of x is 1000, it needs more than 30 iters to get all element converge to minimum.
    k = paddle.full(shape=[1], fill_value=0, dtype='int64')
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')
    is_converge = paddle.full(shape=[1], fill_value=False, dtype='bool')

    def cond(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        return (k < max_iters) & ~done

    def body(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        # --------------   compute pk   -------------- #
        pk = -paddle.matmul(Hk, g1)

        # --------------   compute alpha by line search   -------------- #
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
                f"Currently only support line_search_fn = 'strong_wolfe', but the specified is '{line_search_fn}'"
            )
        num_func_calls += ls_func_calls

        # --------------   update Hk   -------------- #
        sk = alpha * pk
        yk = g2 - g1

        xk = xk + sk
        g1 = g2

        sk = paddle.unsqueeze(sk, 0)
        yk = paddle.unsqueeze(yk, 0)

        rhok_inv = paddle.dot(yk, sk)
        rhok = paddle.static.nn.cond(
            rhok_inv == 0.0,
            lambda: paddle.full(shape=[1], fill_value=1000.0, dtype=dtype),
            lambda: 1.0 / rhok_inv,
        )

        Vk_transpose = I - rhok * sk * yk.t()
        Vk = I - rhok * yk * sk.t()
        Hk = (
            paddle.matmul(paddle.matmul(Vk_transpose, Hk), Vk)
            + rhok * sk * sk.t()
        )

        k += 1

        # --------------   check convergence   -------------- #
        gnorm = paddle.linalg.norm(g1, p=np.inf)
        pk_norm = paddle.linalg.norm(pk, p=np.inf)
        paddle.assign(
            done | (gnorm < tolerance_grad) | (pk_norm < tolerance_change), done
        )
        paddle.assign(done, is_converge)
        # when alpha=0, there is no chance to get xk change.
        paddle.assign(done | (alpha == 0.0), done)
        return [k, done, is_converge, num_func_calls, xk, value, g1, Hk]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[k, done, is_converge, num_func_calls, xk, value, g1, Hk],
    )
    return is_converge, num_func_calls, xk, value, g1, Hk
