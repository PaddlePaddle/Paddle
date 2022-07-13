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
from .utils import _value_and_gradient, check_input_type, check_initial_inverse_hessian_estimate

import paddle
from paddle.fluid.optimizer import Optimizer
from paddle import _C_ops


def minimize_bfgs(objective_func,
                  initial_position,
                  max_iters=10,
                  tolerance_grad=1e-7,
                  tolerance_change=1e-20,
                  initial_inverse_hessian_estimate=None,
                  line_search_fn='strong_wolfe',
                  max_line_search_iters=50,
                  initial_step_length=1.0,
                  dtype='float32',
                  name=None):
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
            - inverse_hessian_estimate (Tensor): the estimate of inverse hessian at the `position`.

    Examples:
        .. code-block:: python

            import paddle
            
            def func(x):
                return paddle.dot(x, x)

            x0 = paddle.to_tensor([1.3, 2.7])
            results = paddle.incubate.optimizer.functional.minimize_bfgs(func, x0)
            print("is_converge: ", results[0])
            print("the minimum of func is: ", results[2])
            # is_converge:  is_converge:  Tensor(shape=[1], dtype=bool, place=Place(gpu:0), stop_gradient=True,
            #        [True])
            # the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [0., 0.])
    """

    if dtype not in ['float32', 'float64']:
        raise ValueError(
            "The dtype must be 'float32' or 'float64', but the specified is {}.".
            format(dtype))

    op_name = 'minimize_bfgs'
    check_input_type(initial_position, 'initial_position', op_name)

    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate is None:
        initial_inverse_hessian_estimate = I
    else:
        check_input_type(initial_inverse_hessian_estimate,
                         'initial_inverse_hessian_estimate', op_name)
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
        #############    compute pk    #############
        pk = -paddle.matmul(Hk, g1)

        initial_step_length = 1. / pk.abs().sum()
        #print("initinitial_step_length: ", initial_step_length.item())
        #############    compute alpha by line serach    #############
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
        #print("value: ", value)
        print("alpha: ", alpha)
        num_func_calls += ls_func_calls
        #############    update Hk    #############
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

        #############    check convergence    #############

        gnorm = paddle.linalg.norm(g1, p=np.inf)
        pk_norm = paddle.linalg.norm(pk, p=np.inf)
        paddle.assign(done | (gnorm < tolerance_grad) |
                      (pk_norm < tolerance_change), done)
        paddle.assign(done, is_converge)
        # when alpha=0, there is no chance to get xk change.
        paddle.assign(done | (alpha < tolerance_change), done)
        return [k, done, is_converge, num_func_calls, xk, value, g1, Hk]

    paddle.static.nn.while_loop(
        cond=cond,
        body=body,
        loop_vars=[k, done, is_converge, num_func_calls, xk, value, g1, Hk])
    return is_converge, num_func_calls, xk, value, g1, Hk


class LossAndFlatGradient:
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, net, build_loss):
        self.net = net
        self.weights = self.net.parameters()
        self.build_loss = build_loss

        # Shapes of all trainable parameters
        self.shapes = [paddle.shape(weight) for weight in self.weights]
        self.n_tensors = len(self.shapes)

        # Information for tf.dynamic_stitch and tf.dynamic_partition later
        self.count = 0
        self.indices = []  # stitch indices
        self.partitions = []  # partition indices
        for i, shape in enumerate(self.shapes):

            n = paddle.prod(shape).item()
            self.indices.append(
                paddle.reshape(
                    paddle.arange(
                        self.count, self.count + n, dtype='float32'),
                    shape))
            self.partitions.append(n)
            self.count += n

    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        Args:
           weights_1d: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights
        for weight in self.weights:
            weight.clear_grad()
        self.set_flat_weights(weights_1d)
        loss = self.build_loss()
        loss.backward()
        grad = self.dynamic_stitch([param.grad for param in self.weights])

        return loss, grad

    def dynamic_stitch(self, inputs):
        flattened_weights = [paddle.flatten(weight) for weight in inputs]
        concat_weights = paddle.concat(flattened_weights)
        return concat_weights

    def dynamic_partition(self, weights_1d, partitions, n_tensors):
        split_weights = paddle.split(weights_1d, self.partitions)
        original_weights = [
            paddle.reshape(weight, shape)
            for weight, shape in zip(split_weights, self.shapes)
        ]
        return original_weights

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.
        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        #weights = self.dynamic_partition(weights_1d, self.partitions,
        #self.n_tensors)
        #for i, (shape, param) in enumerate(zip(self.shapes, weights)):
        #paddle.assign(self.net.parameters()[i], paddle.reshape(param, shape))

        with paddle.no_grad():
            for i in range(len(self._flat_weight)):
                self._flat_weight[i] = weights_1d[i]

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        _all_weights = [None] * len(self.weights)
        _all_weights = [w for w in self.weights]

        self._flat_weight = paddle.create_parameter(
            shape=[self.count], dtype='float32')
        #if paddle.fluid.in_dygraph_mode():
        with paddle.no_grad():
            _C_ops.coalesce_tensor(
                _all_weights, _all_weights, self._flat_weight, "copy_data",
                True, "use_align", False, "dtype", self.weights[0].dtype)

        # 计算loss后马上计算loss对flat_weight的梯度，仍然报错说无关
        # loss = self.build_loss()
        # gradient = paddle.grad([loss], [self._flat_weight], create_graph=False)[0]
        # print("gradient: ", gradient)
        return self._flat_weight
        # for static-graph, append coalesce_tensor into startup program
        # with program_guard(default_startup_program(),
        #                     default_startup_program()):
        #     with paddle.no_grad():
        #         self._helper.append_op(
        #             type="coalesce_tensor",
        #             inputs={"Input": self._all_weights},
        #             outputs={
        #                 "Output": self._all_weights,
        #                 "FusedOutput": self._flat_weight
        #             },
        #             attrs={
        #                 "copy_data": True,
        #                 "use_align": False,
        #                 "dtype": params[0].dtype
        #             })
        #     return self._flat_weight


def bfgs_minimize(net, build_loss, previous_optimizer_results=None):
    func = LossAndFlatGradient(net, build_loss)
    initial_position = func.to_flat_weights(net.parameters())
    results = minimize_bfgs(func, initial_position=initial_position)
    func.set_flat_weights(results[2])
    return results
