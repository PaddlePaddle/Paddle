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

import paddle
from paddle.fluid.framework import Variable
from paddle.fluid.data_feeder import check_type, check_dtype


def check_input_type(input, name, op_name):
    r"""Check whether the input is tensor or variable."""
    if paddle.in_dynamic_mode():
        if not isinstance(input, paddle.Tensor):
            raise ValueError("The input: {} must be tensor.".format(input))
    else:
        check_type(input, name, Variable, op_name)


def check_initial_inverse_hessian_estimate(H0):
    r"""Check whether the specified initial_inverse_hessian_estimate is symmetric and positive definite.
        Raise errors when precondition not met.

    Note: 
        In static graph can not raise error directly, so use py_func make raise_func as a op,
        and use paddle.static.nn.cond to decide if put the op in net.
        cholesky is the fast way to check positive definition, but in static graph can not catch 
        exception to raise value error, so use eigvals rather than cholesky in static graph.
    """
    is_symmetric = paddle.all(paddle.equal(H0, H0.t()))

    def raise_func():
        raise ValueError(
            "The initial_inverse_hessian_estimate should be symmetric and positive definite, but the specified is not."
        )

    if paddle.in_dynamic_mode():
        if not is_symmetric:
            raise_func()
        try:
            paddle.linalg.cholesky(H0)
        except RuntimeError as error:
            raise_func()
    else:

        def create_tmp_var(program, name, dtype, shape):
            return program.current_block().create_var(
                name=name, dtype=dtype, shape=shape)

        out_var = create_tmp_var(
            paddle.static.default_main_program(),
            name='output',
            dtype='float32',
            shape=[-1])

        def false_fn():
            paddle.static.nn.py_func(
                func=raise_func, x=is_symmetric, out=out_var)

        paddle.static.nn.cond(is_symmetric, None, false_fn)
        # eigvals only support cpu
        paddle.set_device("cpu")
        eigvals = paddle.paddle.linalg.eigvals(H0)
        is_positive = paddle.all(eigvals.real() > 0.) and paddle.all(
            eigvals.imag() == 0.)
        paddle.static.nn.cond(is_positive, None, false_fn)


def _value_and_gradient(f, x, v=None):
    r"""Compute function value and gradient of f at x.
    
    Args:
        f (Callable): the objective function.
        x (Tensor): the input tensor.
    Returns:
        value: a tensor that holds the function value.
        gradient: a tensor that holds the function gradients.  
    """
    # use detach to cut off relation between x and original graph
    x = x.detach()
    x.stop_gradient = False
    value = f(x)
    if paddle.in_dynamic_mode():
        # only need to compute first order derivative, and some op dont support high order derivative.
        gradient = paddle.grad([value], [x], create_graph=False)[0]
    else:
        gradient = paddle.static.gradients([value], [x])[0]
    # use detach to make results real number without grad to avoid assign error
    return value.detach(), gradient.detach()
