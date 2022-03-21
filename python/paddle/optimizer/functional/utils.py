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
from paddle.autograd.functional import vjp, Jacobian
from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid.data_feeder import check_type, check_dtype


def check_input_type(input, name, dtype, op_name):
    if in_dygraph_mode():
        if not isinstance(input, paddle.Tensor):
            raise ValueError("The input: {} must be tensor.".format(input))
    else:
        check_type(input, name, Variable, op_name)


def create_tmp_var(program, name, dtype, shape):
    return program.current_block().create_var(
        name=name, dtype=dtype, shape=shape)


def check_H0(H0):
    is_symmetric = paddle.all(paddle.equal(H0, H0.t()))

    def raise_func():
        raise ValueError(
            "The initial_inverse_hessian_estimate should be symmetric and positive definite, but the specified is not."
        )

    if in_dygraph_mode():
        if not is_symmetric:
            raise_func()
        try:
            paddle.linalg.cholesky(H0)
        except RuntimeError as error:
            raise_func()
    else:
        out_var = create_tmp_var(
            paddle.static.default_main_program(),
            name='output',
            dtype='float32',
            shape=[-1])

        def false_fn():
            paddle.static.nn.py_func(
                func=raise_func, x=is_symmetric, out=out_var)

        paddle.static.nn.cond(is_symmetric, None, false_fn)
        paddle.set_device("cpu")
        eigvals = paddle.paddle.linalg.eigvals(H0)
        is_positive = paddle.all(eigvals.real() > 0.) and paddle.all(
            eigvals.imag() == 0.)
        paddle.static.nn.cond(is_positive, None, false_fn)


def _value_and_gradient(f, x, v=None):
    if in_dygraph_mode():
        value, gradient = vjp(f, x, v=v)
        gradient = gradient[0]
    else:
        JJ = Jacobian(f, x)
        gradient = JJ[:][0]
        value = f(x)
    return value, gradient
