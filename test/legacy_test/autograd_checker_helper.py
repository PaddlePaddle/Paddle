# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from collections.abc import Sequence
from itertools import product
from logging import warning

import numpy as np

import paddle
from paddle import base
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core
from paddle.base.backward import _as_list

__all__ = ['check_vjp']

EPS = 1e-4

default_gradient_tolerance = {
    np.float16: 1e-2,
    np.float32: 2e-3,
    np.float64: 1e-5,
    np.complex64: 1e-3,
    np.complex128: 1e-5,
}


def _product(t):
    return int(np.prod(t))


def make_jacobian(x, y_size, np_dtype):
    if isinstance(x, (base.framework.Variable, paddle.pir.Value)):
        return np.zeros((_product(x.shape), y_size), dtype=np_dtype)
    elif isinstance(x, Sequence):
        jacobians = list(
            filter(
                lambda t: t is not None,
                (make_jacobian(item, y_size, np_dtype) for item in x),
            )
        )
        return jacobians
    else:
        pass


def _compute_numerical_jacobian(program, x, y, feeds, eps):
    if not isinstance(x, paddle.pir.Value):
        raise TypeError('x is not Value')

    # To compute the jacobian, treat x and y as one-dimensional vectors.
    y = _as_list(y)
    exe = paddle.static.Executor()

    def run():
        res = exe.run(program, feeds, fetch_list=[y])
        y_res = res[: len(y)]
        return [yi.flatten() for yi in y_res]

    x_name = x.get_defining_op().attrs()['name']
    x_shape = x.shape
    x_size = _product(x_shape)
    np_type = dtype_to_np_dtype(x.dtype)
    np_t = np.array(feeds[x_name]).astype(np_type)
    np_t = np_t.flatten()
    jacobian = [make_jacobian(x, _product(yi.shape), np_type) for yi in y]

    for i in range(x_size):
        orig = np_t[i]
        x_pos = orig + eps
        np_t[i] = x_pos
        np_f = np_t.reshape(x_shape)
        feeds[x_name] = np_f
        y_pos = run()

        x_neg = orig - eps
        np_t[i] = x_neg
        np_f = np_t.reshape(x_shape)
        feeds[x_name] = np_f
        y_neg = run()

        np_t[i] = orig
        for j in range(len(y)):
            ret = (y_pos[j] - y_neg[j]) / eps / 2.0
            jacobian[j][i, :] = ret

    return jacobian


def _compute_analytical_jacobian(program, x, i, y, grads, feeds, name):
    if not isinstance(x, (list, paddle.pir.Value)):
        raise TypeError('x is not Value or list of Value')
    np_type = dtype_to_np_dtype(y[i].dtype)
    exe = paddle.static.Executor()
    y_size = _product(y[i].shape)
    x = _as_list(x)
    jacobian = make_jacobian(x, y_size, np_type)

    # get the name in feeds of dyi
    np_t = np.array(feeds[name]).astype(np_type)
    shape = np_t.shape
    np_t = np_t.flatten()
    for i in range(y_size):
        np_t[i] = 1
        np_f = np_t.reshape(shape)
        feeds[name] = np_f
        res = exe.run(program, feed=feeds, fetch_list=[grads])
        dx_res = res[: len(grads)]
        for j in range(len(grads)):
            if dx_res[j] is not None:
                jacobian[j][:, i] = dx_res[j].flatten()
            else:
                jacobian[j][:, i] = np.zeros(
                    grads[j].shape, dtype=np_type
                ).flatten()

        np_t[i] = 0
        np_f = np_t.reshape(shape)
        feeds[name] = np_f

    return jacobian


def grad_check(
    inputs,
    outputs,
    last_grads_in,
    feeds=None,
    fetch_list=None,
    program=None,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    raise_exception=True,
):
    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    analytical = []
    for i in range(len(outputs)):
        name = last_grads_in[i].name
        feeds.update(
            {
                name: np.zeros(
                    outputs[i].shape, dtype=dtype_to_np_dtype(outputs[i].dtype)
                )
            }
        )
    for i in range(len(outputs)):
        analytical.append(
            _compute_analytical_jacobian(
                program,
                inputs,
                i,
                outputs,
                fetch_list,
                feeds,
                last_grads_in[i].name,
            )
        )
    numerical = [
        _compute_numerical_jacobian(program, input, outputs, feeds, eps)
        for input in inputs
    ]
    for i, (x_idx, y_idx) in enumerate(
        product(*[range(len(inputs)), range(len(outputs))])
    ):
        a = analytical[y_idx][x_idx]
        n = numerical[x_idx][y_idx]
        if not np.allclose(a, n, rtol, atol):
            msg = (
                f'Jacobian mismatch for output {y_idx} in y'
                f'with respect to input {x_idx} in x,\n'
                f'numerical:{n}\nanalytical:{a}\n'
            )
            return fail_test(msg)
    return True


def dtype_to_np_dtype(dtype):
    if dtype == core.VarDesc.VarType.FP32 or dtype == core.DataType.FLOAT32:
        return np.float32
    elif dtype == core.VarDesc.VarType.FP64 or dtype == core.DataType.FLOAT64:
        return np.float64
    elif dtype == core.VarDesc.VarType.FP16 or dtype == core.DataType.FLOAT16:
        return np.float16
    else:
        raise ValueError("Not supported data type " + str(dtype))


def get_eager_vjp(func, inputs, tangents=None, order=1):
    for x in inputs:
        x.stop_gradient = False
    outputs = func(inputs)
    if not tangents:
        tangents = []
        y = _as_list(outputs)
        for yi in y:
            v = paddle.randn(yi.shape, yi.dtype)
            v.stop_gradient = False
            tangents.append(v)
    return _get_eager_vjp(inputs, outputs, tangents, order), tangents


def _get_eager_vjp(inputs, outputs, tangents, order):
    if order > 1:
        create_graph = True
        allow_unused = True
    else:
        create_graph = False
        allow_unused = False

    d_inputs = paddle.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=tangents,
        create_graph=create_graph,
        allow_unused=allow_unused,
    )
    d_inputs = [d_input for d_input in d_inputs if d_input is not None]
    if order > 1:
        ddys = []
        for d_input in d_inputs:
            d_input.stop_gradient = False
            ddy = paddle.ones(shape=d_input.shape, dtype=d_input.dtype)
            ddy.stop_gradient = False
            ddys.append(ddy)
        return _get_eager_vjp(inputs, d_inputs, ddys, order - 1)

    return d_inputs


def get_static_vjp(func, inputs, tangents, order, atol, rtol, eps):
    tangents = _as_list(tangents)
    tangents = [tangent.numpy() for tangent in tangents]
    paddle.enable_static()
    input_vars = []
    feeds = {}
    for idx, input in enumerate(inputs):
        np_type = dtype_to_np_dtype(input.dtype)
        input_var = paddle.static.data(
            'input_' + str(idx), input.shape, dtype=np_type
        )
        input_vars.append(input_var)
        feeds.update({'input_' + str(idx): input.numpy()})
    outputs = func(input_vars)
    outputs = _as_list(outputs)
    program, (keys, values) = paddle.base.libpaddle.pir.clone_program(
        paddle.static.default_main_program()
    )
    op_map = ValueDict()
    for key, value in zip(keys, values):
        op_map[key] = value
    pir_inputs = []
    for input in input_vars:
        pir_inputs.append(op_map[input])
    pir_outputs = []
    grads_in_init = []
    with paddle.static.program_guard(program):
        # Make sure the grad_in_var is in the program
        for idx, output in enumerate(outputs):
            pir_outputs.append(op_map[output])
            np_type = dtype_to_np_dtype(input.dtype)
            grad_in_var = paddle.static.data(
                'grad_in_' + str(idx), output.shape, dtype=np_type
            )
            grads_in_init.append(grad_in_var)
            feeds.update({'grad_in_' + str(idx): tangents[idx]})
        feeds, pre_outputs, d_inputs, last_grads_in = _get_static_vjp_program(
            pir_inputs, pir_outputs, feeds, grads_in_init, order
        )
        exe = paddle.static.Executor()
        res = exe.run(program, feed=feeds, fetch_list=[d_inputs])
    if not d_inputs:
        warning(f"{func.__name__} {order}s grad will return None")
        return []
    grad_check(
        pir_inputs,
        pre_outputs,
        last_grads_in,
        feeds,
        d_inputs,
        program,
        eps,
        atol,
        rtol,
    )
    paddle.disable_static()
    return res


def _get_static_vjp_program(inputs, outputs, feeds, grads_in, order):
    def _require_grads(vars):
        for var in vars:
            var.stop_gradient = False
            var.persistable = True

    inputs = _as_list(inputs)
    outputs = _as_list(outputs)
    _require_grads(inputs)
    _require_grads(outputs)
    _require_grads(grads_in)
    d_inputs = paddle.base.gradients(outputs, inputs, grads_in)
    d_inputs = [d_input for d_input in d_inputs if d_input is not None]
    _require_grads(d_inputs)

    if order > 1:
        ddys = []
        for idx, d_input in enumerate(d_inputs):
            np_type = dtype_to_np_dtype(d_input.dtype)
            ddy = paddle.static.data(
                name=f'dy_{idx}_{order}',
                shape=d_input.shape,
                dtype=np_type,
            )
            ones = np.ones(d_input.shape, dtype=np_type)
            feeds.update({f'dy_{idx}_{order}': ones})
            ddys.append(ddy)
        _require_grads(ddys)
        return _get_static_vjp_program(inputs, d_inputs, feeds, ddys, order - 1)
    return feeds, outputs, d_inputs, grads_in


def check_vjp(func, args, order=1, atol=None, rtol=None, eps=EPS):
    args = _as_list(args)
    np_type = dtype_to_np_dtype(args[0].dtype)
    atol = atol if atol else default_gradient_tolerance[np_type]
    rtol = rtol if rtol else default_gradient_tolerance[np_type]

    # shape like args, [pd.tensor, pd.tensor]
    eager_vjps, cotangents = get_eager_vjp(func, args, order=order)
    # shape like args, [np.array, np.array]
    eager_vjps_np = []
    for eager_vjp in eager_vjps:
        eager_vjps_np.append(eager_vjp.numpy())
    static_vjps_np = get_static_vjp(
        func, args, cotangents, order, atol, rtol, eps=EPS
    )
    np.testing.assert_allclose(
        eager_vjps_np,
        static_vjps_np,
        err_msg="static vjp must be equal to eager",
    )
