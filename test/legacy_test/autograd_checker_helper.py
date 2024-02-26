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

import numpy as np

import paddle
from paddle.autograd.backward_utils import ValueDict
from paddle.base import core
from paddle.base.backward import _as_list

__all__ = ['check_vjp']

EPS = 1e-4

default_gradient_tolerance = {
    np.dtype(np.float16): 1e-2,
    np.dtype(np.float32): 2e-3,
    np.dtype(np.float64): 1e-5,
    np.dtype(np.complex64): 1e-3,
    np.dtype(np.complex128): 1e-5,
}


def check_close(
    eager_vjps_np,
    static_vjps_np,
    numerical_jvps_np,
    tangents,
    cotangents,
    atol,
    rtol,
):
    eager_vjps_np = eager_vjps_np[0].reshape([-1])
    static_vjps_np = static_vjps_np[0].reshape([-1])
    numerical_jvps_np = numerical_jvps_np[0].reshape([-1])
    tangents = tangents[0].reshape([-1])
    cotangents = cotangents[0].reshape([-1])
    dtype = eager_vjps_np.dtype
    atol = atol if atol else default_gradient_tolerance[dtype]
    rtol = rtol if rtol else default_gradient_tolerance[dtype]
    np.testing.assert_allclose(
        eager_vjps_np,
        static_vjps_np,
        atol=atol,
        rtol=rtol,
        err_msg="The grad of the eager is not equal to the static",
    )
    inner_dot_ret = np.dot(eager_vjps_np, tangents)
    inner_dot_ret_expect = np.dot(cotangents, numerical_jvps_np)
    np.testing.assert_allclose(
        inner_dot_ret,
        inner_dot_ret_expect,
        rtol=rtol,
        err_msg="The grad of analytical is not equal to the numerical",
    )


# Use the existing vjp method to calculate n order, provided that n-1 order is correct
def f_vjp_wrapper(func):
    def inner_wrapper(args):
        out, vjp_result = paddle.incubate.autograd.vjp(func, args)
        return vjp_result

    return inner_wrapper


def derivative(func, primals, order=1):
    if order > 1:
        f_vjp = f_vjp_wrapper(func)
        outputs = f_vjp(primals)
        return derivative(f_vjp, _as_list(outputs), order - 1)
    return func(primals)


def derivate_wrapper(func, order=1):
    def inner_wrapper(primals):
        return derivative(func, primals, order=order)

    return inner_wrapper


def numerical_jvp(func, primals, tangents, order=1, eps=EPS):
    func = derivate_wrapper(func, order)
    delta = []
    for value in tangents:
        v = paddle.scale(value, eps)
        delta.append(v)
    f_pos = func(list(map(paddle.add, primals, delta)))
    f_pos = _as_list(f_pos)
    f_neg = func(list(map(paddle.subtract, primals, delta)))
    f_neg = _as_list(f_neg)
    sub_deltas = list(map(paddle.subtract, f_pos, f_neg))
    jvp_rets = []
    for sub_delta in sub_deltas:
        jvp_rets.append(paddle.scale(sub_delta, 0.5 / eps))
    return jvp_rets


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
    create_graph = True
    allow_unused = True

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


def get_static_vjp(func, inputs, tangents, order=1):
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
    for output in outputs:
        pir_outputs.append(op_map[output])
    with paddle.static.program_guard(program):
        feeds, pre_outputs, d_inputs = _get_static_vjp(
            pir_inputs, pir_outputs, feeds, tangents, order
        )
        exe = paddle.static.Executor()
        res = exe.run(program, feed=feeds, fetch_list=[d_inputs])
    print(program)
    paddle.disable_static()

    return res


def _get_static_vjp(inputs, outputs, feeds, tangents, order):
    def _require_grads(vars):
        for var in vars:
            var.stop_gradient = False
            var.persistable = True

    inputs = _as_list(inputs)
    outputs = _as_list(outputs)
    tangents = _as_list(tangents)
    _require_grads(inputs)
    _require_grads(outputs)
    y_grads = []
    for idx, output in enumerate(outputs):
        output.persistable = True
        np_type = dtype_to_np_dtype(output.dtype)
        dy = paddle.static.data(
            name=f'tangents_{order}_{idx}',
            shape=output.shape,
            dtype=np_type,
        )
        feeds.update({f'tangents_{order}_{idx}': tangents[idx]})
        y_grads.append(dy)
    _require_grads(y_grads)
    d_inputs = paddle.base.gradients(outputs, inputs, y_grads)
    d_inputs = [d_input for d_input in d_inputs if d_input is not None]
    _require_grads(d_inputs)

    if order > 1:
        ddys = []
        for d_input in d_inputs:
            ddy = np.ones(d_input.shape, dtype=dtype_to_np_dtype(d_input.dtype))
            ddys.append(ddy)
        return _get_static_vjp(inputs, d_inputs, feeds, ddys, order - 1)
    return feeds, outputs, d_inputs


def check_vjp(func, args, order=2, atol=None, rtol=None, eps=EPS):
    args = _as_list(args)
    tangents = []
    for arg in args:
        t = paddle.randn(arg.shape, arg.dtype)
        tangents.append(t)

    # shape like args, [pd.tensor, pd.tensor]
    eager_vjps, cotangents = get_eager_vjp(func, args, order=order)
    # shape like args, [np.array, np.array]
    eager_vjps_np = []
    for eager_vjp in eager_vjps:
        eager_vjps_np.append(eager_vjp.numpy())
    static_vjps_np = get_static_vjp(func, args, cotangents, order=order)
    numerical_jvps = numerical_jvp(func, args, tangents, order=order, eps=eps)
    numerical_jvps_np = []
    for num_jvp in numerical_jvps:
        numerical_jvps_np.append(num_jvp.numpy())
    check_close(
        eager_vjps_np,
        static_vjps_np,
        numerical_jvps_np,
        tangents,
        cotangents,
        atol,
        rtol,
    )
