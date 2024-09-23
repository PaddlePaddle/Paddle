# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from utils import (
    check_output,
    check_output_allclose,
    extra_cc_args,
    extra_nvcc_args,
    paddle_includes,
)

import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_inplace\\custom_inplace.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_inplace = load(
    name='custom_inplace',
    sources=['custom_inplace.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


def inplace_dynamic_add(custom_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=True)
    y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    if custom_func:
        out = custom_inplace.custom_add(x, y)
    else:
        out = x.add_(y)

    out.backward()
    return x.numpy(), y.numpy(), out.numpy(), x.grad.numpy(), y.grad.numpy()


def inplace_static_add(func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            y = static.data(name="y", shape=[None, np_y.shape[1]], dtype=dtype)
            x.stop_gradient = False
            y.stop_gradient = False
            out = func(x, y)
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                fetch_list = [
                    x,
                    out,
                    ops[-1].result(0),
                    ops[-1].result(1),
                    ops[-2].result(0),
                ]
            else:
                fetch_list = [
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                    y.name + "@GRAD",
                    out.name + "@GRAD",
                ]

            x_v, out_v, x_grad_v, y_grad_v, out_grad_v = exe.run(
                static.default_main_program(),
                feed={
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                },
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v, y_grad_v, out_grad_v


def inplace_dynamic_add_vector(custom_func, device, dtype, np_inputs, np_y):
    paddle.set_device(device)
    inputs = [
        paddle.to_tensor(np_input, dtype=dtype, stop_gradient=True)
        for np_input in np_inputs
    ]
    y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    if custom_func:
        out = custom_inplace.custom_add_vec(inputs, y)
    else:
        out = [x.add_(y) for x in inputs]

    mean_out = paddle.mean(paddle.concat(out))
    mean_out.backward()
    return (
        np.concatenate([input.numpy() for input in inputs]),
        y.numpy(),
        np.concatenate([o.numpy() for o in out]),
        np.concatenate([input.grad.numpy() for input in inputs]),
        y.grad.numpy(),
    )


def inplace_static_add_vector(custom_func, device, dtype, np_inputs, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x1 = static.data(
                name="x1", shape=[None, np_inputs[0].shape[1]], dtype=dtype
            )
            x2 = static.data(
                name="x2", shape=[None, np_inputs[1].shape[1]], dtype=dtype
            )
            y = static.data(name="y", shape=[None, np_y.shape[1]], dtype=dtype)
            x1.stop_gradient = False
            x2.stop_gradient = False
            y.stop_gradient = False
            if custom_func:
                out = custom_inplace.custom_add_vec([x1, x2], y)
            else:
                out = [paddle.add(x1, y), paddle.add(x2, y)]
            mean_out = paddle.mean(paddle.concat(out))
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                if custom_func:
                    fetch_list = [
                        out[0],
                        out[1],
                        ops[-1].result(0),  # x1_grad
                        ops[-1].result(1),  # x2_grad
                        ops[-2].result(1),  # y_grad
                        ops[-5].result(0),  # out0_grad
                        ops[-5].result(1),
                    ]  # out1_grad
                else:
                    fetch_list = [
                        out[0],
                        out[1],
                        ops[-4].result(0),  # x1_grad
                        ops[-3].result(0),  # x2_grad
                        ops[-1].result(0),  # y_grad
                        ops[-5].result(0),  # out0_grad
                        ops[-5].result(1),
                    ]  # out1_grad
            else:
                fetch_list = [
                    out[0].name,
                    out[1].name,
                    x1.name + "@GRAD",
                    x2.name + "@GRAD",
                    y.name + "@GRAD",
                    out[0].name + "@GRAD",
                    out[1].name + "@GRAD",
                ]

            (
                out0_v,
                out1_v,
                x1_grad_v,
                x2_grad_v,
                y_grad_v,
                out0_grad_v,
                out1_grad_v,
            ) = exe.run(
                static.default_main_program(),
                feed={
                    "x1": np_inputs[0].astype(dtype),
                    "x2": np_inputs[1].astype(dtype),
                    "y": np_y.astype(dtype),
                },
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return (
        [out0_v, out1_v],
        [x1_grad_v, x2_grad_v],
        y_grad_v,
        [out0_grad_v, out1_grad_v],
    )


def inplace_dynamic_relu_net(custom_func, device, dtype, np_x, np_y, np_z):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    z = paddle.to_tensor(np_z, dtype=dtype, stop_gradient=False)
    out_xy = x + y
    if custom_func:
        out_xy = custom_inplace.custom_relu_inplace(out_xy)
        out_xyz = out_xy + z
        out = custom_inplace.custom_relu_inplace(out_xyz)
    else:
        out_xy = paddle.nn.functional.relu_(out_xy)
        out_xyz = out_xy + z
        out = paddle.nn.functional.relu_(out_xyz)

    out.backward()
    return x.numpy(), y.numpy(), out.numpy(), x.grad.numpy(), y.grad.numpy()


def inplace_static_relu_net(func, device, dtype, np_x, np_y, np_z):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            y = static.data(name="y", shape=[None, np_y.shape[1]], dtype=dtype)
            z = static.data(name="z", shape=[None, np_z.shape[1]], dtype=dtype)
            x.stop_gradient = False
            y.stop_gradient = False
            z.stop_gradient = False
            out_xy = x + y
            out_xy = func(out_xy)
            out_xyz = out_xy + z
            out = func(out_xyz)
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                fetch_list = [
                    x,
                    y,
                    out,
                    ops[-1].result(0),  # x_grad
                    ops[-1].result(1),
                ]  # y_grad
            else:
                fetch_list = [
                    x.name,
                    y.name,
                    out.name,
                    x.name + "@GRAD",
                    y.name + "@GRAD",
                ]

            x_v, y_v, out_v, x_grad_v, y_grad_v = exe.run(
                static.default_main_program(),
                feed={
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                    "z": np_z.astype(dtype),
                },
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return x_v, y_v, out_v, x_grad_v, y_grad_v


def dynamic_multi_inplace(custom_func, device, dtype, np_x, np_y, np_a, np_b):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=True)
    y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    a = paddle.to_tensor(np_a, dtype=dtype, stop_gradient=True)
    b = paddle.to_tensor(np_b, dtype=dtype, stop_gradient=False)
    if custom_func:
        out_xy, out_ab = custom_inplace.custom_multi_inplace(x, y, a, b)
    else:
        out_xy = x.add_(y)
        out_ab = a.add_(b)
    out = out_xy + out_ab

    out.backward()
    return (
        x.numpy(),
        y.numpy(),
        out_xy.numpy(),
        x.grad.numpy(),
        y.grad.numpy(),
        a.numpy(),
        b.numpy(),
        out_ab.numpy(),
        a.grad.numpy(),
        b.grad.numpy(),
    )


def static_multi_inplace(custom_func, device, dtype, np_x, np_y, np_a, np_b):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            y = static.data(name="y", shape=[None, np_y.shape[1]], dtype=dtype)
            a = static.data(name="a", shape=[None, np_x.shape[1]], dtype=dtype)
            b = static.data(name="b", shape=[None, np_y.shape[1]], dtype=dtype)
            x.stop_gradient = False
            y.stop_gradient = False
            a.stop_gradient = False
            b.stop_gradient = False
            if custom_func:
                out_xy, out_ab = custom_inplace.custom_multi_inplace(x, y, a, b)
            else:
                out_xy = paddle.add(x, y)
                out_ab = paddle.add(a, b)
            mean_out = paddle.mean(paddle.add(out_xy, out_ab))
            static.append_backward(mean_out)

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                if custom_func:
                    fetch_list = [
                        x,
                        out_xy,
                        ops[-1].result(0),  # x_grad
                        ops[-1].result(1),  # y_grad
                        ops[-2].result(0),  # out_xy_grad
                        a,
                        out_ab,
                        ops[-1].result(2),  # a_grad
                        ops[-1].result(3),  # b_grad
                        ops[-2].result(1),
                    ]  # out_ab_grad
                else:
                    fetch_list = [
                        x,
                        out_xy,
                        ops[-2].result(0),  # x_grad
                        ops[-2].result(1),  # y_grad
                        ops[-3].result(0),  # out_xy_grad
                        a,
                        out_ab,
                        ops[-1].result(0),  # a_grad
                        ops[-1].result(1),  # b_grad
                        ops[-3].result(1),
                    ]  # out_ab_grad

            else:
                fetch_list = [
                    x.name,
                    out_xy.name,
                    x.name + "@GRAD",
                    y.name + "@GRAD",
                    out_xy.name + "@GRAD",
                    a.name,
                    out_ab.name,
                    a.name + "@GRAD",
                    b.name + "@GRAD",
                    out_ab.name + "@GRAD",
                ]

            exe = static.Executor()
            exe.run(static.default_startup_program())

            (
                x_v,
                out_xy_v,
                x_grad_v,
                y_grad_v,
                out_xy_grad_v,
                a_v,
                out_ab_v,
                a_grad_v,
                b_grad_v,
                out_ab_grad_v,
            ) = exe.run(
                static.default_main_program(),
                feed={
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                    "a": np_a.astype(dtype),
                    "b": np_b.astype(dtype),
                },
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return (
        x_v,
        out_xy_v,
        x_grad_v,
        y_grad_v,
        out_xy_grad_v,
        a_v,
        out_ab_v,
        a_grad_v,
        b_grad_v,
        out_ab_grad_v,
    )


class TestCustomInplaceJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_y = np.random.random((3, 2)).astype("float32")
        self.np_z = np.random.random((3, 2)).astype("float32")
        self.np_a = np.random.random((3, 2)).astype("float32")
        self.np_b = np.random.random((3, 2)).astype("float32")
        self.np_inputs = [
            np.random.random((3, 2)).astype("float32"),
            np.random.random((3, 2)).astype("float32"),
        ]

    def test_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                    pd_out_grad,
                ) = inplace_static_add(
                    paddle.add,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                (
                    custom_x,
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                    custom_out_grad,
                ) = inplace_static_add(
                    custom_inplace.custom_add,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                check_output(custom_x, custom_out, "inplace_custom_x")
                check_output(
                    custom_x_grad, custom_out_grad, "inplace_custom_x_grad"
                )

                check_output(custom_out, pd_out, "out")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")
                check_output(custom_out_grad, pd_out_grad, "out_grad")

    def test_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_y,
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                ) = inplace_dynamic_add(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                (
                    custom_x,
                    custom_y,
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                ) = inplace_dynamic_add(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )

                check_output(custom_x, custom_out, "inplace_custom_x")
                check_output(pd_x, pd_out, "inplace_pd_x")

                check_output(custom_x, pd_x, "x")
                check_output(custom_y, pd_y, "y")
                check_output(custom_out, pd_out, "out")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")

    def test_static_add_vector(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                    pd_out_grad,
                ) = inplace_static_add_vector(
                    True,
                    device,
                    dtype,
                    self.np_inputs,
                    self.np_y,
                )
                (
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                    custom_out_grad,
                ) = inplace_static_add_vector(
                    False,
                    device,
                    dtype,
                    self.np_inputs,
                    self.np_y,
                )

                check_output(custom_out, pd_out, "out")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")
                check_output(custom_out_grad, pd_out_grad, "out_grad")

    def test_dynamic_add_vector(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_y,
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                ) = inplace_dynamic_add_vector(
                    True,
                    device,
                    dtype,
                    self.np_inputs,
                    self.np_y,
                )
                (
                    custom_x,
                    custom_y,
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                ) = inplace_dynamic_add_vector(
                    False,
                    device,
                    dtype,
                    self.np_inputs,
                    self.np_y,
                )

                check_output(custom_x, custom_out, "inplace_custom_x")
                check_output(pd_x, pd_out, "inplace_pd_x")

                check_output(custom_x, pd_x, "x")
                check_output(custom_y, pd_y, "y")
                check_output(custom_out, pd_out, "out")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")

    def test_static_relu_net(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_y,
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                ) = inplace_static_relu_net(
                    paddle.nn.functional.relu,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                (
                    custom_x,
                    custom_y,
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                ) = inplace_static_relu_net(
                    custom_inplace.custom_relu_inplace,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                check_output_allclose(custom_x, pd_x, "x")
                check_output_allclose(custom_y, pd_y, "y")
                check_output_allclose(custom_out, pd_out, "out")
                check_output_allclose(custom_x_grad, pd_x_grad, "x_grad")
                check_output_allclose(custom_y_grad, pd_y_grad, "y_grad")

    def test_dynamic_relu_net(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_y,
                    pd_out,
                    pd_x_grad,
                    pd_y_grad,
                ) = inplace_dynamic_relu_net(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                (
                    custom_x,
                    custom_y,
                    custom_out,
                    custom_x_grad,
                    custom_y_grad,
                ) = inplace_dynamic_relu_net(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )

                check_output(custom_x, pd_x, "x")
                check_output(custom_y, pd_y, "y")
                check_output(custom_out, pd_out, "out")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")

    def test_static_multi_inplace(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_out_xy,
                    pd_x_grad,
                    pd_y_grad,
                    pd_out_xy_grad,
                    pd_a,
                    pd_out_ab,
                    pd_a_grad,
                    pd_b_grad,
                    pd_out_ab_grad,
                ) = static_multi_inplace(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_a,
                    self.np_b,
                )
                (
                    custom_x,
                    custom_out_xy,
                    custom_x_grad,
                    custom_y_grad,
                    custom_out_xy_grad,
                    custom_a,
                    custom_out_ab,
                    custom_a_grad,
                    custom_b_grad,
                    custom_out_ab_grad,
                ) = static_multi_inplace(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_a,
                    self.np_b,
                )
                check_output(custom_x, pd_out_xy, "inplace_custom_x")
                check_output(
                    custom_x_grad, custom_out_xy_grad, "inplace_custom_x_grad"
                )
                check_output(custom_a, pd_out_ab, "inplace_custom_a")
                check_output(
                    custom_a_grad, custom_out_ab_grad, "inplace_custom_a_grad"
                )

                check_output(custom_out_xy, pd_out_xy, "outxy")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")
                check_output(custom_out_xy_grad, pd_out_xy_grad, "outxy_grad")
                check_output(custom_out_ab, pd_out_ab, "outab")
                check_output(custom_a_grad, pd_a_grad, "a_grad")
                check_output(custom_b_grad, pd_b_grad, "b_grad")
                check_output(custom_out_ab_grad, pd_out_ab_grad, "outab_grad")

    def test_dynamic_multi_inplace(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (
                    pd_x,
                    pd_y,
                    pd_out_xy,
                    pd_x_grad,
                    pd_y_grad,
                    pd_a,
                    pd_b,
                    pd_out_ab,
                    pd_a_grad,
                    pd_b_grad,
                ) = dynamic_multi_inplace(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_a,
                    self.np_b,
                )
                (
                    custom_x,
                    custom_y,
                    custom_out_xy,
                    custom_x_grad,
                    custom_y_grad,
                    custom_a,
                    custom_b,
                    custom_out_ab,
                    custom_a_grad,
                    custom_b_grad,
                ) = dynamic_multi_inplace(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                    self.np_a,
                    self.np_b,
                )

                check_output(custom_x, custom_out_xy, "inplace_custom_x")
                check_output(pd_x, pd_out_xy, "inplace_pd_x")
                check_output(custom_a, custom_out_ab, "inplace_custom_a")
                check_output(pd_a, pd_out_ab, "inplace_pd_a")

                check_output(custom_x, pd_x, "x")
                check_output(custom_y, pd_y, "y")
                check_output(custom_out_xy, pd_out_xy, "outxy")
                check_output(custom_x_grad, pd_x_grad, "x_grad")
                check_output(custom_y_grad, pd_y_grad, "y_grad")
                check_output(custom_a, pd_a, "a")
                check_output(custom_b, pd_b, "b")
                check_output(custom_out_ab, pd_out_ab, "outab")
                check_output(custom_a_grad, pd_a_grad, "a_grad")
                check_output(custom_b_grad, pd_b_grad, "b_grad")


if __name__ == "__main__":
    unittest.main()
