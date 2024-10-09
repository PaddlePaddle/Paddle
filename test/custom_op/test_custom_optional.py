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
from utils import check_output, extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_optional\\custom_optional.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_optional = load(
    name='custom_optional',
    sources=['custom_optional.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


def optional_dynamic_add(custom_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_y is not None:
        y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    else:
        y = x
    if custom_func:
        out = custom_optional.custom_add(x, y if np_y is not None else None)
    else:
        out = paddle.add(x, y)

    out.backward()
    return x.numpy(), out.numpy(), x.grad.numpy()


def optional_static_add(custom_func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            if np_y is not None:
                y = static.data(
                    name="y", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y.stop_gradient = False
                feed_dict = {
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                }
            else:
                y = x
                feed_dict = {
                    "x": np_x.astype(dtype),
                }
            if custom_func:
                out = custom_optional.custom_add(
                    x, y if np_y is not None else None
                )
            else:
                out = paddle.add(x, y)

            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                fetch_list = [x, out, ops[-1].result(0)]
            else:
                fetch_list = [
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                ]

            x_v, out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v


'''
if (y) {
  outX = 2 * x + y;
  outY = x + y;
} else {
  outX = 2 * x;
  outY = None;
}
'''


def optional_inplace_dynamic_add(custom_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_y is not None:
        y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=True)
        if custom_func:
            outx, outy = custom_optional.custom_optional_inplace_add(x, y)
        else:
            # We need to accumulate y's grad here.
            y.stop_gradient = False
            outx = 2 * x + y
            # Inplace leaf Tensor's stop_gradient should be True
            y.stop_gradient = True
            outy = y.add_(x)
    else:
        y = None
        if custom_func:
            outx, outy = custom_optional.custom_optional_inplace_add(x, y)
        else:
            outx = 2 * x
            outy = None
        assert (
            outy is None
        ), "The output `outy` of optional_inplace_dynamic_add should be None"

    out = outx + outy if outy is not None else outx
    out.backward()
    return (
        x.numpy(),
        outx.numpy(),
        y.numpy() if y is not None else None,
        outy.numpy() if outy is not None else None,
        out.numpy(),
        x.grad.numpy(),
        y.grad.numpy() if y is not None and y.grad is not None else None,
    )


def optional_inplace_static_add(custom_func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            if np_y is not None:
                y = static.data(
                    name="y", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y.stop_gradient = False
                feed_dict = {
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                }
                if custom_func:
                    outx, outy = custom_optional.custom_optional_inplace_add(
                        x, y
                    )
                else:
                    outx = 2 * x + y
                    outy = x + y
            else:
                feed_dict = {
                    "x": np_x.astype(dtype),
                }
                if custom_func:
                    outx, outy = custom_optional.custom_optional_inplace_add(
                        x, None
                    )
                else:
                    outx = 2 * x
                    outy = None
            out = outx + outy if outy is not None else outx
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            if np_y is not None:
                if paddle.framework.in_pir_mode():
                    ops = static.default_main_program().global_block().ops
                    if custom_func:
                        fetch_list = [
                            x,
                            out,
                            ops[-1].result(0),  # x_grad
                            ops[-1].result(1),
                        ]  # y_grad
                    else:
                        fetch_list = [
                            x,
                            out,
                            ops[-1].result(0),  # x_grad
                            ops[-3].result(0),
                        ]  # y_grad
                else:
                    fetch_list = [
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                        y.name + "@GRAD",
                    ]
                x_v, out_v, x_grad_v, y_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=fetch_list,
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v, y_grad_v]
            else:
                if paddle.framework.in_pir_mode():
                    ops = static.default_main_program().global_block().ops
                    fetch_list = [x, out, ops[-1].result(0)]

                else:
                    fetch_list = [
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                    ]
                x_v, out_v, x_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=fetch_list,
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v]


def optional_vector_dynamic_add(custom_func, device, dtype, np_x, np_inputs):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_inputs is not None:
        inputs = [
            paddle.to_tensor(np_input, dtype=dtype, stop_gradient=False)
            for np_input in np_inputs
        ]
        if custom_func:
            out = custom_optional.custom_add_vec(x, inputs)
        else:
            out = paddle.add(x, inputs[0])
            for input in inputs[1:]:
                out = paddle.add(out, input)
    else:
        if custom_func:
            out = custom_optional.custom_add_vec(x, None)
        else:
            out = paddle.add(x, x)

    out.backward()
    return x.numpy(), out.numpy(), x.grad.numpy()


def optional_vector_static_add(custom_func, device, dtype, np_x, np_inputs):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            feed_dict = {"x": np_x.astype(dtype)}
            if np_inputs is not None:
                y1 = static.data(
                    name="y1", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y1.stop_gradient = False
                y2 = static.data(
                    name="y2", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y2.stop_gradient = False
                feed_dict.update(
                    {
                        "y1": np_inputs[0].astype(dtype),
                        "y2": np_inputs[1].astype(dtype),
                    }
                )
                if custom_func:
                    out = custom_optional.custom_add_vec(x, [y1, y2])
                else:
                    out = paddle.add(x, y1)
                    out = paddle.add(out, y2)
            else:
                if custom_func:
                    out = custom_optional.custom_add_vec(x, None)
                else:
                    out = paddle.add(x, x)

            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                fetch_list = [x, out, ops[-1].result(0)]

            else:
                fetch_list = [
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                ]

            x_v, out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v


'''
if (y) {
  outX = 2 * x + y[1...n];
  outY[i] = x + y[i];
} else {
  outX = 2 * x;
  outY = None;
}
'''


def optional_inplace_vector_dynamic_add(
    custom_func, device, dtype, np_x, np_inputs
):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_inputs is not None:
        inputs = [
            paddle.to_tensor(np_input, dtype=dtype, stop_gradient=True)
            for np_input in np_inputs
        ]
        if custom_func:
            outx, outy = custom_optional.custom_optional_inplace_add_vec(
                x, inputs
            )
        else:
            outx = 2 * x
            outy = []
            for input in inputs:
                # We need to accumulate y's grad here.
                input.stop_gradient = False
                outx = outx + input
                # Inplace leaf Tensor's stop_gradient should be True
                input.stop_gradient = True
                outy.append(input.add_(x))
    else:
        if custom_func:
            outx, outy = custom_optional.custom_optional_inplace_add_vec(
                x, None
            )
        else:
            outx = 2 * x
            outy = None
        assert (
            outy is None
        ), "The output `outy` of optional_inplace_dynamic_add should be None"

    if outy is not None:
        out = outx
        for tensor in outy:
            out = out + tensor
    else:
        out = outx
    out.backward()
    return (
        x.numpy(),
        outx.numpy(),
        [y.numpy() for y in inputs] if np_inputs is not None else None,
        [t.numpy() for t in outy] if outy is not None else None,
        out.numpy(),
        x.grad.numpy(),
        (
            [y.grad.numpy() for y in inputs]
            if np_inputs is not None and inputs[0].grad is not None
            else None
        ),
    )


def optional_inplace_vector_static_add(
    custom_func, device, dtype, np_x, np_inputs
):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            feed_dict = {
                "x": np_x.astype(dtype),
            }
            if np_inputs is not None:
                y1 = static.data(
                    name="y1", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y1.stop_gradient = False
                y2 = static.data(
                    name="y2", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y2.stop_gradient = False
                feed_dict.update(
                    {
                        "y1": np_inputs[0].astype(dtype),
                        "y2": np_inputs[1].astype(dtype),
                    }
                )
                if custom_func:
                    (
                        outx,
                        outy,
                    ) = custom_optional.custom_optional_inplace_add_vec(
                        x, [y1, y2]
                    )
                else:
                    outx = paddle.add(paddle.add(paddle.add(x, x), y1), y2)
                    # outx = 2 * x + y1 + y2
                    outy = [x + y1, x + y2]
            else:
                if custom_func:
                    (
                        outx,
                        outy,
                    ) = custom_optional.custom_optional_inplace_add_vec(x, None)
                else:
                    outx = 2 * x
                    outy = None
            if np_inputs is not None:
                out = outx + outy[0] + outy[1]
            else:
                out = outx
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if np_inputs is not None:
                if paddle.framework.in_pir_mode():
                    ops = static.default_main_program().global_block().ops
                    if custom_func:
                        fetch_list = [
                            x,
                            out,
                            ops[-2].result(0),  # x_grad
                            ops[-1].result(0),  # y1_grad
                            ops[-1].result(1),
                        ]  # y2_grad
                    else:
                        fetch_list = [
                            x,
                            out,
                            ops[-1].result(0),  # x_grad
                            ops[-3].result(0),  # y1_grad
                            ops[-6].result(0),
                        ]  # y2_grad
                else:
                    fetch_list = [
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                        y1.name + "@GRAD",
                        y2.name + "@GRAD",
                    ]
                x_v, out_v, x_grad_v, y1_grad_v, y2_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=fetch_list,
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v, y1_grad_v, y2_grad_v]
            else:
                if paddle.framework.in_pir_mode():
                    ops = static.default_main_program().global_block().ops
                    fetch_list = [x, out, ops[-1].result(0)]  # y_grad
                else:
                    fetch_list = [
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                    ]
                x_v, out_v, x_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=fetch_list,
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v]


class TestCustomOptionalJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_y = np.random.random((3, 2)).astype("float32")
        self.np_inputs = [
            np.random.random((3, 2)).astype("float32"),
            np.random.random((3, 2)).astype("float32"),
        ]

    def test_optional_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (
                        pd_x,
                        pd_out,
                        pd_x_grad,
                    ) = optional_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        custom_x,
                        custom_out,
                        custom_x_grad,
                    ) = optional_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")

    def test_optional_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (
                        pd_x,
                        pd_out,
                        pd_x_grad,
                    ) = optional_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        custom_x,
                        custom_out,
                        custom_x_grad,
                    ) = optional_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")

    def test_optional_inplace_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    pd_tuple = optional_inplace_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    custom_tuple = optional_inplace_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_tuple[0], pd_tuple[0], "x")
                    check_output(custom_tuple[1], pd_tuple[1], "out")
                    check_output(custom_tuple[2], pd_tuple[2], "x_grad")
                    if len(custom_tuple) > 3:
                        check_output(custom_tuple[3], pd_tuple[3], "y_grad")

    def test_optional_inplace_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (
                        pd_x,
                        pd_outx,
                        pd_y,
                        pd_outy,
                        pd_out,
                        pd_x_grad,
                        pd_y_grad,
                    ) = optional_inplace_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        custom_x,
                        custom_outx,
                        custom_y,
                        custom_outy,
                        custom_out,
                        custom_x_grad,
                        custom_y_grad,
                    ) = optional_inplace_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(pd_y, pd_outy, "inplace_pd_y")
                    check_output(custom_y, custom_outy, "inplace_custom_y")

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_outx, pd_outx, "outx")
                    check_output(custom_y, pd_y, "y")
                    check_output(custom_outy, pd_outy, "outy")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")
                    check_output(custom_y_grad, pd_y_grad, "y_grad")

    def test_optional_vector_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    (
                        custom_x,
                        custom_out,
                        custom_x_grad,
                    ) = optional_vector_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        pd_x,
                        pd_out,
                        pd_x_grad,
                    ) = optional_vector_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")

    def test_optional_vector_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    (
                        custom_x,
                        custom_out,
                        custom_x_grad,
                    ) = optional_vector_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        pd_x,
                        pd_out,
                        pd_x_grad,
                    ) = optional_vector_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")

    def test_optional_inplace_vector_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    pd_tuple = optional_inplace_vector_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    custom_tuple = optional_inplace_vector_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(custom_tuple[0], pd_tuple[0], "x")
                    check_output(custom_tuple[1], pd_tuple[1], "out")
                    check_output(custom_tuple[2], pd_tuple[2], "x_grad")
                    if len(custom_tuple) > 3:
                        check_output(custom_tuple[3], pd_tuple[3], "y1_grad")
                        check_output(custom_tuple[4], pd_tuple[4], "y2_grad")

    def test_optional_inplace_vector_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    (
                        custom_x,
                        custom_outx,
                        custom_y,
                        custom_outy,
                        custom_out,
                        custom_x_grad,
                        custom_y_grad,
                    ) = optional_inplace_vector_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        pd_x,
                        pd_outx,
                        pd_y,
                        pd_outy,
                        pd_out,
                        pd_x_grad,
                        pd_y_grad,
                    ) = optional_inplace_vector_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    check_output(pd_y, pd_outy, "inplace_pd_y")
                    check_output(custom_y, custom_outy, "inplace_custom_y")

                    check_output(custom_x, pd_x, "x")
                    check_output(custom_outx, pd_outx, "outx")
                    check_output(custom_y, pd_y, "y")
                    check_output(custom_outy, pd_outy, "outy")
                    check_output(custom_out, pd_out, "out")
                    check_output(custom_x_grad, pd_x_grad, "x_grad")
                    check_output(custom_y_grad, pd_y_grad, "y_grad")


if __name__ == "__main__":
    unittest.main()
