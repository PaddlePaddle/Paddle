# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.fluid.framework import _test_eager_guard

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_concat\\custom_concat.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

if os.name == 'nt':
    test_include = "..\\python\\paddle\\fluid\\tests\\custom_op"
else:
    test_include = "../python/paddle/fluid/tests/custom_op"
paddle_includes.append(test_include)

custom_ops = load(
    name='custom_concat_jit',
    sources=['custom_concat_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


def concat_dynamic(func, dtype, np_inputs, axis_v, with_attr=False):
    paddle.set_device("cpu")
    inputs = [
        paddle.to_tensor(
            x, dtype=dtype, stop_gradient=False) for x in np_inputs
    ]
    if with_attr:
        axis = axis_v
    else:
        axis = paddle.full(shape=[1], dtype='int64', fill_value=axis_v)
    out = func(inputs, axis)
    out.stop_gradient = False
    out.backward()
    grad_inputs = [x.grad.numpy() for x in inputs]
    return out.numpy(), grad_inputs


def concat_static(func, dtype, np_inputs, axis_v, with_attr=False):
    paddle.enable_static()
    paddle.set_device("cpu")
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x1 = static.data(name="x1", shape=[2, 3], dtype=dtype)
            x2 = static.data(name="x2", shape=[2, 3], dtype=dtype)
            if with_attr:
                axis = axis_v
            else:
                axis = paddle.full(shape=[1], dtype='int64', fill_value=axis_v)
            x1.stop_gradient = False
            x2.stop_gradient = False
            out = func([x1, x2], axis)
            # mean only support float, so here use sum
            sum_out = paddle.sum(out)
            static.append_backward(sum_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if with_attr:
                feed_dict = {
                    "x1": np_inputs[0].astype(dtype),
                    "x2": np_inputs[1].astype(dtype)
                }
            else:
                feed_dict = {
                    "x1": np_inputs[0].astype(dtype),
                    "x2": np_inputs[1].astype(dtype),
                    "axis": axis
                }
            out_v, x1_grad_v, x2_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=[out.name, x1.name + "@GRAD", x2.name + "@GRAD"])
    paddle.disable_static()
    return out_v, x1_grad_v, x2_grad_v


class TestCustomConcatDynamicAxisJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64', 'int32', 'int64']
        self.np_inputs = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[11, 12, 13], [14, 15, 16]])
        ]
        self.axises = [0, 1]

    def check_output(self, out, pd_out, name):
        self.assertTrue(
            np.array_equal(out, pd_out),
            "custom op {}: {},\n paddle api {}: {}".format(name, out, name,
                                                           pd_out))

    def func_dynamic(self):
        for dtype in self.dtypes:
            for axis in self.axises:
                out, grad_inputs = concat_dynamic(custom_ops.custom_concat,
                                                  dtype, self.np_inputs, axis)
                pd_out, pd_grad_inputs = concat_dynamic(paddle.concat, dtype,
                                                        self.np_inputs, axis)

                self.check_output(out, pd_out, "out")
                for x_grad, pd_x_grad in zip(grad_inputs, pd_grad_inputs):
                    self.check_output(x_grad, pd_x_grad, "x_grad")

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()

    def test_static(self):
        for dtype in self.dtypes:
            for axis in self.axises:
                out, x1_grad, x2_grad = concat_static(
                    custom_ops.custom_concat, dtype, self.np_inputs, axis)
                pd_out, pd_x1_grad, pd_x2_grad = concat_static(
                    paddle.concat, dtype, self.np_inputs, axis)

                self.check_output(out, pd_out, "out")
                self.check_output(x1_grad, pd_x1_grad, "x1_grad")
                self.check_output(x2_grad, pd_x2_grad, "x2_grad")

    def func_dynamic_with_attr(self):
        for dtype in self.dtypes:
            for axis in self.axises:
                out, grad_inputs = concat_dynamic(
                    custom_ops.custom_concat_with_attr, dtype, self.np_inputs,
                    axis, True)
                pd_out, pd_grad_inputs = concat_dynamic(
                    paddle.concat, dtype, self.np_inputs, axis, True)

                self.check_output(out, pd_out, "out")
                for x_grad, pd_x_grad in zip(grad_inputs, pd_grad_inputs):
                    self.check_output(x_grad, pd_x_grad, "x_grad")

    def test_dynamic_with_attr(self):
        with _test_eager_guard():
            self.func_dynamic_with_attr()
        self.func_dynamic_with_attr()

    def test_static_with_attr(self):
        for dtype in self.dtypes:
            for axis in self.axises:
                out, x1_grad, x2_grad = concat_static(
                    custom_ops.custom_concat_with_attr, dtype, self.np_inputs,
                    axis, True)
                pd_out, pd_x1_grad, pd_x2_grad = concat_static(
                    paddle.concat, dtype, self.np_inputs, axis, True)

                self.check_output(out, pd_out, "out")
                self.check_output(x1_grad, pd_x1_grad, "x1_grad")
                self.check_output(x2_grad, pd_x2_grad, "x2_grad")


if __name__ == "__main__":
    unittest.main()
