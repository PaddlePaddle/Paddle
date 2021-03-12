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

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_relu_module_jit\\custom_relu_module_jit.pyd'.format(
    get_build_directory())
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


def concat_dynamic(func, device, dtype, np_inputs, axis_v):
    paddle.set_device(device)
    inputs = [
        paddle.to_tensor(
            x, dtype=dtype, place=device, stop_gradient=False)
        for x in np_inputs
    ]
    axis = paddle.full(shape=[1], dtype='int64', fill_value=axis_v)
    out = func(inputs, axis)
    out.stop_gradient = False
    out.backward()
    grad_inputs = [x.grad for x in inputs]
    return out.numpy(), grad_inputs


def concat_static(func, device, dtype, np_inputs, axis_v):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x1 = static.data(name="x1", shape=[2, 3], dtype=dtype)
            x2 = static.data(name="x2", shape=[2, 3], dtype=dtype)
            axis = paddle.full(shape=[1], dtype='int64', fill_value=axis_v)
            x1.stop_gradient = False
            x2.stop_gradient = False
            out = func([x1, x2], axis)
            # mean only support float, so here use sum
            sum_out = paddle.sum(out)
            static.append_backward(sum_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            out_v, x1_grad_v, x2_grad_v = exe.run(
                static.default_main_program(),
                feed={
                    "x1": np_inputs[0].astype(dtype),
                    "x2": np_inputs[1].astype(dtype),
                    "axis": axis
                },
                fetch_list=[out.name, x1.name + "@GRAD", x2.name + "@GRAD"])
    paddle.disable_static()
    return out_v, x1_grad_v, x2_grad_v


class TestCustomConcatDynamicAxisJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64', 'int32', 'int64']
        self.devices = ['cpu']
        self.np_inputs = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[11, 12, 13], [14, 15, 16]])
        ]
        self.axises = [0, 1]

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for axis in self.axises:
                    out, grad_inputs = concat_dynamic(custom_ops.custom_concat,
                                                      device, dtype,
                                                      self.np_inputs, axis)
                    pd_out, pd_grad_inputs = concat_dynamic(
                        paddle.concat, device, dtype, self.np_inputs, axis)

                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    for x_grad, pd_x_grad in zip(grad_inputs, pd_grad_inputs):
                        self.assertTrue(
                            np.array_equal(x_grad, pd_x_grad),
                            "custom op x grad: {},\n paddle api x grad: {}".
                            format(x_grad, pd_x_grad))

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for axis in self.axises:
                    out, x1_grad, x2_grad = concat_static(
                        custom_ops.custom_concat, device, dtype, self.np_inputs,
                        axis)
                    pd_out, pd_x1_grad, pd_x2_grad = concat_static(
                        paddle.concat, device, dtype, self.np_inputs, axis)

                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    self.assertTrue(
                        np.array_equal(x1_grad, pd_x1_grad),
                        "custom op x1_grad: {},\n paddle api x1_grad: {}".
                        format(x1_grad, pd_x1_grad))
                    self.assertTrue(
                        np.array_equal(x2_grad, pd_x2_grad),
                        "custom op x2_grad: {},\n paddle api x2_grad: {}".
                        format(x2_grad, pd_x2_grad))


if __name__ == "__main__":
    unittest.main()
