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
import paddle.nn.functional as F
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.fluid.framework import _test_eager_guard

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_linear\\custom_linear.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_linear_jit',
    sources=['custom_linear_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


def linear_dynamic(func, device, dtype, np_x, np_weight, np_bias):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    weight = paddle.to_tensor(np_weight, dtype=dtype, stop_gradient=False)
    bias = paddle.to_tensor(np_bias, dtype=dtype, stop_gradient=False)
    out = func(x, weight, bias)
    out.backward()
    return out.numpy(), x.grad.numpy(), weight.grad.numpy(), bias.grad.numpy()


def linear_static(func, device, dtype, np_x, np_weight, np_bias):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            weight = static.data(name="weight",
                                 shape=np_weight.shape,
                                 dtype=dtype)
            bias = static.data(name="bias", shape=np_bias.shape, dtype=dtype)
            x.stop_gradient = False
            weight.stop_gradient = False
            bias.stop_gradient = False
            out = func(x, weight, bias)
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            out_v, x_grad_v, weight_grad_v, bias_grad_v = exe.run(
                static.default_main_program(),
                feed={
                    "x": np_x.astype(dtype),
                    "weight": np_weight.astype(dtype),
                    "bias": np_bias.astype(dtype)
                },
                fetch_list=[
                    out.name, x.name + "@GRAD", weight.name + "@GRAD",
                    bias.name + "@GRAD"
                ])
    paddle.disable_static()
    return out_v, x_grad_v, weight_grad_v, bias_grad_v


class TestCustomLinearJit(unittest.TestCase):

    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_weight = np.full([2, 4], fill_value=0.5, dtype="float32")
        self.np_bias = np.ones([4], dtype="float32")

    def check_output(self, out, pd_out, name):
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out))

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                phi_out, phi_x_grad, phi_weight_grad, phi_bias_grad = linear_static(
                    custom_ops.phi_linear, device, dtype, self.np_x,
                    self.np_weight, self.np_bias)
                pd_out, pd_x_grad, pd_weight_grad, pd_bias_grad = linear_static(
                    F.linear, device, dtype, self.np_x, self.np_weight,
                    self.np_bias)
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")
                self.check_output(phi_weight_grad, pd_weight_grad,
                                  "weight_grad")
                self.check_output(phi_bias_grad, pd_bias_grad, "bias_grad")

    def func_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                phi_out, phi_x_grad, phi_weight_grad, phi_bias_grad = linear_dynamic(
                    custom_ops.phi_linear, device, dtype, self.np_x,
                    self.np_weight, self.np_bias)
                pd_out, pd_x_grad, pd_weight_grad, pd_bias_grad = linear_dynamic(
                    F.linear, device, dtype, self.np_x, self.np_weight,
                    self.np_bias)
                self.check_output(phi_out, pd_out, "phi_out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")
                self.check_output(phi_weight_grad, pd_weight_grad,
                                  "weight_grad")
                self.check_output(phi_bias_grad, pd_bias_grad, "bias_grad")

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()


if __name__ == "__main__":
    unittest.main()
