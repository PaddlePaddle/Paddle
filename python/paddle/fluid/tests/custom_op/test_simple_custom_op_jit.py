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
import paddle
import numpy as np
from paddle.utils.cpp_extension import load
from utils import paddle_includes, extra_compile_args
from test_simple_custom_op_setup import relu2_dynamic, relu2_static

# Compile and load custom op Just-In-Time.
custom_module = load(
    name='simple_jit_relu2',
    sources=['relu_op_simple.cc', 'relu_op_simple.cu', 'relu_op3_simple.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cflags=extra_compile_args)  # add for Coverage CI


class TestJITLoad(unittest.TestCase):
    def setUp(self):
        self.custom_ops = [custom_module.relu2, custom_module.relu3]
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu', 'gpu']

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = relu2_static(custom_op, device, dtype, x)
                    pd_out = relu2_static(custom_op, device, dtype, x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out, x_grad = relu2_dynamic(custom_op, device, dtype, x)
                    pd_out, pd_x_grad = relu2_dynamic(custom_op, device, dtype,
                                                      x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    self.assertTrue(
                        np.array_equal(x_grad, pd_x_grad),
                        "custom op x grad: {},\n paddle api x grad: {}".format(
                            x_grad, pd_x_grad))


class TestMultiOutputDtypes(unittest.TestCase):
    def setUp(self):
        self.custom_op = custom_module.relu2
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu', 'gpu']

    def test_static(self):
        paddle.enable_static()
        for device in self.devices:
            for dtype in self.dtypes:
                res = self.run_static(device, dtype)
                self.check_multi_outputs(res)
        paddle.disable_static()

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                paddle.set_device(device)
                x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                x = paddle.to_tensor(x_data)
                outs = self.custom_op(x)

                self.assertTrue(len(outs) == 3)
                self.check_multi_outputs(outs, True)

    def check_multi_outputs(self, outs, is_dynamic=False):
        out, zero_float64, one_int32 = outs
        if is_dynamic:
            zero_float64 = zero_float64.numpy()
            one_int32 = one_int32.numpy()
        # Fake_float64
        self.assertTrue('float64' in str(zero_float64.dtype))
        self.assertTrue(
            np.array_equal(zero_float64, np.zeros([4, 8]).astype('float64')))
        # ZFake_int32
        self.assertTrue('int32' in str(one_int32.dtype))
        self.assertTrue(
            np.array_equal(one_int32, np.ones([4, 8]).astype('int32')))

    def run_static(self, device, dtype):
        paddle.set_device(device)
        x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)

        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='X', shape=[None, 8], dtype=dtype)
                outs = self.custom_op(x)

                exe = paddle.static.Executor()
                exe.run(paddle.static.default_startup_program())
                res = exe.run(paddle.static.default_main_program(),
                              feed={'X': x_data},
                              fetch_list=outs)

                return res


if __name__ == '__main__':
    unittest.main()
