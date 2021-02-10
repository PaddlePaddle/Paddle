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
simple_relu2 = load(
    name='simple_jit_relu2',
    sources=['relu_op_simple.cc', 'relu_op_simple.cu'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cflags=extra_compile_args)  # add for Coverage CI


class TestJITLoad(unittest.TestCase):
    def setUp(self):
        self.custom_op = simple_relu2
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu', 'gpu']

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out = relu2_static(self.custom_op, device, dtype, x)
                pd_out = relu2_static(self.custom_op, device, dtype, x, False)
                self.assertTrue(
                    np.array_equal(out, pd_out),
                    "custom op out: {},\n paddle api out: {}".format(out,
                                                                     pd_out))

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, x_grad = relu2_dynamic(self.custom_op, device, dtype, x)
                pd_out, pd_x_grad = relu2_dynamic(self.custom_op, device, dtype,
                                                  x, False)
                self.assertTrue(
                    np.array_equal(out, pd_out),
                    "custom op out: {},\n paddle api out: {}".format(out,
                                                                     pd_out))
                self.assertTrue(
                    np.array_equal(x_grad, pd_x_grad),
                    "custom op x grad: {},\n paddle api x grad: {}".format(
                        x_grad, pd_x_grad))


if __name__ == '__main__':
    unittest.main()
