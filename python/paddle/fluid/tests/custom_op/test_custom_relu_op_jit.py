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
import subprocess
import unittest
import paddle
import numpy as np
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_compile_args
from test_custom_relu_op_setup import custom_relu_dynamic, custom_relu_static

# Because Windows don't use docker, the shared lib already exists in the 
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_relu_module_jit\\custom_relu_module_jit.pyd'.format(
    get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
# custom_relu_op_dup.cc is only used for multi ops test,
# not a new op, if you want to test only one op, remove this
# source file
custom_module = load(
    name='custom_relu_module_jit',
    sources=[
        'custom_relu_op.cc', 'custom_relu_op.cu', 'custom_relu_op_dup.cc'
    ],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_compile_args,  # add for Coverage CI
    extra_cuda_cflags=extra_compile_args,  # add for Coverage CI
    verbose=True)


class TestJITLoad(unittest.TestCase):
    def setUp(self):
        self.custom_ops = [
            custom_module.custom_relu, custom_module.custom_relu_dup
        ]
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu', 'gpu']

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = custom_relu_static(custom_op, device, dtype, x)
                    pd_out = custom_relu_static(custom_op, device, dtype, x,
                                                False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out, x_grad = custom_relu_dynamic(custom_op, device, dtype,
                                                      x)
                    pd_out, pd_x_grad = custom_relu_dynamic(custom_op, device,
                                                            dtype, x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    self.assertTrue(
                        np.array_equal(x_grad, pd_x_grad),
                        "custom op x grad: {},\n paddle api x grad: {}".format(
                            x_grad, pd_x_grad))


if __name__ == '__main__':
    unittest.main()
