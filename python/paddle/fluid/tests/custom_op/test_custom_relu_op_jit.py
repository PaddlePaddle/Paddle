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
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args, IS_WINDOWS, IS_MAC
from test_custom_relu_op_setup import custom_relu_dynamic, custom_relu_static
from paddle.fluid.framework import _test_eager_guard
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
sources = ['custom_relu_op.cc', 'custom_relu_op_dup.cc']
if not IS_MAC:
    sources.append('custom_relu_op.cu')

custom_module = load(
    name='custom_relu_module_jit',
    sources=sources,
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


class TestJITLoad(unittest.TestCase):

    def setUp(self):
        self.custom_ops = [
            custom_module.custom_relu, custom_module.custom_relu_dup,
            custom_module.custom_relu_no_x_in_backward,
            custom_module.custom_relu_out
        ]
        self.dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            self.dtypes.append('float16')
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = custom_relu_static(custom_op, device, dtype, x)
                    pd_out = custom_relu_static(custom_op, device, dtype, x,
                                                False)
                    np.testing.assert_array_equal(
                        out,
                        pd_out,
                        err_msg='custom op out: {},\n paddle api out: {}'.
                        format(out, pd_out))

    def func_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out, x_grad = custom_relu_dynamic(custom_op, device, dtype,
                                                      x)
                    pd_out, pd_x_grad = custom_relu_dynamic(
                        custom_op, device, dtype, x, False)
                    np.testing.assert_array_equal(
                        out,
                        pd_out,
                        err_msg='custom op out: {},\n paddle api out: {}'.
                        format(out, pd_out))
                    np.testing.assert_array_equal(
                        x_grad,
                        pd_x_grad,
                        err_msg='custom op x grad: {},\n paddle api x grad: {}'.
                        format(x_grad, pd_x_grad))

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()

    def func_exception(self):
        caught_exception = False
        try:
            x = np.random.uniform(-1, 1, [4, 8]).astype('int32')
            custom_relu_dynamic(custom_module.custom_relu, 'cpu', 'int32', x)
        except OSError as e:
            caught_exception = True
            self.assertTrue("relu_cpu_forward" in str(e))
            self.assertTrue("int32" in str(e))
            self.assertTrue("custom_relu_op.cc" in str(e))
        self.assertTrue(caught_exception)
        caught_exception = False
        # MAC-CI don't support GPU
        if IS_MAC:
            return
        try:
            x = np.random.uniform(-1, 1, [4, 8]).astype('int32')
            custom_relu_dynamic(custom_module.custom_relu, 'gpu', 'int32', x)
        except OSError as e:
            caught_exception = True
            self.assertTrue("relu_cuda_forward_kernel" in str(e))
            self.assertTrue("int32" in str(e))
            self.assertTrue("custom_relu_op.cu" in str(e))
        self.assertTrue(caught_exception)

    def test_exception(self):
        with _test_eager_guard():
            self.func_exception()
        self.func_exception()

    def test_load_multiple_module(self):
        custom_module = load(
            name='custom_conj_jit',
            sources=['custom_conj_op.cc'],
            extra_include_paths=paddle_includes,  # add for Coverage CI
            extra_cxx_cflags=extra_cc_args,  # test for cc flags
            extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
            verbose=True)
        custom_conj = custom_module.custom_conj
        self.assertIsNotNone(custom_conj)


if __name__ == '__main__':
    unittest.main()
