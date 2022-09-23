# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.utils.cpp_extension import load, get_build_directory
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.fluid.framework import _test_eager_guard

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\context_pool_jit\\context_pool_jit.pyd'.format(
    get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_ops = load(
    name='context_pool_jit',
    sources=['context_pool_test_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True)


class TestContextPool(unittest.TestCase):

    def setUp(self):
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

    def use_context_pool(self):
        x = paddle.ones([2, 2], dtype='float32')
        out = custom_ops.context_pool_test(x)

        np.testing.assert_array_equal(x.numpy(), out.numpy())

    def test_using_context_pool(self):
        with _test_eager_guard():
            self.use_context_pool()
        self.use_context_pool()


if __name__ == '__main__':
    unittest.main()
