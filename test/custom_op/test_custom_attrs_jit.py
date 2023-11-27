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
from utils import extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_attrs_jit\\custom_attrs_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_attrs = load(
    name='custom_attrs_jit',
    sources=['attr_test_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


class TestJitCustomAttrs(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')
        # prepare test value
        self.bool_attr = True
        self.int_attr = 10
        self.float_attr = 3.14
        self.int64_attr = 10000000000
        self.str_attr = "StrAttr"
        self.int_vec_attr = [10, 10, 10]
        self.float_vec_attr = [3.14, 3.14, 3.14]
        self.int64_vec_attr = [10000000000, 10000000000, 10000000000]
        self.str_vec_attr = ["StrAttr", "StrAttr", "StrAttr"]

    def test_func_attr_value(self):
        x = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        out = custom_attrs.attr_test(
            x,
            self.bool_attr,
            self.int_attr,
            self.float_attr,
            self.int64_attr,
            self.str_attr,
            self.int_vec_attr,
            self.float_vec_attr,
            self.int64_vec_attr,
            self.str_vec_attr,
        )
        out.stop_gradient = False
        out.backward()

        np.testing.assert_array_equal(x.numpy(), out.numpy())

    def test_const_attr_value(self):
        x = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        out = custom_attrs.const_attr_test(
            x,
            self.bool_attr,
            self.int_attr,
            self.float_attr,
            self.int64_attr,
            self.str_attr,
            self.int_vec_attr,
            self.float_vec_attr,
            self.int64_vec_attr,
            self.str_vec_attr,
        )
        out.stop_gradient = False
        out.backward()

        np.testing.assert_array_equal(x.numpy(), out.numpy())


if __name__ == '__main__':
    unittest.main()
