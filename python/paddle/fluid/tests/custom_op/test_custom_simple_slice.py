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
from paddle.fluid.framework import _test_eager_guard
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_simple_slice\\custom_simple_slice.pyd'.format(
    get_build_directory()
)
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_simple_slice_jit',
    sources=['custom_simple_slice_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


class TestCustomSimpleSliceJit(unittest.TestCase):
    def func_slice_output(self):
        np_x = np.random.random((5, 2)).astype("float32")
        x = paddle.to_tensor(np_x)
        custom_op_out = custom_ops.custom_simple_slice(x, 2, 3)
        np_out = np_x[2:3]
        np.testing.assert_array_equal(
            custom_op_out,
            np_out,
            err_msg='custom op: {},\n numpy: {}'.format(
                np_out, custom_op_out.numpy()
            ),
        )

    def test_slice_output(self):
        with _test_eager_guard():
            self.func_slice_output()
        self.func_slice_output()


if __name__ == "__main__":
    unittest.main()
