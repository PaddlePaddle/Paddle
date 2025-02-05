# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from utils import (
    extra_cc_args,
    extra_nvcc_args,
    paddle_includes,
    paddle_libraries,
)

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_contiguous\\custom_contiguous.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

custom_module = load(
    name='custom_contiguous',
    sources=['custom_contiguous.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_library_paths=paddle_libraries,
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


def custom_contiguous_dynamic(device, np_x):
    paddle.set_device(device)

    x = paddle.to_tensor(np_x, dtype="float32")
    x.stop_gradient = True

    x = x.transpose((1, 0))

    out = custom_module.custom_contiguous(x)

    assert out.is_contiguous()


class TestCustomCastOp(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        paddle.set_flags({"FLAGS_use_stride_kernel": 1})

    def test_dynamic(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype("float32")
            custom_contiguous_dynamic('cpu', x)


if __name__ == '__main__':
    unittest.main()
