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

# Compile and load custom op Just-In-Time.
simple_relu2 = load(
    name='simple_jit_relu2',
    sources=['relu_op_simple.cc', 'relu_op_simple.cu'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cflags=extra_compile_args)  # add for Coverage CI


class TestJITLoad(unittest.TestCase):
    def test_api(self):
        raw_data = np.array([[-1, 1, 0], [1, -1, -1]]).astype('float32')
        x = paddle.to_tensor(raw_data, dtype='float32')
        # use custom api
        out = simple_relu2(x)
        self.assertTrue(
            np.array_equal(out.numpy(),
                           np.array([[0, 1, 0], [1, 0, 0]]).astype('float32')))


if __name__ == '__main__':
    unittest.main()
