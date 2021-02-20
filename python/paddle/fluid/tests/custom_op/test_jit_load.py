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
from paddle.utils.cpp_extension.extension_utils import use_new_custom_op_load_method

# switch to old custom op method
use_new_custom_op_load_method(False)

# Compile and load custom op Just-In-Time.
custom_module = load(
    name='custom_relu2',
    sources=['relu_op.cc', 'relu_op.cu', 'relu_op3.cc', 'relu_op3.cu'],
    interpreter='python',  # add for unittest
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cflags=extra_compile_args,  # add for Coverage CI
    verbose=True  # add for unittest
)


class TestJITLoad(unittest.TestCase):
    def test_api(self):
        raw_data = np.array([[-1, 1, 0], [1, -1, -1]]).astype('float32')
        gt_data = np.array([[0, 1, 0], [1, 0, 0]]).astype('float32')
        x = paddle.to_tensor(raw_data, dtype='float32')
        # use custom api
        out = custom_module.relu2(x)
        out3 = custom_module.relu3(x)

        self.assertTrue(np.array_equal(out.numpy(), gt_data))
        self.assertTrue(np.array_equal(out3.numpy(), gt_data))


if __name__ == '__main__':
    unittest.main()
