#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle
import paddle.fluid.core as core
from paddle.static import program_guard, Program
import paddle.compat as cpt
import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


def fill_constant_batch_size_like(input,
                                  shape,
                                  value,
                                  data_type,
                                  input_dim_idx=0,
                                  output_dim_idx=0,
                                  force_cpu=False):
    return paddle.fluid.layers.fill_constant_batch_size_like(
        input, shape, data_type, value, input_dim_idx, output_dim_idx,
        force_cpu)


class TestFillConstatnBatchSizeLike1(OpTest):
    # test basic
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.python_api = fill_constant_batch_size_like
        self.init_data()

        input = np.zeros(self.shape)
        out = np.full_like(input, self.value, self.dtype)

        self.inputs = {'Input': input}
        self.outputs = {'Out': out}
        self.attrs = {
            'shape': self.shape,
            'dtype': convert_np_dtype_to_dtype_(self.dtype),
            'value': self.value,
            'input_dim_idx': self.input_dim_idx,
            'output_dim_idx': self.output_dim_idx,
            'force_cpu': self.force_cpu
        }

    def init_data(self):
        self.shape = [10, 10]
        self.dtype = np.float32
        self.value = 100
        self.input_dim_idx = 0
        self.output_dim_idx = 0
        self.force_cpu = False

    def test_check_output(self):
        self.check_output(check_eager=True)


if __name__ == "__main__":
    unittest.main()
