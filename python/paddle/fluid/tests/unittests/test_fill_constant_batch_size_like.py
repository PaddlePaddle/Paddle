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

import unittest

import numpy as np
from eager_op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.fluid import core
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


def fill_constant_batch_size_like(
    input,
    shape,
    value,
    data_type,
    input_dim_idx=0,
    output_dim_idx=0,
    force_cpu=False,
):
    return paddle.fluid.layers.fill_constant_batch_size_like(
        input, shape, data_type, value, input_dim_idx, output_dim_idx, force_cpu
    )


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
            'force_cpu': self.force_cpu,
        }

    def init_data(self):
        self.shape = [10, 10]
        self.dtype = np.float32
        self.value = 100
        self.input_dim_idx = 0
        self.output_dim_idx = 0
        self.force_cpu = False

    def test_check_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.supports_bfloat16(),
    "core is not compiled with CUDA or place do not support bfloat16",
)
class TestFillConstatnBatchSizeLikeBf16(OpTest):
    # test bf16
    def setUp(self):
        self.op_type = "fill_constant_batch_size_like"
        self.python_api = fill_constant_batch_size_like
        self.init_data()

        input = np.zeros(self.shape).astype("float32")
        input_bf16 = convert_float_to_uint16(input)
        out = np.full_like(input, self.value, np.float32)
        out_bf16 = convert_float_to_uint16(out)

        self.inputs = {'Input': input_bf16}
        self.outputs = {'Out': out_bf16}
        self.attrs = {
            'shape': self.shape,
            'dtype': convert_np_dtype_to_dtype_(self.dtype),
            'value': self.value,
            'input_dim_idx': self.input_dim_idx,
            'output_dim_idx': self.output_dim_idx,
            'force_cpu': self.force_cpu,
        }

    def init_data(self):
        self.shape = [10, 10]
        self.dtype = np.uint16
        self.value = 100
        self.input_dim_idx = 0
        self.output_dim_idx = 0
        self.force_cpu = False

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
