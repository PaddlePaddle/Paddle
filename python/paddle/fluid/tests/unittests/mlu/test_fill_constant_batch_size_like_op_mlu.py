#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("..")
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


class TestFillConstantBatchSizeLike(OpTest):

    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = "fill_constant_batch_size_like"
        self.init_shape()
        self.init_value()
        self.init_dtype()
        self.init_force_cpu()
        self.init_dim_idx()

        self.inputs = {
            'Input': np.random.random(self.input_shape).astype("float32")
        }
        self.attrs = {
            'shape': self.shape,
            'value': self.value,
            'str_value': self.str_value,
            'dtype': self.dtype,
            'force_cpu': self.force_cpu,
            'input_dim_idx': self.input_dim_idx,
            'output_dim_idx': self.output_dim_idx
        }
        self.outputs = {
            'Out': np.full(self.output_shape, self.output_value,
                           self.output_dtype)
        }

    def init_shape(self):
        self.input_shape = [4, 5]
        self.shape = [123, 92]
        self.output_shape = (4, 92)

    def init_value(self):
        self.value = 3.8
        self.str_value = ''
        self.output_value = 3.8

    def init_dtype(self):
        self.dtype = core.VarDesc.VarType.FP32
        self.output_dtype = np.float32

    def init_force_cpu(self):
        self.force_cpu = False

    def init_dim_idx(self):
        self.input_dim_idx = 0
        self.output_dim_idx = 0

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestFillConstantBatchSizeLike2(TestFillConstantBatchSizeLike):

    def init_shape(self):
        # test shape
        self.input_shape = [4, 5, 6, 7]
        self.shape = [10, 123, 92]
        self.output_shape = (4, 123, 92)


class TestFillConstantBatchSizeLike3(TestFillConstantBatchSizeLike):

    def init_value(self):
        # use 'str_value' rather than 'value'
        self.value = 3.8
        self.str_value = '4.5'
        self.output_value = 4.5


class TestFillConstantBatchSizeLike4(TestFillConstantBatchSizeLike):

    def init_value(self):
        # str_value = 'inf'
        self.value = 3.8
        self.str_value = 'inf'
        self.output_value = float('inf')


class TestFillConstantBatchSizeLike5(TestFillConstantBatchSizeLike):

    def init_value(self):
        # str_value = '-inf'
        self.value = 3.8
        self.str_value = '-inf'
        self.output_value = -float('inf')


class TestFillConstantBatchSizeLike6(TestFillConstantBatchSizeLike):

    def init_dtype(self):
        self.dtype = core.VarDesc.VarType.FP16
        self.output_dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-2)


class TestFillConstantBatchSizeLike7(TestFillConstantBatchSizeLike):

    def init_dtype(self):
        self.dtype = core.VarDesc.VarType.INT32
        self.output_dtype = np.int32


class TestFillConstantBatchSizeLike8(TestFillConstantBatchSizeLike):

    def init_force_cpu(self):
        self.force_cpu = True


class TestFillConstantBatchSizeLike9(TestFillConstantBatchSizeLike):

    def init_shape(self):
        self.input_shape = [4, 5]
        self.shape = [123, 92]
        self.output_shape = (123, 4)

    def init_dim_idx(self):
        self.input_dim_idx = 0
        self.output_dim_idx = 1


class TestFillConstantBatchSizeLikeLodTensor(TestFillConstantBatchSizeLike):
    # test LodTensor
    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.op_type = "fill_constant_batch_size_like"
        self.init_shape()
        self.init_value()
        self.init_dtype()
        self.init_force_cpu()
        self.init_dim_idx()

        lod = [[3, 2, 5]]
        self.inputs = {
            'Input': (np.random.random(self.input_shape).astype("float32"), lod)
        }
        self.attrs = {
            'shape': self.shape,
            'value': self.value,
            'str_value': self.str_value,
            'dtype': self.dtype,
            'force_cpu': self.force_cpu,
            'input_dim_idx': self.input_dim_idx,
            'output_dim_idx': self.output_dim_idx
        }
        self.outputs = {
            'Out': np.full(self.output_shape, self.output_value,
                           self.output_dtype)
        }

    def init_shape(self):
        self.input_shape = [10, 20]
        self.shape = [123, 92]
        self.output_shape = (3, 92)


class TestFillConstantBatchSizeLikeLodTensor2(
        TestFillConstantBatchSizeLikeLodTensor):
    # test LodTensor with 'input_dim_idx' != 0
    def init_shape(self):
        self.input_shape = [10, 20]
        self.shape = [123, 92]
        self.output_shape = (20, 92)

    def init_dim_idx(self):
        self.input_dim_idx = 1
        self.output_dim_idx = 0


if __name__ == "__main__":
    unittest.main()
