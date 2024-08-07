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

import unittest

import numpy as np
from op_test import OpTest, OpTestTool

import paddle


@OpTestTool.skip_if_not_cpu_bf16()
class TestFillConstant2DOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "fill_constant"
        self.dtype = np.float32

        self.shape_tensor_list = None
        self.shape_tensor = None
        self.str_value = ""
        real_shape = []
        self.value = 0.1

        self.set_inputs()
        self.set_attrs()

        if 'value' in self.attrs:
            self.value = self.attrs['value']
        if self.str_value != "":
            self.value = float(self.str_value)
        if 'ValueTensor' in self.inputs:
            self.value = self.inputs['ValueTensor']

        if 'shape' in self.attrs:
            real_shape = self.attrs['shape']
        if 'ShapeTensor' in self.inputs:
            real_shape = list(self.inputs['ShapeTensor'])
        if 'ShapeTensorList' in self.inputs:
            real_shape = []
            for shape_tensor in self.inputs['ShapeTensorList']:
                real_shape.append(shape_tensor[1].item())

        self.outputs = {'Out': np.full(real_shape, self.value)}

    def set_inputs(self):
        self.inputs = {}

    def set_attrs(self):
        self.attrs = {'shape': (3, 5), 'use_mkldnn': True, 'value': self.value}

    def test_check_output(self):
        self.check_output(check_pir_onednn=True)


class TestFillZerosLike4DShapeTensorPriorityOneDNNOp(
    TestFillConstant2DOneDNNOp
):
    def set_inputs(self):
        self.inputs = {'ShapeTensor': np.array([5, 6, 7, 8]).astype("int32")}


class TestFillZerosLike4DShapeTensorListPriorityOneDNNOp(
    TestFillConstant2DOneDNNOp
):
    def set_inputs(self):
        shape = (4, 5, 6, 7)
        self.shape_tensor_list = []
        for index, elem in enumerate(shape):
            self.shape_tensor_list.append(
                ("x" + str(index), np.ones(1).astype('int32') * elem)
            )

        self.inputs = {'ShapeTensorList': self.shape_tensor_list}


class TestFillZerosLike2DStringValueInfOneDNNOp(TestFillConstant2DOneDNNOp):
    def set_attrs(self):
        self.str_value = "inf"
        self.attrs = {'shape': (10, 13), 'use_mkldnn': True, 'str_value': "inf"}


class TestFillZerosLike2DStringValueMinusInfOneDNNOp(
    TestFillConstant2DOneDNNOp
):
    def set_attrs(self):
        self.str_value = "-inf"
        self.attrs = {
            'shape': (10, 13),
            'use_mkldnn': True,
            'str_value': "-inf",
        }


class TestFillZerosLike2DStringValueFloatOneDNNOp(TestFillConstant2DOneDNNOp):
    def set_attrs(self):
        self.str_value = "0.123"
        self.attrs = {
            'shape': (10, 13),
            'use_mkldnn': True,
            'str_value': "0.123",
        }


class TestFillZerosLike2DValueTensorPriorityOneDNNOp(
    TestFillZerosLike2DStringValueFloatOneDNNOp
):
    def set_inputs(self):
        self.inputs = {'ValueTensor': np.atleast_1d(2.25).astype("float32")}


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
