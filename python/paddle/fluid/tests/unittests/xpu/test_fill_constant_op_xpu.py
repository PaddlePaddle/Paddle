#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("..")
import unittest
import paddle
from paddle.fluid import core
import numpy as np
from op_test import OpTest, convert_float_to_uint16
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestFillConstantOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fill_constant'
        self.use_dynamic_create_class = False

    # Situation 1: Attr(shape) is a list(without tensor)
    class TestFillConstantOp(XPUOpTest):
        def setUp(self):
            '''Test fill_constant op with specified value
            '''
            self.init_dtype()
            self.set_xpu()
            self.op_type = "fill_constant"
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.convert_dtype2index()
            self.set_value()
            self.set_data()

        def init_dtype(self):
            self.dtype = self.in_type

        def set_shape(self):
            self.shape = [90, 10]

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.in_type

        def convert_dtype2index(self):
            '''
            if new type added, need to add corresponding index
            '''
            if self.dtype == np.bool_:
                self.index = 0
            if self.dtype == np.int16:
                self.index = 1
            if self.dtype == np.int32:
                self.index = 2
            if self.dtype == np.int64:
                self.index = 3
            if self.dtype == np.float16:
                self.index = 4
            if self.dtype == np.float32:
                self.index = 5
            if self.dtype == np.float64:
                self.index = 6
            if self.dtype == np.uint8:
                self.index = 20
            if self.dtype == np.int8:
                self.index = 21
            if self.dtype == np.uint16:  # same as paddle.bfloat16
                self.index = 22
            if self.dtype == np.complex64:
                self.index = 23
            if self.dtype == np.complex128:
                self.index = 24

        def set_value(self):
            if self.index == 3:
                self.value = 10000000000
            elif self.index == 0:
                self.value = np.random.randint(0, 2)
            elif self.index in [20, 21]:
                self.value = 125
            elif self.index in [1, 2]:
                self.value = 7
            elif self.index in [4, 5, 6]:
                self.value = 1e-5
            elif self.index == 22:
                self.value = 1.0
            else:
                self.value = 3.7

        def set_data(self):
            self.inputs = {}
            self.attrs = {
                'shape': self.shape,
                'dtype': self.index,
                'value': self.value
            }
            self.outputs = {'Out': np.full(self.shape, self.value)}

        def test_check_output(self):
            self.check_output_with_place(self.place)

    class TestFillConstantOp2(TestFillConstantOp):
        '''Test fill_constant op with default value
        '''

        def set_shape(self):
            self.shape = [10, 10]

    class TestFillConstantOp3(TestFillConstantOp):
        '''Test fill_constant op with specified int64 value
        '''

        def set_shape(self):
            self.shape = [123, 2, 1]

    class TestFillConstantOp4(TestFillConstantOp):
        '''Test fill_constant op with specified int value
        '''

        def set_shape(self):
            self.shape = [123, 3, 2, 1]

    class TestFillConstantOp5(TestFillConstantOp):
        '''Test fill_constant op with specified float value
        '''

        def set_shape(self):
            self.shape = [123]

    # Situation 2: Attr(shape) is a list(with tensor)
    class TestFillConstantOp1_ShapeTensorList(TestFillConstantOp):
        '''Test fill_constant op with specified value
        '''

        def set_data(self):
            shape_tensor_list = []
            for index, ele in enumerate(self.shape):
                shape_tensor_list.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))

            self.inputs = {"ShapeTensorList": shape_tensor_list}
            self.attrs = {
                'shape': self.infer_shape,
                'dtype': self.index,
                'value': self.value
            }
            self.outputs = {'Out': np.full(self.shape, self.value)}
            if self.index == 22:
                self.outputs = {
                    'Out':
                    np.full(self.shape,
                            convert_float_to_uint16(
                                np.array([self.value]).astype("float32")))
                }

        def set_shape(self):
            self.shape = [123, 92]
            self.infer_shape = [123, 1]

    class TestFillConstantOp2_ShapeTensorList(TestFillConstantOp):
        '''Test fill_constant op with default value
        '''

        def set_data(self):
            shape_tensor_list = []
            for index, ele in enumerate(self.shape):
                shape_tensor_list.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))

            self.inputs = {"ShapeTensorList": shape_tensor_list}
            self.attrs = {'shape': self.infer_shape, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, 0.0)}

        def set_shape(self):
            self.shape = [123, 2, 1]
            self.infer_shape = [1, 1, 1]

    class TestFillConstantOp3_ShapeTensorList(
            TestFillConstantOp1_ShapeTensorList):
        def set_shape(self):
            self.shape = [123, 3, 2, 1]
            self.infer_shape = [123, 111, 11, 1]

    class TestFillConstantOp4_ShapeTensorList(
            TestFillConstantOp1_ShapeTensorList):
        def set_shape(self):
            self.shape = [123]
            self.infer_shape = [1]

    # Situation 3: shape is a tensor
    class TestFillConstantOp1_ShapeTensor(TestFillConstantOp):
        '''Test fill_constant op with specified value
        '''

        def set_data(self):
            self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
            self.attrs = {'value': self.value, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, self.value)}
            if self.index == 22:
                self.outputs = {
                    'Out':
                    np.full(self.shape,
                            convert_float_to_uint16(
                                np.array([self.value]).astype("float32")))
                }

        def set_shape(self):
            self.shape = [123, 92]

    # Situation 4: value is a tensor
    class TestFillConstantOp1_ValueTensor(TestFillConstantOp):
        '''Test fill_constant op with specified value
        '''

        def set_data(self):
            self.inputs = {
                "ShapeTensor": np.array(self.shape).astype("int32"),
                'ValueTensor': np.array([self.value]).astype(self.dtype)
            }
            if self.index == 22:
                self.inputs = {
                    'ValueTensor': convert_float_to_uint16(
                        np.array([self.value]).astype("float32"))
                }
            self.attrs = {'value': self.value, 'dtype': self.index}
            self.outputs = {'Out': np.full(self.shape, self.value)}

        def set_shape(self):
            self.shape = [123, 92]


support_types = get_xpu_op_support_types('fill_constant')
for stype in support_types:
    create_test_class(globals(), XPUTestFillConstantOp, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
