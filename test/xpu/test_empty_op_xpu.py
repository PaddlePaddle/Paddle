# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.base.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


class XPUTestEmptyOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'empty'
        self.use_dynamic_create_class = False

    # Situation 1: Attr(shape) is a list(without tensor)
    class TestEmptyOp(XPUOpTest):
        def setUp(self):
            self.op_type = "empty"
            self.init_dtype()
            self.set_xpu()
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.set_inputs()
            self.init_config()

        def test_check_output(self):
            self.check_output_customized(self.verify_output)

        def verify_output(self, outs):
            data_type = outs[0].dtype
            if data_type in [
                'float32',
                'float64',
                'int32',
                'int64',
                'int8',
                'uint8',
                'float16',
                'int16',
                'uint16',
            ]:
                max_value = np.nanmax(outs[0])
                min_value = np.nanmin(outs[0])

                always_full_zero = max_value == 0.0 and min_value == 0.0
                always_non_full_zero = max_value >= min_value
                self.assertTrue(
                    always_full_zero or always_non_full_zero,
                    'always_full_zero or always_non_full_zero.',
                )
            elif data_type in ['bool']:
                total_num = outs[0].size
                true_num = np.sum(outs[0])
                false_num = np.sum(~outs[0])
                self.assertTrue(
                    total_num == true_num + false_num,
                    'The value should always be True or False.',
                )
            else:
                # pass
                self.assertTrue(False, 'invalid data type')

        def set_shape(self):
            self.shape = [500, 3]

        def set_inputs(self):
            self.inputs = {}

        def init_config(self):
            dtype_inner = convert_np_dtype_to_dtype_(self.dtype)
            self.attrs = {'shape': self.shape, 'dtype': dtype_inner}
            self.outputs = {'Out': np.zeros(self.shape).astype(self.dtype)}

        def init_dtype(self):
            self.dtype = self.in_type

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.op_type

    class TestEmptyOpCase1(TestEmptyOp):
        def set_shape(self):
            self.shape = [50]

    class TestEmptyOpCase2(TestEmptyOp):
        def set_shape(self):
            self.shape = [1, 50, 3, 4]

    class TestEmptyOpCase3(TestEmptyOp):
        def set_shape(self):
            self.shape = [5, 5, 5]

    # Situation 2: shape is a tensor
    class TestEmptyOp_ShapeTensor(TestEmptyOp):
        def set_inputs(self):
            self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}

    # Situation 3: Attr(shape) is a list(with tensor)
    class TestEmptyOp_ShapeTensorList(TestEmptyOp):
        def set_inputs(self):
            shape_tensor_list = []
            for index, ele in enumerate(self.shape):
                shape_tensor_list.append(
                    ("x" + str(index), np.ones(1).astype('int32') * ele)
                )

            self.inputs = {"ShapeTensorList": shape_tensor_list}


support_types = get_xpu_op_support_types('empty')
for stype in support_types:
    create_test_class(globals(), XPUTestEmptyOp, stype)

if __name__ == '__main__':
    unittest.main()
