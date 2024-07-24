#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle


class XPUTestXPUFullBatchSizeLikeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fill_constant_batch_size_like'
        self.use_dynamic_create_class = False

    class TestXPUFullBatchSizeLikeOp(XPUOpTest):
        def setUp(self):
            self.init_op_type()
            self.initTestCase()
            self.use_xpu = True
            self.use_mkldnn = False
            self.no_need_check_grad = True
            self.dtype = self.in_type
            self.input_dim_idx = 0
            self.output_dim_idx = 0
            self.convert_dtype2index()
            self.inputs = {
                'Input': np.random.random(self.shape).astype(self.dtype)
            }
            self.attrs = {
                'shape': self.shape,
                'value': self.value,  # 'str_value'
                'dtype': self.index,
                'input_dim_idx': self.input_dim_idx,
                'output_dim_idx': self.output_dim_idx,
            }
            self.outputs = {
                'Out': np.full(self.shape, self.value).astype(self.dtype)
            }

        def init_op_type(self):
            self.op_type = "fill_constant_batch_size_like"
            self.use_mkldnn = False

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

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place=place)

        def initTestCase(self):
            self.shape = (3, 40)
            self.value = 0.0

    class TestCase0(TestXPUFullBatchSizeLikeOp):
        def initTestCase(self):
            self.shape = (100,)
            self.value = 1.0

    class TestCase1(TestXPUFullBatchSizeLikeOp):
        def initTestCase(self):
            self.shape = (3, 4, 10)
            self.value = -1e-5

    class TestCase2(TestXPUFullBatchSizeLikeOp):
        def initTestCase(self):
            self.shape = (2, 3, 4, 5)
            self.value = -5.0

    @check_run_big_shape_test()
    class TestCase3(TestXPUFullBatchSizeLikeOp):
        def initTestCase(self):
            self.shape = (8192, 4096)
            self.value = 5.0


support_types = get_xpu_op_support_types('fill_constant_batch_size_like')
for stype in support_types:
    create_test_class(globals(), XPUTestXPUFullBatchSizeLikeOp, stype)

if __name__ == "__main__":
    unittest.main()
