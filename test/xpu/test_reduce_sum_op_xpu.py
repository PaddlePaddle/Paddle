#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestReduceSumOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_sum'

    class XPUTestReduceSumBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type
            self.init_case()
            self.set_case()

        def gen_data_depend_on_dtype(self, shape):
            if (
                self.dtype == np.int32
                or self.dtype == np.int64
                or self.dtype == np.uint8
            ):
                return np.random.randint(1, 100, size=shape)
            else:
                return np.random.uniform(-1, 1, size=shape)

        def set_case(self):
            self.op_type = 'reduce_sum'
            self.attrs = {
                'use_xpu': True,
                'reduce_all': self.reduce_all,
                'keep_dim': self.keep_dim,
                'dim': self.axis,
            }
            tmp_x = self.gen_data_depend_on_dtype(self.shape)
            if self.dtype == np.uint16:
                tmp_out = (
                    tmp_x.sum()
                    if self.attrs['reduce_all']
                    else tmp_x.sum(
                        axis=self.axis, keepdims=self.attrs['keep_dim']
                    )
                )
                self.outputs = {'Out': tmp_out}
                tmp_x = convert_float_to_uint16(tmp_x)
                self.inputs = {'X': tmp_x}
            else:
                tmp_x = tmp_x.astype(self.dtype)
                self.inputs = {'X': tmp_x}
                tmp_out = (
                    tmp_x.sum()
                    if self.attrs['reduce_all']
                    else tmp_x.sum(
                        axis=self.axis, keepdims=self.attrs['keep_dim']
                    )
                )
                self.outputs = {'Out': tmp_out}

        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class XPUTestReduceSumCase1(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase2(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase3(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (0,)
            self.reduce_all = True
            self.keep_dim = False

    class XPUTestReduceSumCase4(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    class XPUTestReduceSumCase5(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = True

    class XPUTestReduceSumCase6(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (5, 6, 10)
            self.axis = (1,)
            self.reduce_all = True
            self.keep_dim = False

    @check_run_big_shape_test()
    class XPUTestReduceSumLargeShape1(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (8192,)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    @check_run_big_shape_test()
    class XPUTestReduceSumLargeShape2(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (1, 8192)
            self.axis = (1,)
            self.reduce_all = False
            self.keep_dim = False

    @check_run_big_shape_test()
    class XPUTestReduceSumLargeShape3(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (224, 1)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    @check_run_big_shape_test()
    class XPUTestReduceSumLargeShape4(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (334, 1)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False

    @check_run_big_shape_test()
    class XPUTestReduceSumLargeShape5(XPUTestReduceSumBase):
        def init_case(self):
            self.shape = (338, 1)
            self.axis = (0,)
            self.reduce_all = False
            self.keep_dim = False


support_types = get_xpu_op_support_types('reduce_sum')
for stype in support_types:
    create_test_class(globals(), XPUTestReduceSumOp, stype)

if __name__ == '__main__':
    unittest.main()
