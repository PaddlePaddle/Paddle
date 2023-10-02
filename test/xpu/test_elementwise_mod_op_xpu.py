#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
from op_test import OpTest
from op_test_xpu import XPUOpTest

import paddle
from paddle import base

paddle.enable_static()


class XPUTestElementwiseModOp(XPUOpTestWrapper):
    def __init__(self) -> None:
        self.op_name = 'elementwise_mod'
        self.use_dynamic_create_class = False

    class ElementwiseModOp(XPUOpTest):
        def init_kernel_type(self):
            self.use_mkldnn = False

        def init_input_output(self):
            self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
            self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
            self.out = np.mod(self.x, self.y)
            self.inputs = {
                'X': OpTest.np_dtype_to_base_dtype(self.x),
                'Y': OpTest.np_dtype_to_base_dtype(self.y),
            }
            self.outputs = {'Out': self.out}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

        def init_dtype(self):
            pass

        def init_axis(self):
            pass

        def setUp(self):
            self.op_type = 'elementwise_mod'
            self.use_xpu = True
            self.dtype = self.in_type
            self.axis = -1
            self.init_dtype()
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestRemainderOp(unittest.TestCase):
        def test_dygraph(self):
            with base.dygraph.guard():
                np_x = np.random.rand(22, 128, 3).astype('int64')
                np_y = np.random.rand(22, 128, 3).astype('int64')
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.remainder(x, y)
                np_z = z.numpy()
                z_expected = np.mod(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)

                np_x = np.array([-3.3, 11.5, -2, 3.5])
                np_y = np.array([-1.2, 2.0, 3.3, -2.3])
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = x % y
                z_expected = np.array([-0.9, 1.5, 1.3, -1.1])
                np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)

                np_x = np.random.rand(22, 128, 3).astype('int32')
                np_y = np.random.rand(22, 128, 3).astype('int32')
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.remainder(x, y)
                np_z = z.numpy()
                z_expected = np.mod(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)

                np_x = np.array([-3, 11, -2, 3])
                np_y = np.array([-1, 2, 3, -2])
                x = paddle.to_tensor(np_x, dtype="float16")
                y = paddle.to_tensor(np_y, dtype="float16")
                z = x % y
                z_expected = np.array([0, 1, 1, -1])
                np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)


support_types = get_xpu_op_support_types('elementwise_mod')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseModOp, stype)

if __name__ == '__main__':
    unittest.main()
