#  Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestArgMin(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'arg_min'

    class XPUBaseTestCase(XPUOpTest):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

        def setUp(self):
            self.op_type = 'arg_min'
            self.dtype = self.in_type
            self.initTestCase()

            self.x = (np.random.random(self.dims)).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )

            self.inputs = {
                'X': (
                    self.x
                    if self.dtype != np.uint16
                    else convert_float_to_uint16(self.x)
                )
            }
            self.attrs = {'axis': self.axis, 'use_xpu': True}
            self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestArgMinCase1(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = -1

    class TestArgMinCase2(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 0

    class TestArgMinCase3(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 1

    class TestArgMinCase4(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 2

    class TestArgMinCase5(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = -1

    class TestArgMinCase6(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 0

    class TestArgMinCase7(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

    class TestArgMinCase8(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (1,)
            self.axis = 0

    class TestArgMinCase9(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (2,)
            self.axis = 0

    class TestArgMinCase10(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3,)
            self.axis = 0


support_types = get_xpu_op_support_types('arg_min')
for stype in support_types:
    create_test_class(globals(), XPUTestArgMin, stype)


class TestArgMinAPI(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_xpu = True
        self.place = [paddle.XPUPlace(0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input, axis=self.axis)
            paddle_output = paddle.argmin(tensor_input, axis=self.axis)
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMinAPI_2(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0
        self.keep_dims = True

    def setUp(self):
        self.initTestCase()
        self.__class__.use_xpu = True
        self.place = [paddle.XPUPlace(0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input, axis=self.axis).reshape(
                1, 4, 5
            )
            paddle_output = paddle.argmin(
                tensor_input, axis=self.axis, keepdim=self.keep_dims
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
