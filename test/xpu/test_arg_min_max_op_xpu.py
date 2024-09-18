#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


# Combine arg_max and arg_min tests


class XPUTestArgMinMax(XPUOpTestWrapper):
    def __init__(self, op_name):
        self.op_name = op_name

    class XPUBaseTestCase(XPUOpTest):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

        def setUp(self):
            self.op_type = self.__class__.op_name
            self.dtype = self.in_type
            self.initTestCase()

            self.x = (np.random.random(self.dims)).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.attrs = {'axis': self.axis, 'use_xpu': True}
            if self.op_type == 'arg_max':
                self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}
            else:
                self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestCase1(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = -1

    class TestCase2(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 0

    class TestCase3(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 1

    class TestCase4(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 2

    class TestCase5(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = -1

    class TestCase6(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 0

    class TestCase7(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

    class TestCase8(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (1,)
            self.axis = 0

    class TestCase9(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (2,)
            self.axis = 0

    class TestCase10(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3,)
            self.axis = 0


# Create arg_max and arg_min tests

support_types = get_xpu_op_support_types('arg_max')
for stype in support_types:
    test_class = XPUTestArgMinMax('arg_max')
    create_test_class(globals(), test_class, stype)

support_types = get_xpu_op_support_types('arg_min')
for stype in support_types:
    test_class = XPUTestArgMinMax('arg_min')
    create_test_class(globals(), test_class, stype)


# API Tests for arg_max and arg_min
class TestArgMinMaxAPI(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_Xpu = True
        self.place = [paddle.XPUPlace(0)]

    def test_dygraph_api(self):
        def run(place, op_name):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            if op_name == 'arg_max':
                numpy_output = np.argmax(numpy_input, axis=self.axis)
                paddle_output = paddle.argmax(tensor_input, axis=self.axis)
            else:
                numpy_output = np.argmin(numpy_input, axis=self.axis)
                paddle_output = paddle.argmin(tensor_input, axis=self.axis)

            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
            paddle.enable_static()

        for place in self.place:
            run(place, 'arg_max')
            run(place, 'arg_min')


class TestArgMinMaxAPI_2(unittest.TestCase):
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
        def run(place, op_name):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            if op_name == 'arg_max':
                numpy_output = np.argmax(numpy_input, axis=self.axis).reshape(
                    1, 4, 5
                )
                paddle_output = paddle.argmax(
                    tensor_input, axis=self.axis, keepdim=self.keep_dims
                )
            else:
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
            run(place, 'arg_max')
            run(place, 'arg_min')


if __name__ == '__main__':
    unittest.main()
