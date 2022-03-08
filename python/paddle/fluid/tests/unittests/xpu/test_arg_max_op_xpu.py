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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")

import paddle
from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestArgMax(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'arg_max'

    class XPUBaseTestCase(XPUOpTest):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

        def setUp(self):
            self.op_type = 'arg_max'
            self.dtype = self.in_type
            self.initTestCase()

            self.x = (np.random.random(self.dims)).astype(self.dtype)
            self.inputs = {'X': self.x}
            self.attrs = {'axis': self.axis, 'use_xpu': True}
            self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestArgMaxCase1(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = -1

    class TestArgMaxCase2(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 0

    class TestArgMaxCase3(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 1

    class TestArgMaxCase4(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4, 5)
            self.axis = 2

    class TestArgMaxCase5(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = -1

    class TestArgMaxCase6(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 0

    class TestArgMaxCase7(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, 4)
            self.axis = 1

    class TestArgMaxCase8(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (1, )
            self.axis = 0

    class TestArgMaxCase9(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (2, )
            self.axis = 0

    class TestArgMaxCase10(XPUBaseTestCase):
        def initTestCase(self):
            self.dims = (3, )
            self.axis = 0


support_types = get_xpu_op_support_types('arg_max')
for stype in support_types:
    create_test_class(globals(), XPUTestArgMax, stype)


class TestArgMaxAPI(unittest.TestCase):
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_Xpu = True
        self.place = [paddle.XPUPlace(0)]

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input, axis=self.axis)
            paddle_output = paddle.argmax(tensor_input, axis=self.axis)
            self.assertEqual(
                np.allclose(numpy_output, paddle_output.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMaxAPI_2(unittest.TestCase):
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
            numpy_output = np.argmax(
                numpy_input, axis=self.axis).reshape(1, 4, 5)
            paddle_output = paddle.argmax(
                tensor_input, axis=self.axis, keepdim=self.keep_dims)
            self.assertEqual(
                np.allclose(numpy_output, paddle_output.numpy()), True)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
