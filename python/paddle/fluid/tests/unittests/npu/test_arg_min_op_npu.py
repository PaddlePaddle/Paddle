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
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid.core as core

paddle.enable_static()


class BaseTestCase(OpTest):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = 1

    def setUp(self):
        self.initTestCase()
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)
        np.random.seed(2021)
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        if self.op_type == "arg_min":
            self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}
        else:
            self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


# test argmin, dtype: float16
class TestArgMinFloat16Case1(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = -1


class TestArgMinFloat16Case2(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 0


class TestArgMinFloat16Case3(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 1


class TestArgMinFloat16Case4(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 2


class TestArgMinFloat16Case5(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = -1


class TestArgMinFloat16Case6(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = 0


class TestArgMinFloat16Case7(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = 1


class TestArgMinFloat16Case8(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1, )
        self.dtype = 'float16'
        self.axis = 0


class TestArgMinFloat16Case9(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (2, )
        self.dtype = 'float16'
        self.axis = 0


class TestArgMinFloat16Case10(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, )
        self.dtype = 'float16'
        self.axis = 0


# test argmin, dtype: float32
class TestArgMinFloat32Case1(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = -1


class TestArgMinFloat32Case2(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0


class TestArgMinFloat32Case3(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 1


class TestArgMinFloat32Case4(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 2


class TestArgMinFloat32Case5(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = -1


class TestArgMinFloat32Case6(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = 0


class TestArgMinFloat32Case7(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = 1


class TestArgMinFloat32Case8(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1, )
        self.dtype = 'float32'
        self.axis = 0


class TestArgMinFloat32Case9(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (2, )
        self.dtype = 'float32'
        self.axis = 0


class TestArgMinFloat32Case10(BaseTestCase):

    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, )
        self.dtype = 'float32'
        self.axis = 0


class TestArgMinAPI(unittest.TestCase):

    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_npu = True
        self.place = [paddle.NPUPlace(0)]

    def test_dygraph_api(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input, axis=self.axis)
            paddle_output = paddle.argmin(tensor_input, axis=self.axis)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=1e-05)
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
        self.__class__.use_npu = True
        self.place = [paddle.NPUPlace(0)]

    def test_dygraph_api(self):

        def run(place):
            paddle.disable_static(place)
            np.random.seed(2021)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmin(numpy_input,
                                     axis=self.axis).reshape(1, 4, 5)
            paddle_output = paddle.argmin(tensor_input,
                                          axis=self.axis,
                                          keepdim=self.keep_dims)
            np.testing.assert_allclose(numpy_output,
                                       paddle_output.numpy(),
                                       rtol=1e-05)
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
