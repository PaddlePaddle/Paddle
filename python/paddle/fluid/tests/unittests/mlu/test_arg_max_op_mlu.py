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

<<<<<<< HEAD
from __future__ import print_function

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
import unittest
import numpy as np
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

paddle.enable_static()


class BaseTestCase(OpTest):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.set_mlu()
        self.initTestCase()
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestArgMaxSameValue1(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.set_mlu()
        self.initTestCase()
        self.x = np.array([1, 2, 3, 5, 4, 5]).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}


class TestArgMaxSameValue2(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dtype = 'float16'
        self.axis = 0

    def setUp(self):
        self.set_mlu()
        self.initTestCase()
        self.x = np.array([[2, 3, 5, 5], [3, 2, 5, 5]]).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}


# test argmax, dtype: float16
class TestArgMaxFloat16Case1(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = -1


class TestArgMaxFloat16Case2(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 0


class TestArgMaxFloat16Case3(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 1


class TestArgMaxFloat16Case4(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float16'
        self.axis = 2


class TestArgMaxFloat16Case5(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = -1


class TestArgMaxFloat16Case6(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = 0


class TestArgMaxFloat16Case7(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float16'
        self.axis = 1


class TestArgMaxFloat16Case8(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (1, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (1,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float16'
        self.axis = 0


class TestArgMaxFloat16Case9(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (2, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (2,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float16'
        self.axis = 0


class TestArgMaxFloat16Case10(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float16'
        self.axis = 0


# test argmax, dtype: float32
class TestArgMaxFloat32Case1(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = -1


class TestArgMaxFloat32Case2(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0


class TestArgMaxFloat32Case3(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 1


class TestArgMaxFloat32Case4(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 2


class TestArgMaxFloat32Case5(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = -1


class TestArgMaxFloat32Case6(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = 0


class TestArgMaxFloat32Case7(BaseTestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'float32'
        self.axis = 1


class TestArgMaxFloat32Case8(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (1, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (1,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float32'
        self.axis = 0


class TestArgMaxFloat32Case9(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (2, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (2,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float32'
        self.axis = 0


class TestArgMaxFloat32Case10(BaseTestCase):
<<<<<<< HEAD

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
=======
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3,)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        self.dtype = 'float32'
        self.axis = 0


class BaseTestComplex1_1(OpTest):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'float32'
        self.axis = 2

    def setUp(self):
        self.set_mlu()
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
<<<<<<< HEAD
            'dtype': int(core.VarDesc.VarType.INT32)
=======
            'dtype': int(core.VarDesc.VarType.INT32),
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        }
        self.outputs = {
            'Out': np.argmax(self.x, axis=self.axis).astype("int32")
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


class BaseTestComplex1_2(OpTest):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def set_mlu(self):
        self.__class__.use_mlu = True
        self.place = paddle.MLUPlace(0)

    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'float16'
        self.axis = 2

    def setUp(self):
        self.set_mlu()
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
<<<<<<< HEAD
            'dtype': int(core.VarDesc.VarType.INT32)
=======
            'dtype': int(core.VarDesc.VarType.INT32),
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        }
        self.outputs = {
            'Out': np.argmax(self.x, axis=self.axis).astype("int32")
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestArgMaxAPI(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.__class__.use_mlu = True
        self.place = [paddle.MLUPlace(0)]

    def test_dygraph_api(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input, axis=self.axis)
            paddle_output = paddle.argmax(tensor_input, axis=self.axis)
<<<<<<< HEAD
            self.assertEqual(np.allclose(numpy_output, paddle_output.numpy()),
                             True)
=======
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMaxAPI_2(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0
        self.keep_dims = True

    def setUp(self):
        self.initTestCase()
        self.__class__.use_mlu = True
        self.place = [paddle.MLUPlace(0)]

    def test_dygraph_api(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
<<<<<<< HEAD
            numpy_output = np.argmax(numpy_input,
                                     axis=self.axis).reshape(1, 4, 5)
            paddle_output = paddle.argmax(tensor_input,
                                          axis=self.axis,
                                          keepdim=self.keep_dims)
            self.assertEqual(np.allclose(numpy_output, paddle_output.numpy()),
                             True)
=======
            numpy_output = np.argmax(numpy_input, axis=self.axis).reshape(
                1, 4, 5
            )
            paddle_output = paddle.argmax(
                tensor_input, axis=self.axis, keepdim=self.keep_dims
            )
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


class TestArgMaxAPI_3(unittest.TestCase):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def initTestCase(self):
        self.dims = (1, 9)
        self.dtype = 'float32'

    def setUp(self):
        self.initTestCase()
        self.__class__.use_mlu = True
        self.place = [paddle.MLUPlace(0)]

    def test_dygraph_api(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def run(place):
            paddle.disable_static(place)
            np.random.seed(2022)
            numpy_input = (np.random.random(self.dims)).astype(self.dtype)
            tensor_input = paddle.to_tensor(numpy_input)
            numpy_output = np.argmax(numpy_input).reshape([1])
            paddle_output = paddle.argmax(tensor_input)
<<<<<<< HEAD
            self.assertEqual(np.allclose(numpy_output, paddle_output.numpy()),
                             True)
=======
            np.testing.assert_allclose(
                numpy_output, paddle_output.numpy(), rtol=1e-05
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            self.assertEqual(numpy_output.shape, paddle_output.numpy().shape)
            paddle.enable_static()

        for place in self.place:
            run(place)


if __name__ == '__main__':
    unittest.main()
