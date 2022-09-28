#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard

paddle.enable_static()

# Correct: General.


class TestSqueezeOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "squeeze"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(paddle.NPUPlace(0))

    def test_check_grad(self):
        self.check_grad_with_place(paddle.NPUPlace(0), ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.


class TestSqueezeOp1(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, -2)
        self.new_shape = (3, 40)


# Correct: No axes input.


class TestSqueezeOp2(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.


class TestSqueezeOp3(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


# Correct: The demension of axis is not of size 1 remains unchanged.


class TestSqueezeOp4(TestSqueezeOp):

    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, 2)
        self.new_shape = (6, 5, 1, 4, 1)


class TestSqueezeOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of softmax_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         paddle.NPUPlace(0))
            self.assertRaises(TypeError, paddle.squeeze, x1)
            # The input axes of squeeze must be list.
            x2 = paddle.static.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, paddle.squeeze, x2, axes=0)
            # The input dtype of squeeze not support float16.
            x3 = paddle.static.data(name='x3', shape=[4], dtype="float16")
            self.assertRaises(TypeError, paddle.squeeze, x3, axes=0)


class API_TestSqueeze(unittest.TestCase):

    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data1 = paddle.static.data('data1',
                                       shape=[-1, 1, 10],
                                       dtype='float64')
            result_squeeze = self.squeeze(data1, axis=[1])
            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            input1 = np.random.random([5, 1, 10]).astype('float64')
            result, = exe.run(feed={"data1": input1},
                              fetch_list=[result_squeeze])
            expected_result = np.squeeze(input1, axis=1)
            np.testing.assert_allclose(expected_result, result)


class API_TestStaticSqueeze_(API_TestSqueeze):

    def executed_api(self):
        self.squeeze = paddle.squeeze_


class API_TestDygraphSqueeze(unittest.TestCase):

    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_out(self):
        paddle.disable_static()
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = paddle.to_tensor(input_1)
            output = self.squeeze(input, axis=[1])
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            np.testing.assert_allclose(expected_out, out_np)

    def test_out_int8(self):
        paddle.disable_static()
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            input_1 = np.random.random([5, 1, 10]).astype("int8")
            input = paddle.to_tensor(input_1)
            output = self.squeeze(input, axis=[1])
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            np.testing.assert_allclose(expected_out, out_np)

    def test_out_uint8(self):
        paddle.disable_static()
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            input_1 = np.random.random([5, 1, 10]).astype("uint8")
            input = paddle.to_tensor(input_1)
            output = self.squeeze(input, axis=[1])
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            np.testing.assert_allclose(expected_out, out_np)

    def test_axis_not_list(self):
        paddle.disable_static()
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = paddle.to_tensor(input_1)
            output = self.squeeze(input, axis=1)
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            np.testing.assert_allclose(expected_out, out_np)

    def test_dimension_not_1(self):
        paddle.disable_static()
        with fluid.dygraph.guard(paddle.NPUPlace(0)):
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = paddle.to_tensor(input_1)
            output = self.squeeze(input, axis=(1, 0))
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            np.testing.assert_allclose(expected_out, out_np)


class API_TestDygraphSqueezeInplace(API_TestDygraphSqueeze):

    def executed_api(self):
        self.squeeze = paddle.squeeze_


# Correct: General.
class TestSqueeze2Op(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "squeeze2"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float32")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(paddle.NPUPlace(0),
                                     no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(paddle.NPUPlace(0), ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueeze2Op1(TestSqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


# Correct: No axes input.
class TestSqueeze2Op2(TestSqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.
class TestSqueeze2Op3(TestSqueeze2Op):

    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


if __name__ == "__main__":
    unittest.main()
