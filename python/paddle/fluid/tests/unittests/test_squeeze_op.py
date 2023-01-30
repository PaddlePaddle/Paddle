#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
=======
from __future__ import print_function
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from op_test import OpTest, convert_float_to_uint16
import paddle.fluid.core as core
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "squeeze"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


class TestSqueezeBF16Op(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "squeeze"
        self.dtype = np.uint16
        self.init_test_case()
        x = np.random.random(self.ori_shape).astype("float32")
        out = x.reshape(self.new_shape)
        self.inputs = {"X": convert_float_to_uint16(x)}
        self.init_attrs()
        self.outputs = {"Out": convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, -2)
        self.new_shape = (3, 40)


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.
class TestSqueezeOp3(TestSqueezeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


# Correct: The demension of axis is not of size 1 remains unchanged.
class TestSqueezeOp4(TestSqueezeOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, 2)
        self.new_shape = (6, 5, 1, 4, 1)


class TestSqueezeOpError(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The input type of softmax_op must be Variable.
<<<<<<< HEAD
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], paddle.CPUPlace()
            )
=======
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         paddle.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, paddle.squeeze, x1)
            # The input axes of squeeze must be list.
            x2 = paddle.static.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, paddle.squeeze, x2, axes=0)
            # The input dtype of squeeze not support float16.
            x3 = paddle.static.data(name='x3', shape=[4], dtype="float16")
            self.assertRaises(TypeError, paddle.squeeze, x3, axes=0)


class API_TestSqueeze(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_out(self):
        paddle.enable_static()
<<<<<<< HEAD
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data1 = paddle.static.data(
                'data1', shape=[-1, 1, 10], dtype='float64'
            )
=======
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data1 = paddle.static.data('data1',
                                       shape=[-1, 1, 10],
                                       dtype='float64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            result_squeeze = self.squeeze(data1, axis=[1])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([5, 1, 10]).astype('float64')
<<<<<<< HEAD
            (result,) = exe.run(
                feed={"data1": input1}, fetch_list=[result_squeeze]
            )
=======
            result, = exe.run(feed={"data1": input1},
                              fetch_list=[result_squeeze])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            expected_result = np.squeeze(input1, axis=1)
            np.testing.assert_allclose(expected_result, result, rtol=1e-05)


class API_TestStaticSqueeze_(API_TestSqueeze):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def executed_api(self):
        self.squeeze = paddle.squeeze_


class API_TestDygraphSqueeze(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_out(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.squeeze(input, axis=[1])
        out_np = output.numpy()
        expected_out = np.squeeze(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_int8(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int8")
        input = paddle.to_tensor(input_1)
        output = self.squeeze(input, axis=[1])
        out_np = output.numpy()
        expected_out = np.squeeze(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_uint8(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("uint8")
        input = paddle.to_tensor(input_1)
        output = self.squeeze(input, axis=[1])
        out_np = output.numpy()
        expected_out = np.squeeze(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_axis_not_list(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.squeeze(input, axis=1)
        out_np = output.numpy()
        expected_out = np.squeeze(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_dimension_not_1(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.squeeze(input, axis=(1, 0))
        out_np = output.numpy()
        expected_out = np.squeeze(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)


class API_TestDygraphSqueezeInplace(API_TestDygraphSqueeze):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def executed_api(self):
        self.squeeze = paddle.squeeze_


class TestSqueezeDoubleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def squeeze_wrapper(self, x):
        return paddle.squeeze(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

<<<<<<< HEAD
        data = paddle.static.data('data', [2, 3], dtype)
=======
        data = layers.data('data', [2, 3], False, dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        data.persistable = True
        out = paddle.squeeze(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

<<<<<<< HEAD
        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.squeeze_wrapper, [data], out, x_init=[data_arr], place=place
        )
=======
        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.squeeze_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestSqueezeTripleGradCheck(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def squeeze_wrapper(self, x):
        return paddle.squeeze(x[0])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

<<<<<<< HEAD
        data = paddle.static.data('data', [2, 3], dtype)
=======
        data = layers.data('data', [2, 3], False, dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        data.persistable = True
        out = paddle.squeeze(data)
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

<<<<<<< HEAD
        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.squeeze_wrapper, [data], out, x_init=[data_arr], place=place
        )
=======
        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.squeeze_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
