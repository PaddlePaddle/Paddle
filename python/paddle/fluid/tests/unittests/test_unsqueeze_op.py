# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from op_test import OpTest, convert_float_to_uint16
import paddle.fluid.core as core
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers

paddle.enable_static()


# Correct: General.
class TestUnsqueezeOp(OpTest):

    def setUp(self):
        self.init_test_case()
        self.op_type = "unsqueeze"
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (1, 2)
        self.new_shape = (3, 1, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


class TestUnsqueezeBF16Op(OpTest):

    def setUp(self):
        self.init_test_case()
        self.op_type = "unsqueeze"
        self.dtype = np.uint16
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
        self.ori_shape = (3, 40)
        self.axes = (1, 2)
        self.new_shape = (3, 1, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: Single input index.
class TestUnsqueezeOp1(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (-1, )
        self.new_shape = (20, 5, 1)


# Correct: Mixed input axis.
class TestUnsqueezeOp2(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -1)
        self.new_shape = (1, 20, 5, 1)


# Correct: There is duplicated axis.
class TestUnsqueezeOp3(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (0, 3, 3)
        self.new_shape = (1, 10, 2, 1, 1, 5)


# Correct: Reversed axes.
class TestUnsqueezeOp4(TestUnsqueezeOp):

    def init_test_case(self):
        self.ori_shape = (10, 2, 5)
        self.axes = (3, 1, 1)
        self.new_shape = (10, 1, 1, 2, 5, 1)


class API_TestUnsqueeze(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 10], dtype='float64')
            result_squeeze = paddle.unsqueeze(data1, axis=[1])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([5, 1, 10]).astype('float64')
            input = np.squeeze(input1, axis=1)
            result, = exe.run(feed={"data1": input},
                              fetch_list=[result_squeeze])
            np.testing.assert_allclose(input1, result, rtol=1e-05)


class TestUnsqueezeOpError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            # The type of axis in split_op should be int or Variable.
            def test_axes_type():
                x6 = paddle.static.data(shape=[-1, 10],
                                        dtype='float16',
                                        name='x3')
                paddle.unsqueeze(x6, axis=3.2)

            self.assertRaises(TypeError, test_axes_type)


class API_TestUnsqueeze2(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 10], dtype='float64')
            data2 = paddle.static.data('data2', shape=[1], dtype='int32')
            result_squeeze = paddle.unsqueeze(data1, axis=data2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([5, 1, 10]).astype('float64')
            input2 = np.array([1]).astype('int32')
            input = np.squeeze(input1, axis=1)
            result1, = exe.run(feed={
                "data1": input,
                "data2": input2
            },
                               fetch_list=[result_squeeze])
            np.testing.assert_allclose(input1, result1, rtol=1e-05)


class API_TestUnsqueeze3(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data1 = paddle.static.data('data1', shape=[-1, 10], dtype='float64')
            data2 = paddle.static.data('data2', shape=[1], dtype='int32')
            result_squeeze = paddle.unsqueeze(data1, axis=[data2, 3])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([5, 1, 10, 1]).astype('float64')
            input2 = np.array([1]).astype('int32')
            input = np.squeeze(input1)
            result1, = exe.run(feed={
                "data1": input,
                "data2": input2
            },
                               fetch_list=[result_squeeze])
            np.testing.assert_array_equal(input1, result1)
            self.assertEqual(input1.shape, result1.shape)


class API_TestDyUnsqueeze(unittest.TestCase):

    def test_out(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input1 = np.expand_dims(input_1, axis=1)
        input = paddle.to_tensor(input_1)
        output = paddle.unsqueeze(input, axis=[1])
        out_np = output.numpy()
        np.testing.assert_array_equal(input1, out_np)
        self.assertEqual(input1.shape, out_np.shape)


class API_TestDyUnsqueeze2(unittest.TestCase):

    def test_out(self):
        paddle.disable_static()
        input1 = np.random.random([5, 10]).astype("int32")
        out1 = np.expand_dims(input1, axis=1)
        input = paddle.to_tensor(input1)
        output = paddle.unsqueeze(input, axis=1)
        out_np = output.numpy()
        np.testing.assert_array_equal(out1, out_np)
        self.assertEqual(out1.shape, out_np.shape)


class API_TestDyUnsqueezeAxisTensor(unittest.TestCase):

    def test_out(self):
        paddle.disable_static()
        input1 = np.random.random([5, 10]).astype("int32")
        out1 = np.expand_dims(input1, axis=1)
        out1 = np.expand_dims(out1, axis=2)
        input = paddle.to_tensor(input1)
        output = paddle.unsqueeze(input, axis=paddle.to_tensor([1, 2]))
        out_np = output.numpy()
        np.testing.assert_array_equal(out1, out_np)
        self.assertEqual(out1.shape, out_np.shape)


class API_TestDyUnsqueezeAxisTensorList(unittest.TestCase):

    def test_out(self):
        paddle.disable_static()
        input1 = np.random.random([5, 10]).astype("int32")
        # Actually, expand_dims supports tuple since version 1.18.0
        out1 = np.expand_dims(input1, axis=1)
        out1 = np.expand_dims(out1, axis=2)
        input = paddle.to_tensor(input1)
        output = paddle.unsqueeze(
            paddle.to_tensor(input1),
            axis=[paddle.to_tensor([1]),
                  paddle.to_tensor([2])])
        out_np = output.numpy()
        np.testing.assert_array_equal(out1, out_np)
        self.assertEqual(out1.shape, out_np.shape)


class API_TestDygraphUnSqueeze(unittest.TestCase):

    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.unsqueeze = paddle.unsqueeze

    def test_out(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.unsqueeze(input, axis=[1])
        out_np = output.numpy()
        expected_out = np.expand_dims(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_int8(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int8")
        input = paddle.to_tensor(input_1)
        output = self.unsqueeze(input, axis=[1])
        out_np = output.numpy()
        expected_out = np.expand_dims(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_out_uint8(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("uint8")
        input = paddle.to_tensor(input_1)
        output = self.unsqueeze(input, axis=1)
        out_np = output.numpy()
        expected_out = np.expand_dims(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_axis_not_list(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.unsqueeze(input, axis=1)
        out_np = output.numpy()
        expected_out = np.expand_dims(input_1, axis=1)
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)

    def test_dimension_not_1(self):
        paddle.disable_static()
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        input = paddle.to_tensor(input_1)
        output = self.unsqueeze(input, axis=(1, 2))
        out_np = output.numpy()
        expected_out = np.expand_dims(input_1, axis=(1, 2))
        np.testing.assert_allclose(expected_out, out_np, rtol=1e-05)


class API_TestDygraphUnSqueezeInplace(API_TestDygraphUnSqueeze):

    def executed_api(self):
        self.unsqueeze = paddle.unsqueeze_


class TestUnsqueezeDoubleGradCheck(unittest.TestCase):

    def unsqueeze_wrapper(self, x):
        return paddle.unsqueeze(x[0], [0, 2])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 3, 4], False, dtype)
        data.persistable = True
        out = paddle.unsqueeze(data, [0, 2])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.unsqueeze_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestUnsqueezeTripleGradCheck(unittest.TestCase):

    def unsqueeze_wrapper(self, x):
        return paddle.unsqueeze(x[0], [0, 2])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 3, 4], False, dtype)
        data.persistable = True
        out = paddle.unsqueeze(data, [0, 2])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.unsqueeze_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
