# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
from paddle.fluid import core
from paddle.static import program_guard, Program


def output_hist(out):
    hist, _ = np.histogram(out, range=(-10, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestRandintOp(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"shape": [10000, 784], "low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


class TestRandintOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            self.assertRaises(TypeError, paddle.randint, 5, shape=np.array([2]))
            self.assertRaises(TypeError, paddle.randint, 5, dtype='float32')
            self.assertRaises(ValueError, paddle.randint, 5, 5)
            self.assertRaises(ValueError, paddle.randint, -5)
            self.assertRaises(TypeError, paddle.randint, 5, shape=['2'])
            shape_tensor = paddle.static.data('X', [1])
            self.assertRaises(TypeError, paddle.randint, 5, shape=shape_tensor)
            self.assertRaises(
                TypeError, paddle.randint, 5, shape=[shape_tensor])


class TestRandintOp_attr_tensorlist(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.new_shape = (10000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype("int64") * ele))
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int32")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


class TestRandint_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.inputs = {"ShapeTensor": np.array([10000, 784]).astype("int64")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int64")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


# Test python API
class TestRandintAPI(unittest.TestCase):
    def test_api(self):
        with program_guard(Program(), Program()):
            # results are from [0, 5).
            out1 = paddle.randint(5)
            # shape is a list and dtype is 'int32'
            out2 = paddle.randint(
                low=-100, high=100, shape=[64, 64], dtype='int32')
            # shape is a tuple and dtype is 'int64'
            out3 = paddle.randint(
                low=-100, high=100, shape=(32, 32, 3), dtype='int64')
            # shape is a tensorlist and dtype is 'float32'
            dim_1 = paddle.fluid.layers.fill_constant([1], "int64", 32)
            dim_2 = paddle.fluid.layers.fill_constant([1], "int32", 50)
            out4 = paddle.randint(
                low=-100, high=100, shape=[dim_1, 5, dim_2], dtype='int32')
            # shape is a tensor and dtype is 'float64'
            var_shape = paddle.static.data(
                name='var_shape', shape=[2], dtype="int64")
            out5 = paddle.randint(
                low=1, high=1000, shape=var_shape, dtype='int64')

            place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            outs = exe.run(
                feed={'var_shape': np.array([100, 100]).astype('int64')},
                fetch_list=[out1, out2, out3, out4, out5])


class TestRandintLikeOp(OpTest):
    def setUp(self):
        self.op_type = "randint_like"
        x = np.zeros((10000, 784)).astype("float32")
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


# Test python API
class TestRandintLikeAPI(unittest.TestCase):
    def setUp(self):
        self.x_int32 = np.zeros((10, 12)).astype('int32')
        self.x_float32 = np.zeros((10, 12)).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # results are from [-100, 100).
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[10, 12], dtype='int32')
            x_float32 = paddle.fluid.data(
                name='x_float32', shape=[10, 12], dtype='float32')
            # x dtype is int32 and output dtype is 'int32'
            out1 = paddle.randint_like(
                x_int32, low=-100, high=100, dtype='int32')
            # x dtype is int32 and output dtype is 'int64'
            out2 = paddle.randint_like(
                x_int32, low=-100, high=100, dtype='int64')
            # x dtype is float32 and output dtype is 'int32'
            out3 = paddle.randint_like(
                x_float32, low=-100, high=100, dtype='int32')
            # x dtype is float32 and output dtype is 'int64'
            out4 = paddle.randint_like(
                x_float32, low=-100, high=100, dtype='int64')

            exe = paddle.static.Executor(self.place)
            outs_int32 = exe.run(feed={'X': self.x_int32},
                                 fetch_list=[out1, out2])
            outs_float32 = exe.run(feed={'X': self.x_float32},
                                   fetch_list=[out3, out4])

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x_int32 = paddle.to_tensor(self.x_int32)
        x_float32 = paddle.to_tensor(self.x_float32)
        # x dtype is int32 and output dtype is 'int32'
        out1 = paddle.randint_like(x_int32, low=-100, high=100, dtype='int32')
        # x dtype is int32 and output dtype is 'int64'
        out2 = paddle.randint_like(x_int32, low=-100, high=100, dtype='int64')
        # x dtype is float32 and output dtype is 'int32'
        out3 = paddle.randint_like(x_float32, low=-100, high=100, dtype='int32')
        # x dtype is float32 and output dtype is 'int64'
        out4 = paddle.randint_like(x_float32, low=-100, high=100, dtype='int64')
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(
                TypeError,
                paddle.randint_like,
                x_int32,
                high=5,
                dtype='float32')
            self.assertRaises(
                ValueError, paddle.randint_like, x_int32, low=5, high=5)
            self.assertRaises(ValueError, paddle.randint_like, x_int32, high=-5)


class TestRandintImperative(unittest.TestCase):
    def test_api(self):
        n = 10
        paddle.disable_static()
        x1 = paddle.randint(n, shape=[10], dtype="int32")
        x2 = paddle.tensor.randint(n)
        x3 = paddle.tensor.random.randint(n)
        for i in [x1, x2, x3]:
            for j in i.numpy().tolist():
                self.assertTrue((j >= 0 and j < n))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
