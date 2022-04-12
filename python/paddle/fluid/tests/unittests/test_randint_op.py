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

import os
import paddle
import unittest
import numpy as np
from op_test import OpTest
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard
from paddle.static import program_guard, Program

paddle.enable_static()


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

    def test_check_output_eager(self):
        with _test_eager_guard():
            self.test_check_output()


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

    def test_errors_eager(self):
        with _test_eager_guard():
            self.test_errors()


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

    def test_check_output_eager(self):
        with _test_eager_guard():
            self.test_check_output()


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

    def test_check_output_eager(self):
        with _test_eager_guard():
            self.test_check_output()


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

    def test_api_eager(self):
        with _test_eager_guard():
            self.test_api()


class TestRandintImperative(unittest.TestCase):
    def test_api(self):
        paddle.disable_static()

        self.run_test_case()

        with _test_eager_guard():
            self.run_test_case()

        paddle.enable_static()

    def run_test_case(self):
        n = 10
        x1 = paddle.randint(n, shape=[10], dtype="int32")
        x2 = paddle.tensor.randint(n)
        x3 = paddle.tensor.random.randint(n)
        for i in [x1, x2, x3]:
            for j in i.numpy().tolist():
                self.assertTrue((j >= 0 and j < n))


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generatte different random value. Only test V100 here.
        if not "V100" in paddle.device.cuda.get_device_name():
            return

        print("Test Fixed Random number on GPU------>")
        paddle.disable_static()

        self.run_test_case()

        with _test_eager_guard():
            self.run_test_case()

        paddle.enable_static()

    def run_test_case(self):
        paddle.set_device('gpu')
        paddle.seed(100)

        x = paddle.randint(
            -10000, 10000, [32, 3, 1024, 1024], dtype='int32').numpy()
        self.assertTrue(x.mean(), -0.7517569760481516)
        self.assertTrue(x.std(), 5773.696619107639)
        expect = [2535, 2109, 5916, -5011, -261]
        self.assertTrue(np.array_equal(x[10, 0, 100, 100:105], expect))
        expect = [3465, 7206, -8660, -9628, -6574]
        self.assertTrue(np.array_equal(x[20, 1, 600, 600:605], expect))
        expect = [881, 1560, 1100, 9664, 1669]
        self.assertTrue(np.array_equal(x[30, 2, 1000, 1000:1005], expect))

        x = paddle.randint(
            -10000, 10000, [32, 3, 1024, 1024], dtype='int64').numpy()
        self.assertTrue(x.mean(), -1.461287518342336)
        self.assertTrue(x.std(), 5773.023477548159)
        expect = [7213, -9597, 754, 8129, -1158]
        self.assertTrue(np.array_equal(x[10, 0, 100, 100:105], expect))
        expect = [-7159, 8054, 7675, 6980, 8506]
        self.assertTrue(np.array_equal(x[20, 1, 600, 600:605], expect))
        expect = [3581, 3420, -8027, -5237, -2436]
        self.assertTrue(np.array_equal(x[30, 2, 1000, 1000:1005], expect))


if __name__ == "__main__":
    unittest.main()
