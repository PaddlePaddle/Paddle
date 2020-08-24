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
import paddle
import paddle.tensor as tensor
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import numpy as np
import unittest


class TestMultiplyAPI(unittest.TestCase):
    """TestMultiplyAPI."""

    def __run_static_graph_case(self, x_data, y_data, axis=-1):
        with program_guard(Program(), Program()):
            x = paddle.static.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype)
            y = paddle.static.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype)
            res = tensor.multiply(x, y, axis=axis)

            place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            outs = exe.run(fluid.default_main_program(),
                           feed={'x': x_data,
                                 'y': y_data},
                           fetch_list=[res])
            res = outs[0]
            return res

    def __run_dynamic_graph_case(self, x_data, y_data, axis=-1):
        paddle.disable_static()
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        res = paddle.multiply(x, y, axis=axis)
        return res.numpy()

    def test_multiply(self):
        """test_multiply."""
        np.random.seed(7)
        # test static computation graph: 1-d array
        x_data = np.random.rand(200)
        y_data = np.random.rand(200)
        res = self.__run_static_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test static computation graph: 2-d array
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(2, 500)
        res = self.__run_static_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test static computation graph: broadcast
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(500)
        res = self.__run_static_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test static computation graph: broadcast with axis
        x_data = np.random.rand(2, 300, 40)
        y_data = np.random.rand(300)
        res = self.__run_static_graph_case(x_data, y_data, axis=1)
        expected = np.multiply(x_data, y_data[..., np.newaxis])
        self.assertTrue(np.allclose(res, expected))

        # test dynamic computation graph: 1-d array
        x_data = np.random.rand(200)
        y_data = np.random.rand(200)
        res = self.__run_dynamic_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test dynamic computation graph: 2-d array
        x_data = np.random.rand(20, 50)
        y_data = np.random.rand(20, 50)
        res = self.__run_dynamic_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test dynamic computation graph: broadcast
        x_data = np.random.rand(2, 500)
        y_data = np.random.rand(500)
        res = self.__run_dynamic_graph_case(x_data, y_data)
        self.assertTrue(np.allclose(res, np.multiply(x_data, y_data)))

        # test dynamic computation graph: broadcast with axis
        x_data = np.random.rand(2, 300, 40)
        y_data = np.random.rand(300)
        res = self.__run_dynamic_graph_case(x_data, y_data, axis=1)
        expected = np.multiply(x_data, y_data[..., np.newaxis])
        self.assertTrue(np.allclose(res, expected))


class TestMultiplyError(unittest.TestCase):
    """TestMultiplyError."""

    def test_errors(self):
        """test_errors."""
        # test static computation graph: dtype can not be int8
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[100], dtype=np.int8)
            y = paddle.static.data(name='y', shape=[100], dtype=np.int8)
            self.assertRaises(TypeError, tensor.multiply, x, y)

        # test static computation graph: inputs must be broadcastable 
        with program_guard(Program(), Program()):
            x = paddle.static.data(name='x', shape=[20, 50], dtype=np.float64)
            y = paddle.static.data(name='y', shape=[20], dtype=np.float64)
            self.assertRaises(fluid.core.EnforceNotMet, tensor.multiply, x, y)

        np.random.seed(7)
        # test dynamic computation graph: dtype can not be int8
        paddle.disable_static()
        x_data = np.random.randn(200).astype(np.int8)
        y_data = np.random.randn(200).astype(np.int8)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        self.assertRaises(fluid.core.EnforceNotMet, paddle.multiply, x, y)

        # test dynamic computation graph: inputs must be broadcastable
        x_data = np.random.rand(200, 5)
        y_data = np.random.rand(200)
        x = paddle.to_variable(x_data)
        y = paddle.to_variable(y_data)
        self.assertRaises(fluid.core.EnforceNotMet, paddle.multiply, x, y)


if __name__ == '__main__':
    unittest.main()
