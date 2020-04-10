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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def p_norm(x, axis, porder, keepdims=False):
    xp = np.power(np.abs(x), porder)
    s = np.sum(xp, axis=axis, keepdims=keepdims)
    r = np.power(s, 1.0 / porder)
    return r


def frobenius_norm(x, axis=None, keepdims=False):
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims)
    return r


class APT_NormTest(unittest.TestCase):
    def test_output_result(self):
        with fluid.program_guard(fluid.Program()):
            data1 = fluid.data(name="X", shape=[3, 4], dtype="float32")
            data2 = fluid.data(name="Y", shape=[3], dtype="int64")
            out = paddle.norm(input=data1, p=2, axis=1, out=data2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(
                feed={"X": np.random.rand(3, 4).astype("float32")},
                fetch_list=[data2, out])
            self.assertEqual((result[0] == result[1]).all(), True)
        with fluid.program_guard(fluid.Program()):
            data1 = fluid.data(name="X", shape=[3, 4], dtype="float32")
            data2 = fluid.data(name="Y", shape=[1], dtype="int64")
            out = paddle.norm(input=data1, p='fro', out=data2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(
                feed={"X": np.random.rand(3, 4).astype("float32")},
                fetch_list=[data2, out])
            self.assertEqual((result[0] == result[1]).all(), True)

    def test_basic(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[3, 3, 4], dtype="float32")
            out = paddle.norm(input=data, p='fro')

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = (np.random.rand(3, 3, 4) + 1.0).astype("float32")
            expected_result = frobenius_norm(np_input, axis=(1, 2))

            result, = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[2, 3, 4], dtype="float64")
            out = paddle.norm(input=data, p='fro', axis=[0, 1])

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = np.random.rand(2, 3, 4).astype("float64")
            expected_result = frobenius_norm(np_input, axis=(0, 1))

            result = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[3, 4], dtype="float32")
            out = paddle.norm(input=data, p=2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = np.random.rand(3, 4).astype("float32")
            expected_result = p_norm(
                np_input, porder=2, axis=1).astype(np.float32)

            result = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)

        with fluid.program_guard(fluid.Program()):
            data1 = fluid.data(name="X", shape=[3, 4], dtype="float32")
            data2 = fluid.data(name="Y", shape=[3], dtype="int64")
            out = paddle.norm(input=data, p=2, axis=1, out=data2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(
                feed={"X": np.random.rand(3, 4).astype("float32")},
                fetch_list=[data2, out])
        self.assertEqual((result[0] == result[1]).all(), True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[10, 10], dtype="float32")
            y_1 = paddle.norm(x, p='fro', name='frobenius_name')
            y_2 = paddle.norm(x, p=2, name='pnorm_name')
            self.assertEqual(('frobenius_name' in y_1.name), True)
            self.assertEqual(('pnorm_name' in y_2.name), True)

    def test_errors(self):
        def test_dtype_frobenius():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="int64")
                paddle.norm(data, p='fro')

        self.assertRaises(TypeError, test_dtype_frobenius)

        def test_dtype_frobenius_out():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="foat64")
                out = fluid.data(name="out", shape=[1], dtype="int32")
                paddle.norm(data, p='fro', out=out)

        self.assertRaises(TypeError, test_dtype_frobenius_out)

        def test_dtype_pnorm():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="int64")
                paddle.norm(data, p=2)

        self.assertRaises(TypeError, test_dtype_pnorm)

        def test_dtype_pnorm_out():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="int64")
                out = fluid.data(name="out", shape=[1], dtype="int32")
                paddle.norm(data, p=2, out=out)

        self.assertRaises(TypeError, test_dtype_pnorm_out)

        def test_value_error1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="float64")
                paddle.norm(data, p="unsupport norm")

        self.assertRaises(ValueError, test_value_error1)

        def test_value_error2():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="float64")
                paddle.norm(data, p=[1])

        self.assertRaises(ValueError, test_value_error2)

        def test_value_error3():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="float64")
                paddle.norm(data, p=[1], axis=-1)

        self.assertRaises(ValueError, test_value_error3)

        def test_value_error4():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2], dtype="float64")
                paddle.norm(data, p='unspport matrix norm', axis=[-2, -1])

        self.assertRaises(ValueError, test_value_error4)

        def test_value_error5():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2, 2], dtype="float64")
                paddle.norm(data, p='unspport matrix norm', axis=[-3, -2, -1])

        self.assertRaises(ValueError, test_value_error5)

        def test_value_error_fro_1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[2, 2, 2], dtype="int64")
                paddle.norm(data, p='unspport matrix norm', axis=[-3, -2, -1])

        self.assertRaises(ValueError, test_value_error_fro_1)


if __name__ == '__main__':
    unittest.main()
