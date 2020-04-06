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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.compat as cpt
import unittest
import numpy as np
from op_test import OpTest


class TestFillAnyLikeOp(OpTest):
    def setUp(self):
        self.op_type = "fill_any_like"
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {'value': self.value}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs["X"])}

    def init(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestFillAnyLikeOpFloat32(TestFillAnyLikeOp):
    def init(self):
        self.dtype = np.float32
        self.value = 0.0


class TestFillAnyLikeOpValue1(TestFillAnyLikeOp):
    def init(self):
        self.value = 1.0


class TestFillAnyLikeOpValue2(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e-10


class TestFillAnyLikeOpValue3(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e-100


class TestFillAnyLikeOpType(TestFillAnyLikeOp):
    def setUp(self):
        self.op_type = "fill_any_like"
        self.dtype = np.int32
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((219, 232)).astype(self.dtype)}
        self.attrs = {
            'value': self.value,
            'dtype': int(core.VarDesc.VarType.FP32)
        }
        self.outputs = {
            'Out':
            self.value * np.ones_like(self.inputs["X"]).astype(np.float32)
        }


class TestFillAnyLikeOpOverflow(TestFillAnyLikeOp):
    def init(self):
        self.value = 1e100

    def test_check_output(self):
        exception = None
        try:
            self.check_output(check_dygraph=False)
        except core.EnforceNotMet as ex:
            exception = ex
        self.assertIsNotNone(exception)


class TestFillAnyLikeOpFloat16(TestFillAnyLikeOp):
    def init(self):
        self.dtype = np.float16


class ApiOnesLikeTest(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            ones = paddle.ones_like(data, device="cpu")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[ones])
            expected_result = np.ones(10, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            ones = paddle.ones_like(data, device="cpu", dtype="float32")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[ones])
            expected_result = np.ones(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            ones = paddle.ones_like(data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[ones])
            expected_result = np.ones(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)


class ApiZerosLikeTest(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            zeros = paddle.zeros_like(data, device="cpu")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            zeros = paddle.zeros_like(data, device="cpu", dtype="float32")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            zeros = paddle.zeros_like(data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"data": np.random.rand(10)},
                              fetch_list=[zeros])
            expected_result = np.zeros(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)


class TestOnesZerosError(unittest.TestCase):
    def test_errors(self):
        def test_device_error1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float32")
                paddle.ones_like(data, device="opu")

        self.assertRaises(ValueError, test_device_error1)

        def test_device_error2():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float32")
                paddle.ones_like(data, dtype="float")

        self.assertRaises(ValueError, test_device_error2)

        def test_device_error3():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float32")
                paddle.zeros_like(data, device="opu")

        self.assertRaises(ValueError, test_device_error3)

        def test_device_error4():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float32")
                paddle.zeros_like(data, dtype="float")

        self.assertRaises(ValueError, test_device_error4)


if __name__ == "__main__":
    unittest.main()
