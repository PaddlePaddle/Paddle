#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
import paddle


class TestAllcloseOp(OpTest):
    def set_args(self):
        self.input = np.array([10000., 1e-07]).astype("float32")
        self.other = np.array([10000.1, 1e-08]).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False

    def setUp(self):
        self.set_args()
        self.op_type = "allclose"
        self.inputs = {
            'Input': self.input,
            'Other': self.other,
            "Rtol": self.rtol,
            "Atol": self.atol
        }
        self.attrs = {'equal_nan': self.equal_nan}
        self.outputs = {
            'Out': np.array([
                np.allclose(
                    self.inputs['Input'],
                    self.inputs['Other'],
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=self.equal_nan)
            ])
        }

    def test_check_output(self):
        self.check_output()


class TestAllcloseOpException(TestAllcloseOp):
    def test_check_output(self):
        def test_rtol_num():
            self.inputs['Rtol'] = np.array([1e-05, 1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([1e-08]).astype("float64")
            self.check_output()

        self.assertRaises(ValueError, test_rtol_num)

        def test_rtol_type():
            self.inputs['Rtol'] = np.array([5]).astype("int32")
            self.inputs['Atol'] = np.array([1e-08]).astype("float64")
            self.check_output()

        self.assertRaises(ValueError, test_rtol_type)

        def test_atol_num():
            self.inputs['Rtol'] = np.array([1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([1e-08, 1e-08]).astype("float64")
            self.check_output()

        self.assertRaises(ValueError, test_atol_num)

        def test_atol_type():
            self.inputs['Rtol'] = np.array([1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([8]).astype("int32")
            self.check_output()

        self.assertRaises(ValueError, test_atol_type)


class TestAllcloseOpSmallNum(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10000., 1e-08]).astype("float32")
        self.other = np.array([10000.1, 1e-09]).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False


class TestAllcloseOpNanFalse(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([1.0, float('nan')]).astype("float32")
        self.other = np.array([1.0, float('nan')]).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False


class TestAllcloseOpNanTrue(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([1.0, float('nan')]).astype("float32")
        self.other = np.array([1.0, float('nan')]).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = True


class TestAllcloseDygraph(unittest.TestCase):
    def test_api_case(self):
        paddle.disable_static()
        x_data = np.random.rand(10, 10)
        y_data = np.random.rand(10, 10)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.allclose(x, y, rtol=1e-05, atol=1e-08)
        expected_out = np.allclose(x_data, y_data, rtol=1e-05, atol=1e-08)
        self.assertTrue((out.numpy() == expected_out).all(), True)
        paddle.enable_static()


class TestAllcloseError(unittest.TestCase):
    def test_input_dtype(self):
        def test_x_dtype():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float16')
                y = paddle.fluid.data(name='y', shape=[10, 10], dtype='float64')
                result = paddle.allclose(x, y)

        self.assertRaises(TypeError, test_x_dtype)

        def test_y_dtype():
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')
                y = paddle.fluid.data(name='y', shape=[10, 10], dtype='int32')
                result = paddle.allclose(x, y)

        self.assertRaises(TypeError, test_y_dtype)

    def test_attr(self):
        x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')
        y = paddle.fluid.data(name='y', shape=[10, 10], dtype='float64')

        def test_rtol():
            result = paddle.allclose(x, y, rtol=True)

        self.assertRaises(TypeError, test_rtol)

        def test_atol():
            result = paddle.allclose(x, y, rtol=True)

        self.assertRaises(TypeError, test_atol)

        def test_equal_nan():
            result = paddle.allclose(x, y, equal_nan=1)

        self.assertRaises(TypeError, test_equal_nan)


class TestAllcloseOpFloat32(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10.1]).astype("float32")
        self.other = np.array([10]).astype("float32")
        self.rtol = np.array([0.01]).astype("float64")
        self.atol = np.array([0]).astype("float64")
        self.equal_nan = False


class TestAllcloseOpFloat64(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10.1]).astype("float64")
        self.other = np.array([10]).astype("float64")
        self.rtol = np.array([0.01]).astype("float64")
        self.atol = np.array([0]).astype("float64")
        self.equal_nan = False


class TestAllcloseOpLargeDimInput(TestAllcloseOp):
    def set_args(self):
        self.input = np.array(np.zeros([2048, 1024])).astype("float64")
        self.other = np.array(np.zeros([2048, 1024])).astype("float64")
        self.input[-1][-1] = 100
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False


if __name__ == "__main__":
    unittest.main()
