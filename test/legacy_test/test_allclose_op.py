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
from utils import dygraph_guard, static_guard

import paddle
from paddle.base import core


class TestAllcloseOp(OpTest):
    def set_args(self):
        self.input = np.array([10000.0, 1e-07]).astype("float32")
        self.other = np.array([10000.1, 1e-08]).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False

    def setUp(self):
        self.set_args()
        self.op_type = "allclose"
        self.python_api = paddle.allclose
        self.inputs = {
            'Input': self.input,
            'Other': self.other,
            "Rtol": self.rtol,
            "Atol": self.atol,
        }
        self.attrs = {'equal_nan': self.equal_nan}
        self.outputs = {
            'Out': np.array(
                np.allclose(
                    self.inputs['Input'],
                    self.inputs['Other'],
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=self.equal_nan,
                )
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAllcloseOpException(TestAllcloseOp):
    def test_check_output(self):
        def test_rtol_num():
            self.inputs['Rtol'] = np.array([1e-05, 1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([1e-08]).astype("float64")
            self.check_output(check_pir=True)

        self.assertRaises(ValueError, test_rtol_num)

        def test_rtol_type():
            self.inputs['Rtol'] = np.array([5]).astype("int32")
            self.inputs['Atol'] = np.array([1e-08]).astype("float64")
            self.check_output(check_pir=True)

        self.assertRaises(ValueError, test_rtol_type)

        def test_atol_num():
            self.inputs['Rtol'] = np.array([1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([1e-08, 1e-08]).astype("float64")
            self.check_output(check_pir=True)

        self.assertRaises(ValueError, test_atol_num)

        def test_atol_type():
            self.inputs['Rtol'] = np.array([1e-05]).astype("float64")
            self.inputs['Atol'] = np.array([8]).astype("int32")
            self.check_output(check_pir=True)

        self.assertRaises(ValueError, test_atol_type)


class TestAllcloseOpSmallNum(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10000.0, 1e-08]).astype("float32")
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
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[10, 10], dtype='complex32'
                )
                y = paddle.static.data(
                    name='y', shape=[10, 10], dtype='float64'
                )
                result = paddle.allclose(x, y)

        self.assertRaises(TypeError, test_x_dtype)

        def test_y_dtype():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[10, 10], dtype='float64'
                )
                y = paddle.static.data(
                    name='y', shape=[10, 10], dtype='complex32'
                )
                result = paddle.allclose(x, y)

        self.assertRaises(TypeError, test_y_dtype)

    def test_attr(self):
        x = paddle.static.data(name='x', shape=[10, 10], dtype='float64')
        y = paddle.static.data(name='y', shape=[10, 10], dtype='float64')

        def test_rtol():
            result = paddle.allclose(x, y, rtol=True)

        self.assertRaises(TypeError, test_rtol)

        def test_atol():
            result = paddle.allclose(x, y, rtol=True)

        self.assertRaises(TypeError, test_atol)

        def test_equal_nan():
            result = paddle.allclose(x, y, equal_nan=1)

        self.assertRaises(TypeError, test_equal_nan)


class TestAllcloseOpFp16(unittest.TestCase):

    def test_fp16(self):
        if core.is_compiled_with_cuda():
            x_data = np.random.rand(10, 10).astype('float16')
            y_data = np.random.rand(10, 10).astype('float16')
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    shape=[10, 10], name='x', dtype='float16'
                )
                y = paddle.static.data(
                    shape=[10, 10], name='y', dtype='float16'
                )
                out = paddle.allclose(x, y, rtol=1e-05, atol=1e-08)
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': x_data, 'y': y_data}, fetch_list=[out])


class TestAllcloseOpFloat16(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10.1]).astype("float16")
        self.other = np.array([10]).astype("float16")
        self.rtol = np.array([0.01]).astype("float64")
        self.atol = np.array([0]).astype("float64")
        self.equal_nan = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, check_pir=True)


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


class TestAllcloseOpBool(unittest.TestCase):
    def test_close_True(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([1]).astype("bool")
                self.other = np.array([1]).astype("bool")
                self.rtol = np.array([0.0]).astype("float32")
                self.atol = np.array([0.0]).astype("float32")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    True,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='bool')
                    y = paddle.static.data(shape=[1], name='y', dtype='bool')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], True)

    def test_close_False(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([0]).astype("bool")
                self.other = np.array([1]).astype("bool")
                self.rtol = np.array([0.0]).astype("float32")
                self.atol = np.array([0.0]).astype("float32")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    False,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='bool')
                    y = paddle.static.data(shape=[1], name='y', dtype='bool')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], False)


class TestAllcloseOpInt32(unittest.TestCase):
    def test_close_True(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([100]).astype("int32")
                self.other = np.array([1]).astype("int32")
                self.rtol = np.array([50.0]).astype("float32")
                self.atol = np.array([49]).astype("float32")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    True,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='int32')
                    y = paddle.static.data(shape=[1], name='y', dtype='int32')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], True)

    def test_close_False(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([100]).astype("int32")
                self.other = np.array([1]).astype("int32")
                self.rtol = np.array([50.0]).astype("float32")
                self.atol = np.array([48]).astype("float32")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    False,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='int32')
                    y = paddle.static.data(shape=[1], name='y', dtype='int32')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], False)


class TestAllcloseOpInt64(unittest.TestCase):
    def test_close_True(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([100]).astype("int64")
                self.other = np.array([1]).astype("int64")
                self.rtol = np.array([50.0]).astype("float64")
                self.atol = np.array([49]).astype("float64")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    True,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='int64')
                    y = paddle.static.data(shape=[1], name='y', dtype='int64')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], True)

    def test_close_False(self):
        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with dygraph_guard():
                # absolute(a−b)≤(atol+rtol×absolute(b))
                self.input = np.array([100]).astype("int64")
                self.other = np.array([1]).astype("int64")
                self.rtol = np.array([50.0]).astype("float64")
                self.atol = np.array([48]).astype("float64")
                self.equal_nan = False
                input = paddle.to_tensor(self.input, place=place)
                other = paddle.to_tensor(self.other, place=place)
                self.assertEqual(
                    paddle.allclose(
                        input, other, self.rtol, self.atol, self.equal_nan
                    ).item(),
                    False,
                )

            with static_guard():
                with paddle.static.program_guard(paddle.static.Program()):
                    x = paddle.static.data(shape=[1], name='x', dtype='int64')
                    y = paddle.static.data(shape=[1], name='y', dtype='int64')
                    out = paddle.allclose(
                        x, y, self.rtol.item(), self.atol.item(), self.equal_nan
                    )
                    exe = paddle.static.Executor(place)
                    exe.run(paddle.static.default_startup_program())
                    out = exe.run(
                        feed={'x': self.input, 'y': self.other},
                        fetch_list=[out],
                    )
                    self.assertEqual(out[0], False)


class TestAllcloseOpLargeDimInput(TestAllcloseOp):
    def set_args(self):
        self.input = np.array(np.zeros([2048, 1024])).astype("float64")
        self.other = np.array(np.zeros([2048, 1024])).astype("float64")
        self.input[-1][-1] = 100
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False


class TestAllcloseOp_ZeroSize(OpTest):
    def set_args(self):
        self.input = np.random.random((2, 0)).astype("float32")
        self.other = np.random.random((2, 0)).astype("float32")
        self.rtol = np.array([1e-05]).astype("float64")
        self.atol = np.array([1e-08]).astype("float64")
        self.equal_nan = False

    def setUp(self):
        self.set_args()
        self.op_type = "allclose"
        self.python_api = paddle.allclose
        self.inputs = {
            'Input': self.input,
            'Other': self.other,
            "Rtol": self.rtol,
            "Atol": self.atol,
        }
        self.attrs = {'equal_nan': self.equal_nan}
        self.outputs = {
            'Out': np.array(
                np.allclose(
                    self.inputs['Input'],
                    self.inputs['Other'],
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=self.equal_nan,
                )
            )
        }

    def test_check_output(self):
        self.check_output(check_pir=True)


if __name__ == "__main__":
    unittest.main()
