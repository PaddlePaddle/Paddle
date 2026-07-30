#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
    paddle_static_guard,
)

import paddle
from paddle import base
from paddle.base import core


class TestBmmOp(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.python_api = paddle.Tensor.bmm
        self.public_python_api = paddle.Tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float64")
        Y = np.random.random((10, 4, 5)).astype("float64")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


class TestBmmFP16Op(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.dtype = np.float16
        self.python_api = paddle.Tensor.bmm
        self.public_python_api = paddle.Tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float16")
        Y = np.random.random((10, 4, 5)).astype("float16")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestBmmBF16Op(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.prim_op_type = "comp"
        self.dtype = np.uint16
        self.python_api = paddle.Tensor.bmm
        self.public_python_api = paddle.Tensor.bmm
        X = np.random.random((10, 3, 4)).astype("float32")
        Y = np.random.random((10, 4, 5)).astype("float32")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.inputs['Y'] = convert_float_to_uint16(self.inputs['Y'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = get_device_place()

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_pir=True, check_prim_pir=True
        )

    def test_checkout_grad(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'], 'Out', check_pir=True
        )


class API_TestBmm(unittest.TestCase):
    def test_out(self):
        with paddle_static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                data1 = paddle.static.data(
                    'data1', shape=[-1, 3, 4], dtype='float64'
                )
                data2 = paddle.static.data(
                    'data2', shape=[-1, 4, 5], dtype='float64'
                )
                result_bmm = paddle.bmm(data1, data2)
                place = base.CPUPlace()
                exe = base.Executor(place)
                input1 = np.random.random([10, 3, 4]).astype('float64')
                input2 = np.random.random([10, 4, 5]).astype('float64')
                (result,) = exe.run(
                    feed={"data1": input1, "data2": input2},
                    fetch_list=[result_bmm],
                )
                expected_result = np.matmul(input1, input2)
            np.testing.assert_allclose(expected_result, result, rtol=1e-05)


class API_TestDygraphBmm(unittest.TestCase):
    def test_out(self):
        input1 = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ]
        )
        input2 = np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
            ]
        )
        with base.dygraph.guard():
            x = paddle.to_tensor(input1)
            y = paddle.to_tensor(input2)
            out = paddle.bmm(x, y)
            out_np = out.numpy()
        expected_result = np.matmul(input1, input2)
        np.testing.assert_allclose(expected_result, out_np, rtol=1e-05)

    def test_legacy_linalg_entry(self):
        x = paddle.randn([2, 3, 4])
        y = paddle.randn([2, 4, 5])
        out = paddle.tensor.linalg.bmm(x, y)
        expected = paddle.bmm(x, y)

        self.assertIs(paddle.tensor.linalg.bmm, paddle.bmm)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-05)


class TestBmmAPIError(unittest.TestCase):
    def test_api_error(self):
        x_data = np.arange(24, dtype='float32').reshape((2, 3, 4))
        y_data = np.arange(16, dtype='float32').reshape((2, 4, 2))
        y_data_wrong1 = np.arange(16, dtype='float32').reshape((2, 2, 4))
        y_data_wrong2 = np.arange(16, dtype='float32').reshape((2, 2, 2, 2))
        y_data_wrong3 = np.arange(24, dtype='float32').reshape((3, 4, 2))
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong1)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong2)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong3)


class TestBmmOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        self.python_api = paddle.bmm
        self.public_python_api = paddle.bmm
        X = np.random.random((10, 0, 4)).astype("float64")
        Y = np.random.random((10, 4, 5)).astype("float64")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)


class TestBmmOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.random((10, 3, 4)).astype("float64")
        self.y_np = np.random.random((10, 4, 5)).astype("float64")
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.bmm(x, y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator':
            result = paddle.bmm(input=x, mat2=y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'out':
            out = paddle.empty([10, 3, 5], dtype='float64')
            out.stop_gradient = False
            paddle.bmm(x, y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        elif test_type == 'out_decorator':
            out = paddle.empty([10, 3, 5], dtype='float64')
            out.stop_gradient = False
            paddle.bmm(input=x, mat2=y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, grad_x_std, grad_y_std = self.do_test('raw')
        for test_type in self.test_types:
            out, grad_x, grad_y = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-7)
            np.testing.assert_allclose(
                grad_x.numpy(), grad_x_std.numpy(), rtol=1e-7
            )

            np.testing.assert_allclose(
                grad_y.numpy(), grad_y_std.numpy(), rtol=1e-7
            )


class TestBmmOutDtypeDynamicOnly(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def _skip_if_no_fp16_cuda(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            self.skipTest("CUDA is required for bmm out_dtype")
        if paddle.device.cuda.get_device_capability() < (5, 3):
            self.skipTest(
                "FP16 bmm out_dtype requires CUDA compute capability >= 5.3"
            )

    def _skip_if_no_bf16_cuda(self):
        if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
            self.skipTest("CUDA is required for bmm out_dtype")
        if paddle.device.cuda.get_device_capability()[0] < 8:
            self.skipTest(
                "BF16 bmm out_dtype requires CUDA compute capability >= 8"
            )

    def test_fp16_to_fp32(self):
        self._skip_if_no_fp16_cuda()
        x = paddle.randn([2, 3, 4], dtype='float16')
        y = paddle.randn([2, 4, 5], dtype='float16')
        result = paddle.bmm(x, y, out_dtype=paddle.float32)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_fp16_to_fp32_non_contiguous(self):
        self._skip_if_no_fp16_cuda()
        x = paddle.randn([2, 4, 3], dtype='float16').transpose([0, 2, 1])
        y = paddle.randn([2, 5, 4], dtype='float16').transpose([0, 2, 1])
        result = paddle.bmm(x, y, out_dtype=paddle.float32)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_fp16_to_fp32_out(self):
        self._skip_if_no_fp16_cuda()
        x = paddle.randn([2, 3, 4], dtype='float16')
        y = paddle.randn([2, 4, 5], dtype='float16')
        out = paddle.empty([2, 3, 5], dtype='float32')
        result = paddle.bmm(x, y, out_dtype=paddle.float32, out=out)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3
        )

    def test_out_dtype_rejects_mixed_input_dtypes(self):
        self._skip_if_no_fp16_cuda()
        x = paddle.randn([2, 3, 4], dtype='float16')
        y = paddle.randn([2, 4, 5], dtype='bfloat16')
        with self.assertRaises(TypeError):
            paddle.bmm(x, y, out_dtype=paddle.float32)

    def test_bf16_to_fp32(self):
        self._skip_if_no_bf16_cuda()
        x = paddle.randn([2, 3, 4], dtype='bfloat16')
        y = paddle.randn([2, 4, 5], dtype='bfloat16')
        result = paddle.bmm(x, y, out_dtype=paddle.float32)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_bf16_to_fp32_non_contiguous(self):
        self._skip_if_no_bf16_cuda()
        x = paddle.randn([2, 4, 3], dtype='bfloat16').transpose([0, 2, 1])
        y = paddle.randn([2, 5, 4], dtype='bfloat16').transpose([0, 2, 1])
        result = paddle.bmm(x, y, out_dtype=paddle.float32)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_bf16_to_fp32_out(self):
        self._skip_if_no_bf16_cuda()
        x = paddle.randn([2, 3, 4], dtype='bfloat16')
        y = paddle.randn([2, 4, 5], dtype='bfloat16')
        out = paddle.empty([2, 3, 5], dtype='float32')
        result = paddle.bmm(x, y, out_dtype=paddle.float32, out=out)
        expected = paddle.bmm(x.astype('float32'), y.astype('float32'))
        self.assertEqual(result.dtype, paddle.float32)
        np.testing.assert_allclose(
            result.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-2, atol=1e-2
        )

    def test_out_dtype_rejects_unsupported_cases(self):
        x = paddle.to_tensor(np.ones([2, 3, 4], dtype='float32'))
        y = paddle.to_tensor(np.ones([2, 4, 5], dtype='float32'))
        with self.assertRaises(TypeError):
            paddle.bmm(x, y, out_dtype=paddle.float32)

        x_fp16 = paddle.to_tensor(np.ones([2, 3, 4], dtype='float16'))
        y_fp16 = paddle.to_tensor(np.ones([2, 4, 5], dtype='float16'))
        with self.assertRaises(TypeError):
            paddle.bmm(x_fp16, y_fp16, out_dtype=paddle.float16)
        with self.assertRaises(TypeError):
            paddle.bmm(x_fp16, y, out_dtype=paddle.float32)
        with self.assertRaises(ValueError):
            paddle.bmm(
                x_fp16.reshape([6, 4]),
                y_fp16.reshape([8, 5]),
                out_dtype=paddle.float32,
            )
        with self.assertRaises(TypeError):
            paddle.bmm(
                x_fp16,
                y_fp16,
                out_dtype=paddle.float32,
                out=paddle.empty([2, 3, 5], dtype='float16'),
            )

    def test_static_shape_validation(self):
        paddle.enable_static()
        try:
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x_2d = paddle.static.data('x_2d', [3, 4], dtype='float32')
                y_2d = paddle.static.data('y_2d', [4, 5], dtype='float32')
                with self.assertRaises(ValueError):
                    paddle.bmm(x_2d, y_2d)

                x = paddle.static.data('x', [2, 3, 4], dtype='float32')
                y_bad_width = paddle.static.data(
                    'y_bad_width', [2, 5, 6], dtype='float32'
                )
                with self.assertRaises(ValueError):
                    paddle.bmm(x, y_bad_width)

                y_bad_batch = paddle.static.data(
                    'y_bad_batch', [3, 4, 5], dtype='float32'
                )
                with self.assertRaises(ValueError):
                    paddle.bmm(x, y_bad_batch)
        finally:
            paddle.disable_static()

    def test_static_out_dtype_fails_closed(self):
        paddle.enable_static()
        try:
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data('x', [2, 3, 4], dtype='bfloat16')
                y = paddle.static.data('y', [2, 4, 5], dtype='bfloat16')
                with self.assertRaises(NotImplementedError):
                    paddle.bmm(x, y, out_dtype=paddle.float32)
        finally:
            paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
