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

import paddle
from paddle import base


def skip_if_xpu_or_onednn_and_not_float32(dtype):
    """Skip test if using XPU or OneDNN and dtype is not float32"""

    def decorator(test_func):
        def wrapper(self):
            # Check if we're using XPU
            is_xpu = (hasattr(self, 'use_xpu') and self.use_xpu) or (
                paddle.device.get_device().startswith('xpu')
            )

            # Check if we're using OneDNN
            is_onednn = base.core.globals().get("FLAGS_use_onednn", False) or (
                hasattr(self, 'use_onednn') and self.use_onednn
            )

            # Skip if using XPU or OneDNN and dtype is not float32
            if (is_xpu or is_onednn) and dtype != 'float32':
                self.skipTest(
                    f"Skip {dtype} test for XPU/OneDNN, only test float32"
                )

            return test_func(self)

        return wrapper

    return decorator


class TestMeanDtypeParameter(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_dtype_float32(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)

    def test_dtype_float32_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, dtype='float32')
        result.backward()

        # Check gradient shape matches input shape
        self.assertEqual(x.grad.shape, x.shape)
        # Check gradient values (should be 1/numel for mean)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_float64(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_float64_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, dtype='float64')
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_dtype_none_default(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, dtype=None)
        result2 = paddle.mean(x)
        self.assertEqual(result1.dtype, result2.dtype)
        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_dtype_none_default_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, dtype=None)
        result2 = paddle.mean(x2)

        result1.backward()
        result2.backward()

        # Gradients should be identical
        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_with_axis(self):
        x = paddle.to_tensor(self.x_data)
        result = paddle.mean(x, axis=1, dtype='float64')
        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(result.shape, [3, 5])

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_with_axis_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        result = paddle.mean(x, axis=1, dtype='float64')
        loss = paddle.sum(result)
        loss.backward()

        # Check gradient shape
        self.assertEqual(x.grad.shape, x.shape)
        # For mean along axis=1, gradient should be 1/axis_size for each element
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


class TestMeanOutParameter(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_out_parameter_basic(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([], dtype='float32')
        result = paddle.mean(x, out=out)

        # Check that out is modified in-place
        self.assertTrue(paddle.allclose(out, result))
        np.testing.assert_allclose(
            out.numpy(), np.mean(self.x_data), rtol=1e-05
        )

    def test_out_parameter_basic_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([], dtype='float32')
        result = paddle.mean(x, out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_with_axis(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.mean(x, axis=1, out=out)

        self.assertTrue(paddle.allclose(out, result))
        self.assertEqual(out.shape, [3, 5])

    def test_out_parameter_with_axis_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([3, 5], dtype='float32')
        result = paddle.mean(x, axis=1, out=out)
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_with_keepdim(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([1, 1, 1], dtype='float32')
        result = paddle.mean(x, axis=[0, 1, 2], keepdim=True, out=out)

        self.assertTrue(paddle.allclose(out, result))
        self.assertEqual(out.shape, [1, 1, 1])

    def test_out_parameter_with_keepdim_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([1, 1, 1], dtype='float32')
        result = paddle.mean(x, axis=[0, 1, 2], keepdim=True, out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_out_parameter_none_default(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, out=None)
        result2 = paddle.mean(x)

        self.assertEqual(result1.dtype, result2.dtype)
        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_out_parameter_none_default_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, out=None)
        result2 = paddle.mean(x2)

        result1.backward()
        result2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)


class TestMeanDtypeAndOutCombination(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(2, 3, 4).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_and_out_compatible(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([], dtype='float64')
        result = paddle.mean(x, dtype='float64', out=out)

        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(result.dtype, paddle.float64)
        self.assertTrue(paddle.allclose(out, result))

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_dtype_and_out_compatible_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([], dtype='float64')
        result = paddle.mean(x, dtype='float64', out=out)
        result.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.size
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)

    def test_dtype_and_out_with_keepdim(self):
        x = paddle.to_tensor(self.x_data)
        out = paddle.empty([2, 1, 4], dtype='float32')
        result = paddle.mean(x, axis=1, keepdim=True, dtype='float32', out=out)

        self.assertEqual(out.shape, [2, 1, 4])
        self.assertTrue(paddle.allclose(out, result))

    def test_dtype_and_out_with_keepdim_backward(self):
        x = paddle.to_tensor(self.x_data, stop_gradient=False)
        out = paddle.empty([2, 1, 4], dtype='float32')
        result = paddle.mean(x, axis=1, keepdim=True, dtype='float32', out=out)
        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(self.x_data) / self.x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


class TestMeanParameterAlias(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_data = np.random.rand(3, 4, 5).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_x_alias_input(self):
        # Test x parameter alias
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x=x, axis=1)
        result2 = paddle.mean(input=x, axis=1)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_x_alias_input_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x=x1, axis=1)
        result2 = paddle.mean(input=x2, axis=1)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    def test_axis_alias_dim(self):
        # Test axis parameter alias
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, axis=1)
        result2 = paddle.mean(x, dim=1)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_axis_alias_dim_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, axis=1)
        result2 = paddle.mean(x2, dim=1)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    def test_multiple_axis_alias(self):
        x = paddle.to_tensor(self.x_data)
        result1 = paddle.mean(x, axis=[0, 2])
        result2 = paddle.mean(x, dim=[0, 2])

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)

    def test_multiple_axis_alias_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        result1 = paddle.mean(x1, axis=[0, 2])
        result2 = paddle.mean(x2, dim=[0, 2])

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_alias_with_dtype_and_out(self):
        x = paddle.to_tensor(self.x_data)
        out1 = paddle.empty([4], dtype='float64')
        out2 = paddle.empty([4], dtype='float64')

        result1 = paddle.mean(input=x, axis=[0, 2], dtype='float64', out=out1)
        result2 = paddle.mean(x=x, dim=[0, 2], dtype='float64', out=out2)

        np.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_alias_with_dtype_and_out_backward(self):
        x1 = paddle.to_tensor(self.x_data, stop_gradient=False)
        x2 = paddle.to_tensor(self.x_data, stop_gradient=False)

        out1 = paddle.empty([4], dtype='float64')
        out2 = paddle.empty([4], dtype='float64')

        result1 = paddle.mean(input=x1, axis=[0, 2], dtype='float64', out=out1)
        result2 = paddle.mean(x=x2, dim=[0, 2], dtype='float64', out=out2)

        loss1 = paddle.sum(result1)
        loss2 = paddle.sum(result2)

        loss1.backward()
        loss2.backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), rtol=1e-05)


class TestMeanNewParametersStatic(unittest.TestCase):
    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_static_dtype_parameter(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data('x', shape=[3, 4], dtype='float32')
            result = paddle.mean(x, dtype='float64')

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)

            exe.run(startup_prog)
            x_np = np.random.rand(3, 4).astype('float32')
            out = exe.run(main_prog, feed={'x': x_np}, fetch_list=[result])

            expected = np.mean(x_np).astype('float64')
            np.testing.assert_allclose(out[0], expected, rtol=1e-05)

    def test_static_alias_parameters(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data('x', shape=[3, 4], dtype='float32')
            result1 = paddle.mean(input=x, dim=1)
            result2 = paddle.mean(x=x, axis=1)

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)

            exe.run(startup_prog)
            x_np = np.random.rand(3, 4).astype('float32')
            out1, out2 = exe.run(
                main_prog, feed={'x': x_np}, fetch_list=[result1, result2]
            )

            np.testing.assert_allclose(out1, out2, rtol=1e-05)


class TestMeanBoundaryConditions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_dtype_with_int_input(self):
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int32')
        result = paddle.mean(x, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)
        expected = 3.5
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

    def test_dtype_with_int_input_backward(self):
        # Int input tensors don't support gradients, so we test the conversion
        x_float = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype='float32',
            stop_gradient=False,
        )
        result = paddle.mean(x_float, dtype='float32')
        result.backward()

        self.assertEqual(x_float.grad.shape, x_float.shape)
        expected_grad = np.ones_like(x_float.numpy()) / x_float.numel()
        np.testing.assert_allclose(
            x_float.grad.numpy(), expected_grad, rtol=1e-05
        )

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_all_parameters_combination(self):
        # Test all new parameters together
        x_data = np.random.rand(2, 3, 4).astype('float32')
        x = paddle.to_tensor(x_data)
        out = paddle.empty([2, 4], dtype='float64')

        result = paddle.mean(
            input=x, dim=1, keepdim=False, dtype='float64', out=out
        )

        self.assertEqual(result.dtype, paddle.float64)
        self.assertEqual(result.shape, [2, 4])
        self.assertTrue(paddle.allclose(out, result))

        expected = np.mean(x_data, axis=1).astype('float64')
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-05)

    @skip_if_xpu_or_onednn_and_not_float32('float64')
    def test_all_parameters_combination_backward(self):
        x_data = np.random.rand(2, 3, 4).astype('float32')
        x = paddle.to_tensor(x_data, stop_gradient=False)
        out = paddle.empty([2, 4], dtype='float64')

        result = paddle.mean(
            input=x, dim=1, keepdim=False, dtype='float64', out=out
        )

        loss = paddle.sum(result)
        loss.backward()

        self.assertEqual(x.grad.shape, x.shape)
        expected_grad = np.ones_like(x_data) / x_data.shape[1]
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
