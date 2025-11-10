# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core


class TestGradientDtypeConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtypes = [paddle.float32, paddle.float64]
        if core.is_compiled_with_cuda() and core.is_float16_supported(
            core.CUDAPlace(0)
        ):
            cls.dtypes.append(paddle.float16)
        if core.is_compiled_with_cuda() and core.is_bfloat16_supported(
            core.CUDAPlace(0)
        ):
            cls.dtypes.append(paddle.bfloat16)

    def _check_grad_dtype(self, tensors_with_grad):
        for tensor in tensors_with_grad:
            self.assertIsNotNone(
                tensor.grad,
                f"Gradient is None for tensor with dtype {tensor.dtype}",
            )
            self.assertEqual(
                tensor.grad.dtype,
                tensor.dtype,
                f"Expected grad dtype {tensor.dtype}, but got {tensor.grad.dtype}",
            )

    def _test_op_with_backward(
        self, op_func, inputs, output_shape, grad_dtype=None
    ):
        output = op_func(*inputs)

        if grad_dtype is not None:
            grad_tensor = paddle.randn(output_shape, dtype=grad_dtype)
            output.backward(grad_tensor=grad_tensor)
        else:
            output.backward()

        self._check_grad_dtype([inp for inp in inputs if not inp.stop_gradient])

    def _test_op_with_paddle_grad(
        self, op_func, inputs, output_shape, grad_dtype=None
    ):
        output = op_func(*inputs)

        grad_inputs = [inp for inp in inputs if not inp.stop_gradient]
        if grad_dtype is not None:
            grad_outputs = paddle.randn(output_shape, dtype=grad_dtype)
        else:
            grad_outputs = None

        grads = paddle.grad(
            outputs=output, inputs=grad_inputs, grad_outputs=grad_outputs
        )

        for grad, inp in zip(grads, grad_inputs):
            self.assertEqual(
                grad.dtype,
                inp.dtype,
                f"Expected grad dtype {inp.dtype}, but got {grad.dtype}",
            )

    # ==================== Test Add Operation ====================

    def test_add_mixed_dtype(self):
        """Test add with mixed dtypes"""
        test_cases = [
            (paddle.float32, paddle.float64),
            (paddle.float64, paddle.float32),
        ]
        if core.is_compiled_with_cuda() and core.is_bfloat16_supported(
            core.CUDAPlace(0)
        ):
            test_cases.append((paddle.float32, paddle.bfloat16))
            test_cases.append((paddle.bfloat16, paddle.float32))
        if core.is_compiled_with_cuda() and core.is_float16_supported(
            core.CUDAPlace(0)
        ):
            test_cases.append((paddle.bfloat16, paddle.float32))
            test_cases.append((paddle.float32, paddle.float16))

        for dtype1, dtype2 in test_cases:
            with self.subTest(dtype1=dtype1, dtype2=dtype2):
                x = paddle.randn([10, 10], dtype=dtype1)
                x.stop_gradient = False
                y = paddle.randn([10, 10], dtype=dtype2)
                y.stop_gradient = False

                self._test_op_with_backward(
                    lambda a, b: paddle.add(a, b), [x, y], [10, 10]
                )

    def test_add_with_different_grad_dtype(self):
        """Test add with different grad_tensor dtype"""
        x = paddle.randn([10, 10], dtype=paddle.float32)
        x.stop_gradient = False
        y = paddle.randn([10, 10], dtype=paddle.float32)
        y.stop_gradient = False

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._test_op_with_backward(
                    lambda a, b: paddle.add(a, b),
                    [x, y],
                    [10, 10],
                    grad_dtype=dtype,
                )

    def test_add_with_paddle_grad_api(self):
        """Test add using paddle.grad API"""
        x = paddle.randn([10, 10], dtype=paddle.float32)
        x.stop_gradient = False
        y = paddle.randn([10, 10], dtype=paddle.float32)
        y.stop_gradient = False

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._test_op_with_paddle_grad(
                    lambda a, b: paddle.add(a, b),
                    [x, y],
                    [10, 10],
                    grad_dtype=dtype,
                )

    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_float16_supported(core.CUDAPlace(0)),
        reason="only support float16 when compiled with CUDA.",
    )
    def test_add_with_middle_grad_change(self):
        """Test add with middle gradient change"""
        x_fp32 = paddle.randn([10, 10], dtype=paddle.float32)
        y_fp16 = paddle.randn([10, 10], dtype=paddle.float16)
        z_fp16 = paddle.randn([10, 10], dtype=paddle.float16)
        x_fp32.stop_gradient = False
        y_fp16.stop_gradient = False
        z_fp16.stop_gradient = False

        self._test_op_with_backward(
            lambda a, b, c: paddle.add(a, paddle.add(b, c)),
            [x_fp32, y_fp16, z_fp16],
            [10, 10],
        )

    # ==================== Test Matmul Operation ====================

    def test_matmul_with_different_grad_dtype(self):
        """Test matmul with different grad_tensor dtype"""
        x = paddle.randn([10, 10], dtype=paddle.float32)
        x.stop_gradient = False
        y = paddle.randn([10, 10], dtype=paddle.float32)
        y.stop_gradient = False

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._test_op_with_backward(
                    lambda a, b: paddle.matmul(a, b),
                    [x, y],
                    [10, 10],
                    grad_dtype=dtype,
                )

    def test_matmul_with_different_shape(self):
        """Test matmul with different shapes"""
        x = paddle.randn([3, 4], dtype=paddle.float32)
        x.stop_gradient = False
        y = paddle.randn([4, 5], dtype=paddle.float32)
        y.stop_gradient = False

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._test_op_with_backward(
                    lambda a, b: paddle.matmul(a, b),
                    [x, y],
                    [3, 5],
                    grad_dtype=dtype,
                )

    # ==================== Test Scatter Operation ====================

    def test_scatter_with_different_grad_dtype(self):
        """Test scatter with different grad_tensor dtype"""
        x = paddle.randn([10, 10], dtype=paddle.float32)
        x.stop_gradient = False
        index = paddle.randint(
            low=0, high=10, shape=[10, 1], dtype=paddle.int64
        )
        updates = paddle.randn([10, 10], dtype=paddle.float32)
        updates.stop_gradient = False

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._test_op_with_backward(
                    lambda a, b, c: paddle.scatter(a, b, c),
                    [x, index, updates],
                    [10, 10],
                    grad_dtype=dtype,
                )

    # ==================== Test Slice Operation ====================

    def test_slice_use_shared_buffer(self):
        """Test slice if IsSharedBufferWith"""
        x = paddle.randn([10], dtype=paddle.float32)
        x.stop_gradient = False
        y = x[:]

        y.backward()
        self._check_grad_dtype([x])

    def test_slice_with_different_grad_dtype(self):
        """Test slice with different grad_tensor dtype"""
        x = paddle.randn([10], dtype=paddle.float32)
        x.stop_gradient = False
        y = x[:]

        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                grad_tensor = paddle.ones_like(y, dtype=dtype)
                y.backward(grad_tensor=grad_tensor, retain_graph=True)
                self._check_grad_dtype([x])

    # ==================== Test AMP Transform ====================

    def test_amp(self):
        """Test amp transform"""
        model = paddle.nn.Linear(5, 5)
        model = paddle.amp.decorate(models=model, level="O2")

        input = paddle.randn([5, 5])
        input.stop_gradient = False

        with paddle.amp.auto_cast(level="O2"):
            output = model(input)
        loss = output.sum()
        loss.backward()

        self._check_grad_dtype([input, model.weight, model.bias])


if __name__ == "__main__":
    unittest.main()
