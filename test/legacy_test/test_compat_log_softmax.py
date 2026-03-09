#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F


class TestCompatLogSoftmaxCorrectness(unittest.TestCase):
    """Test that compat log_softmax matches paddle.nn.functional.log_softmax."""

    def _compare_with_origin(self, input_tensor, dim):
        expected = F.log_softmax(input_tensor, axis=dim).numpy()
        actual = paddle.compat.nn.functional.log_softmax(
            input_tensor, dim=dim
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_2d_all_axes(self):
        x = paddle.randn([3, 4], dtype=paddle.float32)
        self._compare_with_origin(x, dim=0)
        self._compare_with_origin(x, dim=1)
        self._compare_with_origin(x, dim=-1)
        self._compare_with_origin(x, dim=-2)

    def test_3d_all_axes(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float32)
        for dim in [0, 1, 2, -1, -2, -3]:
            self._compare_with_origin(x, dim=dim)

    def test_4d_all_axes(self):
        x = paddle.randn([2, 3, 4, 5], dtype=paddle.float32)
        for dim in [0, 1, 2, 3, -1, -2, -3, -4]:
            self._compare_with_origin(x, dim=dim)

    def test_float64(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float64)
        for dim in [0, 1, 2, -1]:
            self._compare_with_origin(x, dim=dim)

    def test_1d(self):
        x = paddle.randn([5], dtype=paddle.float32)
        self._compare_with_origin(x, dim=0)
        self._compare_with_origin(x, dim=-1)


class TestCompatLogSoftmaxDimNoneDefault(unittest.TestCase):
    """Test PyTorch-compatible dim=None default behavior."""

    def test_0d_defaults_to_dim0(self):
        # 0-D tensor: dim defaults to 0
        x = paddle.to_tensor(1.0)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_1d_defaults_to_dim0(self):
        x = paddle.randn([4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_2d_defaults_to_dim1(self):
        x = paddle.randn([3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=1)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_3d_defaults_to_dim0(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_4d_defaults_to_dim1(self):
        x = paddle.randn([2, 3, 4, 5], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=1)
        np.testing.assert_allclose(out.numpy(), expected.numpy())


class TestCompatLogSoftmaxDtype(unittest.TestCase):
    """Test dtype casting behavior."""

    def test_float32_to_float64(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(
            x, dim=-1, dtype='float64'
        )
        self.assertEqual(out.dtype, paddle.float64)
        # Values should match float64 reference
        x64 = x.cast('float64')
        expected = F.log_softmax(x64, axis=-1)
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-10, atol=1e-10
        )

    def test_float64_to_float32(self):
        x = paddle.randn([2, 3], dtype=paddle.float64)
        out = paddle.compat.nn.functional.log_softmax(x, dim=1, dtype='float32')
        self.assertEqual(out.dtype, paddle.float32)

    def test_dtype_none_preserves_input_dtype(self):
        for dtype in [paddle.float32, paddle.float64]:
            x = paddle.randn([3, 4], dtype=dtype)
            out = paddle.compat.nn.functional.log_softmax(x, dim=-1)
            self.assertEqual(out.dtype, dtype)

    def test_dtype_as_paddle_dtype(self):
        x = paddle.randn([2, 3], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(
            x, dim=1, dtype=paddle.float64
        )
        self.assertEqual(out.dtype, paddle.float64)


class TestCompatLogSoftmaxOutputProperties(unittest.TestCase):
    """Test mathematical properties of log_softmax output."""

    def test_output_shape_preserved(self):
        for shape in [(4,), (3, 4), (2, 3, 4), (2, 3, 4, 5)]:
            x = paddle.randn(shape, dtype=paddle.float32)
            out = paddle.compat.nn.functional.log_softmax(x, dim=-1)
            self.assertEqual(list(out.shape), list(shape))

    def test_log_probabilities_sum_to_zero(self):
        # exp(log_softmax) should sum to 1 along dim, i.e. logsumexp of output = 0
        x = paddle.randn([3, 5], dtype=paddle.float64)
        out = paddle.compat.nn.functional.log_softmax(x, dim=1)
        out_np = out.numpy()
        sums = np.exp(out_np).sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-10)

    def test_output_is_non_positive(self):
        # log_softmax output must be <= 0 (log of probabilities in [0,1])
        x = paddle.randn([4, 6], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x, dim=-1)
        self.assertTrue((out.numpy() <= 0).all())

    def test_translation_invariance(self):
        # log_softmax(x + c) == log_softmax(x)
        x = paddle.randn([3, 4], dtype=paddle.float32)
        c = paddle.full([3, 4], fill_value=5.0, dtype=paddle.float32)
        out1 = paddle.compat.nn.functional.log_softmax(x, dim=1)
        out2 = paddle.compat.nn.functional.log_softmax(x + c, dim=1)
        np.testing.assert_allclose(
            out1.numpy(), out2.numpy(), rtol=1e-5, atol=1e-5
        )


class TestCompatLogSoftmaxStacklevel(unittest.TestCase):
    """Test that _stacklevel is silently ignored (torch compat)."""

    def test_stacklevel_ignored(self):
        x = paddle.randn([3, 4], dtype=paddle.float32)
        out1 = paddle.compat.nn.functional.log_softmax(x, dim=-1)
        out2 = paddle.compat.nn.functional.log_softmax(x, dim=-1, _stacklevel=5)
        np.testing.assert_allclose(out1.numpy(), out2.numpy())


class TestCompatLogSoftmaxErrorHandling(unittest.TestCase):
    """Test that paddle-style keyword arguments are rejected."""

    def test_rejects_x_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'x'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x=x, dim=-1)
        self.assertEqual(str(cm.exception), msg)

    def test_rejects_axis_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'axis'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x, axis=-1)
        self.assertEqual(str(cm.exception), msg)

    def test_rejects_name_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'name'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x, dim=-1, name='test')
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
