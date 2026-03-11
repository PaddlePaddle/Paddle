# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for paddle.nansum PHI kernel implementation."""

import unittest

import numpy as np

import paddle


def np_nansum(x, axis=None, keepdims=False, dtype=None):
    """Reference implementation using numpy."""
    if dtype is not None:
        return np.nansum(x, axis=axis, keepdims=keepdims).astype(dtype)
    return np.nansum(x, axis=axis, keepdims=keepdims)


def np_nansum_grad(x, out_grad_broadcast):
    """Reference grad: broadcast(out_grad) masked by ~isnan(x)."""
    grad = out_grad_broadcast.copy()
    grad[np.isnan(x)] = 0.0
    return grad


class TestNansumForward(unittest.TestCase):
    """Test nansum forward correctness on various cases."""

    def setUp(self):
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def _run_test(self, x_np, axis=None, keepdim=False, dtype=None):
        expected = np_nansum(x_np, axis=axis, keepdims=keepdim, dtype=dtype)
        paddle_dtype = None
        if dtype == 'float64':
            paddle_dtype = paddle.float64
        elif dtype == 'float32':
            paddle_dtype = paddle.float32
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.to_tensor(x_np, place=place)
            out = paddle.nansum(
                x, axis=axis, keepdim=keepdim, dtype=paddle_dtype
            )
            np.testing.assert_allclose(
                out.numpy(),
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Failed on {place}, axis={axis}, keepdim={keepdim}",
            )

    def test_all_nan(self):
        """nansum of all-NaN tensor should be 0."""
        x = np.array(
            [float('nan'), float('nan'), float('nan')], dtype='float32'
        )
        self._run_test(x)

    def test_no_nan(self):
        """nansum without NaN should equal sum."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32')
        self._run_test(x)

    def test_mixed_nan(self):
        """Basic mixed NaN/value test."""
        x = np.array(
            [[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]],
            dtype='float32',
        )
        self._run_test(x)
        self._run_test(x, axis=0)
        self._run_test(x, axis=1)
        self._run_test(x, axis=-1)

    def test_keepdim(self):
        x = np.array(
            [[float('nan'), 1.0], [2.0, float('nan')]], dtype='float32'
        )
        self._run_test(x, axis=1, keepdim=True)
        self._run_test(x, axis=0, keepdim=True)

    def test_reduce_all(self):
        """axis=None reduces all dims."""
        x = np.array(
            [[[1, float('nan')], [3, 4]], [[5, 6], [float('nan'), 8]]],
            dtype='float32',
        )
        self._run_test(x)

    def test_multi_axis(self):
        x = np.array(
            [[[1, float('nan')], [3, 4]], [[5, 6], [float('nan'), 8]]],
            dtype='float32',
        )
        self._run_test(x, axis=(1, 2))
        self._run_test(x, axis=(0, 1))

    def test_dtype_promotion(self):
        """Test output dtype control."""
        x = np.array([1.0, float('nan'), 3.0], dtype='float32')
        self._run_test(x, dtype='float64')

    def test_integer_input(self):
        """Integer types have no NaN; nansum == sum."""
        x = np.array([1, 2, 3, 4], dtype='int32')
        self._run_test(x)
        self._run_test(x, axis=0)

    def test_empty_tensor(self):
        """nansum of empty tensor should be 0."""
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.empty([0, 3], dtype='float32')
            out = paddle.nansum(x)
            self.assertEqual(out.item(), 0.0)

    def test_empty_tensor_int64(self):
        """nansum of empty int32 tensor with dtype=int64 should be 0."""
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.empty([0, 3], dtype='int32')
            out = paddle.nansum(x, dtype=paddle.int64)
            self.assertEqual(out.item(), 0)
            self.assertEqual(out.dtype, paddle.int64)

    def test_neg_nan(self):
        """-NaN should also be treated as 0."""
        x = np.array([1.0, float('-nan'), 3.0], dtype='float32')
        self._run_test(x)

    def test_single_element(self):
        x_nan = np.array([float('nan')], dtype='float32')
        x_val = np.array([5.0], dtype='float32')
        self._run_test(x_nan)
        self._run_test(x_val)


class TestNansumBackward(unittest.TestCase):
    """Test nansum backward (gradient) correctness."""

    def setUp(self):
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def _check_grad(self, x_np, axis=None, keepdim=False):
        expected_out = np_nansum(x_np, axis=axis, keepdims=keepdim)
        # Compute expected gradient: ones broadcast to x shape, masked by ~isnan
        grad_out = np.ones_like(expected_out)
        # Broadcast grad_out to x shape
        if axis is not None:
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)
            # Normalize negative axes
            axes = [a % x_np.ndim for a in axes]
            expand_shape = list(x_np.shape)
            for a in axes:
                expand_shape[a] = 1
            grad_broadcast = grad_out.reshape(expand_shape)
            grad_broadcast = np.broadcast_to(grad_broadcast, x_np.shape)
        else:
            grad_broadcast = np.broadcast_to(grad_out, x_np.shape)
        expected_grad = np_nansum_grad(x_np, grad_broadcast)

        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.to_tensor(x_np, place=place, stop_gradient=False)
            out = paddle.nansum(x, axis=axis, keepdim=keepdim)
            out.backward()
            np.testing.assert_allclose(
                x.grad.numpy(),
                expected_grad,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Grad failed on {place}, axis={axis}",
            )

    def test_grad_basic(self):
        x = np.array(
            [[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]],
            dtype='float32',
        )
        self._check_grad(x)

    def test_grad_axis0(self):
        x = np.array(
            [[float('nan'), 1.0], [2.0, float('nan')]], dtype='float32'
        )
        self._check_grad(x, axis=0)

    def test_grad_axis1(self):
        x = np.array(
            [[float('nan'), 1.0], [2.0, float('nan')]], dtype='float32'
        )
        self._check_grad(x, axis=1)

    def test_grad_all_nan(self):
        """All-NaN: gradient should be all zeros."""
        x = np.array([float('nan'), float('nan')], dtype='float32')
        self._check_grad(x)

    def test_grad_no_nan(self):
        """No NaN: gradient should be all ones (like sum)."""
        x = np.array([1.0, 2.0, 3.0], dtype='float32')
        self._check_grad(x)

    def test_grad_keepdim(self):
        x = np.array([[float('nan'), 1.0], [2.0, 3.0]], dtype='float32')
        self._check_grad(x, axis=1, keepdim=True)

    def test_grad_3d_multi_axis(self):
        x = np.array(
            [[[1, float('nan')], [3, 4]], [[5, 6], [float('nan'), 8]]],
            dtype='float32',
        )
        self._check_grad(x, axis=(1, 2))

    def test_grad_float64(self):
        x = np.array([float('nan'), 1.0, 2.0], dtype='float64')
        self._check_grad(x)

    def test_grad_empty_tensor(self):
        """Backward on empty tensor: x_grad should be empty with correct shape."""
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.empty([0, 3], dtype='float32')
            x.stop_gradient = False
            out = paddle.nansum(x)
            out.backward()
            self.assertEqual(list(x.grad.shape), [0, 3])


class TestNansumAlignPyTorch(unittest.TestCase):
    """Explicit alignment tests with known PyTorch outputs."""

    def setUp(self):
        self.places = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_torch_example_1(self):
        """torch.nansum(tensor([nan, 0.3, 0.5, 0.9, 0.1, 0.2, nan, 0.7])) = 2.7"""
        x_np = np.array(
            [float('nan'), 0.3, 0.5, 0.9, 0.1, 0.2, float('nan'), 0.7],
            dtype='float32',
        )
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.to_tensor(x_np, place=place)
            out = paddle.nansum(x)
            np.testing.assert_allclose(out.numpy(), 2.7, rtol=1e-5)

    def test_torch_example_2d_axis0(self):
        """Matches torch.nansum(x, dim=0) for 2x4 with NaN."""
        x_np = np.array(
            [[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('-nan'), 0.7]],
            dtype='float32',
        )
        expected = np.array([0.1, 0.5, 0.5, 1.6], dtype='float32')
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.to_tensor(x_np, place=place)
            out = paddle.nansum(x, axis=0)
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_scalar_output_stop_gradient(self):
        """Verify nansum returns scalar for full reduce."""
        for place in self.places:
            paddle.device.set_device(str(place))
            x = paddle.to_tensor([float('nan'), 1.0, 2.0], place=place)
            out = paddle.nansum(x)
            self.assertEqual(out.shape, [])


class TestNansumStaticGraph(unittest.TestCase):
    """Test nansum under jit.to_static to trigger InferSymbolicShape."""

    def test_to_static(self):
        class NansumLayer(paddle.nn.Layer):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return paddle.nansum(x, axis=1, keepdim=True)

        net = NansumLayer()
        x_spec = paddle.static.InputSpec(
            shape=[None, None, None], dtype='float32'
        )
        static_net = paddle.jit.to_static(
            net, input_spec=[x_spec], full_graph=True
        )
        x = paddle.randn([2, 3, 4])
        out = static_net(x)
        expected = paddle.nansum(x, axis=1, keepdim=True)
        np.testing.assert_allclose(out.numpy(), expected.numpy())


if __name__ == '__main__':
    unittest.main()
