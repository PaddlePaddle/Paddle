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


import math
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle.nn import functional as F


def max_pool1d_dilation_forward_naive(
    x, ksize, strides, paddings, dilations, ceil_mode=False
):
    """
    Compute 1D dilated max pooling result using numpy.

    For dilated pooling, the effective kernel size is:
        effective_ksize = dilation * (ksize - 1) + 1

    Output size formula:
        output_size = (input_size + 2*padding - dilation*(ksize-1) - 1) / stride + 1
    """
    N, C, L = x.shape
    dilation = dilations

    # Compute effective kernel size
    effective_ksize = dilation * (ksize - 1) + 1

    # Compute output length
    if ceil_mode:
        L_out = math.ceil((L + 2 * paddings - effective_ksize) / strides) + 1
    else:
        L_out = (L + 2 * paddings - effective_ksize) // strides + 1

    out = np.zeros((N, C, L_out), dtype=x.dtype)
    mask = np.zeros((N, C, L_out), dtype=np.int32)

    # Pad input
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (paddings, paddings)),
        mode='constant',
        constant_values=float('-inf'),
    )

    for i in range(L_out):
        start = i * strides
        # Collect elements at dilated positions
        for n in range(N):
            for c in range(C):
                max_val = float('-inf')
                max_idx = 0
                for k in range(ksize):
                    pos = start + k * dilation
                    if pos < x_padded.shape[2]:
                        val = x_padded[n, c, pos]
                        if val > max_val:
                            max_val = val
                            # Original index in padded tensor
                            max_idx = pos - paddings + k * dilation
                out[n, c, i] = max_val
                mask[n, c, i] = max_idx

    return out, mask


def max_pool2d_dilation_forward_naive(
    x, ksize, strides, paddings, dilations, ceil_mode=False
):
    """
    Compute 2D dilated max pooling result using numpy.

    For dilated pooling, the effective kernel size is:
        effective_ksize[i] = dilation[i] * (ksize[i] - 1) + 1

    Output size formula:
        output_size = (input_size + 2*padding - dilation*(ksize-1) - 1) / stride + 1
    """
    N, C, H, W = x.shape
    kh, kw = ksize
    sh, sw = strides
    ph, pw = paddings
    dh, dw = dilations

    # Compute effective kernel size
    effective_kh = dh * (kh - 1) + 1
    effective_kw = dw * (kw - 1) + 1

    # Compute output size
    if ceil_mode:
        H_out = math.ceil((H + 2 * ph - effective_kh) / sh) + 1
        W_out = math.ceil((W + 2 * pw - effective_kw) / sw) + 1
    else:
        H_out = (H + 2 * ph - effective_kh) // sh + 1
        W_out = (W + 2 * pw - effective_kw) // sw + 1

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    mask = np.zeros((N, C, H_out, W_out), dtype=np.int32)

    # Pad input
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=float('-inf'),
    )

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * sh
            w_start = j * sw
            for n in range(N):
                for c in range(C):
                    max_val = float('-inf')
                    max_idx = 0
                    for kh_idx in range(kh):
                        for kw_idx in range(kw):
                            h_pos = h_start + kh_idx * dh
                            w_pos = w_start + kw_idx * dw
                            if (
                                h_pos < x_padded.shape[2]
                                and w_pos < x_padded.shape[3]
                            ):
                                val = x_padded[n, c, h_pos, w_pos]
                                if val > max_val:
                                    max_val = val
                                    orig_h = h_pos - ph
                                    orig_w = w_pos - pw
                                    max_idx = orig_h * W + orig_w
                    out[n, c, i, j] = max_val
                    mask[n, c, i, j] = max_idx

    return out, mask


def max_pool3d_dilation_forward_naive(
    x, ksize, strides, paddings, dilations, ceil_mode=False
):
    """
    Compute 3D dilated max pooling result using numpy.
    """
    N, C, D, H, W = x.shape
    kd, kh, kw = ksize
    sd, sh, sw = strides
    pd, ph, pw = paddings
    dd, dh, dw = dilations

    # Compute effective kernel size
    effective_kd = dd * (kd - 1) + 1
    effective_kh = dh * (kh - 1) + 1
    effective_kw = dw * (kw - 1) + 1

    # Compute output size
    if ceil_mode:
        D_out = math.ceil((D + 2 * pd - effective_kd) / sd) + 1
        H_out = math.ceil((H + 2 * ph - effective_kh) / sh) + 1
        W_out = math.ceil((W + 2 * pw - effective_kw) / sw) + 1
    else:
        D_out = (D + 2 * pd - effective_kd) // sd + 1
        H_out = (H + 2 * ph - effective_kh) // sh + 1
        W_out = (W + 2 * pw - effective_kw) // sw + 1

    out = np.zeros((N, C, D_out, H_out, W_out), dtype=x.dtype)
    mask = np.zeros((N, C, D_out, H_out, W_out), dtype=np.int32)

    # Pad input
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=float('-inf'),
    )

    for d_idx in range(D_out):
        for i in range(H_out):
            for j in range(W_out):
                d_start = d_idx * sd
                h_start = i * sh
                w_start = j * sw
                for n in range(N):
                    for c in range(C):
                        max_val = float('-inf')
                        max_idx = 0
                        for kd_idx in range(kd):
                            for kh_idx in range(kh):
                                for kw_idx in range(kw):
                                    d_pos = d_start + kd_idx * dd
                                    h_pos = h_start + kh_idx * dh
                                    w_pos = w_start + kw_idx * dw
                                    if (
                                        d_pos < x_padded.shape[2]
                                        and h_pos < x_padded.shape[3]
                                        and w_pos < x_padded.shape[4]
                                    ):
                                        val = x_padded[
                                            n, c, d_pos, h_pos, w_pos
                                        ]
                                        if val > max_val:
                                            max_val = val
                                            orig_d = d_pos - pd
                                            orig_h = h_pos - ph
                                            orig_w = w_pos - pw
                                            max_idx = (
                                                orig_d * H * W
                                                + orig_h * W
                                                + orig_w
                                            )
                        out[n, c, d_idx, i, j] = max_val
                        mask[n, c, d_idx, i, j] = max_idx

    return out, mask


# ===================== Functional API Tests =====================


class TestMaxPool1DDilation(unittest.TestCase):
    """Test MaxPool1D with dilation parameter."""

    def setUp(self):
        np.random.seed(123)

    def test_max_pool1d_dilation_functional(self):
        """Test F.max_pool1d with dilation parameter."""
        input_np = np.random.random([2, 3, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # Test with dilation=2
        result, mask = F.max_pool1d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )

        # Verify output shape
        # effective_ksize = 2 * (3 - 1) + 1 = 5
        # L_out = (32 + 2*1 - 5) / 2 + 1 = 15
        expected_shape = [2, 3, 15]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertEqual(list(mask.shape), expected_shape)

    def test_max_pool1d_dilation_layer(self):
        """Test nn.MaxPool1D with dilation parameter."""
        input_np = np.random.random([2, 3, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        pool_layer = paddle.nn.MaxPool1D(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )
        result, mask = pool_layer(input_tensor)

        expected_shape = [2, 3, 15]
        self.assertEqual(list(result.shape), expected_shape)

    def test_max_pool1d_dilation_default(self):
        """Test that dilation=1 gives the same result as without dilation."""
        input_np = np.random.random([2, 3, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # With dilation=1 (default behavior)
        result_dilation1, _ = F.max_pool1d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            return_mask=True,
        )

        # Without specifying dilation (should default to 1)
        result_no_dilation = F.max_pool1d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            return_mask=False,
        )

        np.testing.assert_allclose(
            result_dilation1.numpy(),
            result_no_dilation.numpy(),
            rtol=1e-05,
        )

    def test_max_pool1d_dilation_numerical_correctness(self):
        """Test the numerical correctness of 1D dilated max pooling."""
        # Use small tensor for easy verification
        input_np = np.random.random([1, 2, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        ksize = 3
        stride = 2
        padding = 1
        dilation = 2

        result, mask = F.max_pool1d(
            input_tensor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_mask=True,
        )

        # Compare with numpy reference implementation
        expected_result, expected_mask = max_pool1d_dilation_forward_naive(
            input_np,
            ksize=ksize,
            strides=stride,
            paddings=padding,
            dilations=dilation,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected_result,
            rtol=1e-05,
            err_msg="MaxPool1D dilation output mismatch",
        )

    def test_max_pool1d_dilation_various_params(self):
        """Test 1D dilated max pooling with various parameter combinations."""
        test_configs = [
            {"ksize": 2, "stride": 1, "padding": 0, "dilation": 2},
            {"ksize": 3, "stride": 2, "padding": 1, "dilation": 3},
            {"ksize": 4, "stride": 2, "padding": 2, "dilation": 2},
        ]

        for config in test_configs:
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)

            result, _ = F.max_pool1d(
                input_tensor,
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )

            expected, _ = max_pool1d_dilation_forward_naive(
                input_np,
                ksize=config["ksize"],
                strides=config["stride"],
                paddings=config["padding"],
                dilations=config["dilation"],
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool1D mismatch with config: {config}",
            )


class TestMaxPool2DDilation(unittest.TestCase):
    """Test MaxPool2D with dilation parameter."""

    def setUp(self):
        np.random.seed(123)

    def test_max_pool2d_dilation_functional(self):
        """Test F.max_pool2d with dilation parameter."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # Test with dilation=2
        result, mask = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )

        # Verify output shape
        # effective_ksize = 2 * (3 - 1) + 1 = 5
        # H_out = (32 + 2*1 - 5) / 2 + 1 = 15
        expected_shape = [2, 3, 15, 15]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertEqual(list(mask.shape), expected_shape)

    def test_max_pool2d_dilation_layer(self):
        """Test nn.MaxPool2D with dilation parameter."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        pool_layer = paddle.nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )
        result, mask = pool_layer(input_tensor)

        expected_shape = [2, 3, 15, 15]
        self.assertEqual(list(result.shape), expected_shape)

    def test_max_pool2d_dilation_asymmetric(self):
        """Test with different dilation values for height and width."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # dilation = (2, 3)
        result, mask = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=(2, 3),
            return_mask=True,
        )

        # effective_kh = 2 * (3 - 1) + 1 = 5
        # effective_kw = 3 * (3 - 1) + 1 = 7
        # H_out = (32 + 2 - 5) / 2 + 1 = 15
        # W_out = (32 + 2 - 7) / 2 + 1 = 14
        expected_shape = [2, 3, 15, 14]
        self.assertEqual(list(result.shape), expected_shape)

    def test_max_pool2d_dilation_default(self):
        """Test that dilation=1 gives the same result as without dilation."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # With dilation=1
        result_dilation1, _ = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            return_mask=True,
        )

        # Without specifying dilation
        result_no_dilation = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            return_mask=False,
        )

        np.testing.assert_allclose(
            result_dilation1.numpy(),
            result_no_dilation.numpy(),
            rtol=1e-05,
        )

    def test_max_pool2d_dilation_correctness(self):
        """Test the correctness of dilated max pooling result."""
        input_np = np.random.random([1, 1, 8, 8]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = F.max_pool2d(
            input_tensor,
            kernel_size=2,
            stride=1,
            padding=0,
            dilation=2,
            return_mask=True,
        )

        # With dilation=2 and kernel=2x2, the effective kernel covers:
        # positions (0,0), (0,2), (2,0), (2,2) in first window
        # Output size: (8 + 0 - 2*(2-1) - 1) / 1 + 1 = (8 - 3) / 1 + 1 = 6
        expected_shape = [1, 1, 6, 6]
        self.assertEqual(list(result.shape), expected_shape)

        # Compare with numpy reference
        expected_result, _ = max_pool2d_dilation_forward_naive(
            input_np,
            ksize=[2, 2],
            strides=[1, 1],
            paddings=[0, 0],
            dilations=[2, 2],
        )
        np.testing.assert_allclose(result.numpy(), expected_result, rtol=1e-05)

    def test_max_pool2d_dilation_various_params(self):
        """Test 2D dilated max pooling with various parameter combinations."""
        test_configs = [
            {
                "ksize": [2, 2],
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [2, 2],
            },
            {
                "ksize": [3, 3],
                "stride": [2, 2],
                "padding": [1, 1],
                "dilation": [2, 3],
            },
            {
                "ksize": [2, 3],
                "stride": [1, 2],
                "padding": [1, 1],
                "dilation": [3, 2],
            },
        ]

        for config in test_configs:
            input_np = np.random.random([2, 3, 24, 24]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)

            result, _ = F.max_pool2d(
                input_tensor,
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )

            expected, _ = max_pool2d_dilation_forward_naive(
                input_np,
                ksize=config["ksize"],
                strides=config["stride"],
                paddings=config["padding"],
                dilations=config["dilation"],
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool2D mismatch with config: {config}",
            )


class TestMaxPool3DDilation(unittest.TestCase):
    """Test MaxPool3D with dilation parameter."""

    def setUp(self):
        np.random.seed(123)

    def test_max_pool3d_dilation_functional(self):
        """Test F.max_pool3d with dilation parameter."""
        input_np = np.random.random([2, 3, 8, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # Test with dilation=2
        result, mask = F.max_pool3d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )

        # effective_ksize = 2 * (3 - 1) + 1 = 5
        # D_out = (8 + 2 - 5) / 2 + 1 = 3
        # H_out = (16 + 2 - 5) / 2 + 1 = 7
        expected_shape = [2, 3, 3, 7, 7]
        self.assertEqual(list(result.shape), expected_shape)

    def test_max_pool3d_dilation_layer(self):
        """Test nn.MaxPool3D with dilation parameter."""
        input_np = np.random.random([2, 3, 8, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        pool_layer = paddle.nn.MaxPool3D(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )
        result, mask = pool_layer(input_tensor)

        expected_shape = [2, 3, 3, 7, 7]
        self.assertEqual(list(result.shape), expected_shape)

    def test_max_pool3d_dilation_default(self):
        """Test that dilation=1 gives the same result as without dilation."""
        input_np = np.random.random([2, 3, 8, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # With dilation=1
        result_dilation1, _ = F.max_pool3d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            return_mask=True,
        )

        # Without specifying dilation
        result_no_dilation = F.max_pool3d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            return_mask=False,
        )

        np.testing.assert_allclose(
            result_dilation1.numpy(),
            result_no_dilation.numpy(),
            rtol=1e-05,
        )

    def test_max_pool3d_dilation_numerical_correctness(self):
        """Test the numerical correctness of 3D dilated max pooling."""
        # Use smaller tensor for 3D to reduce computation
        input_np = np.random.random([1, 2, 8, 8, 8]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        ksize = [2, 2, 2]
        stride = [1, 1, 1]
        padding = [0, 0, 0]
        dilation = [2, 2, 2]

        result, mask = F.max_pool3d(
            input_tensor,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_mask=True,
        )

        # Compare with numpy reference
        expected_result, _ = max_pool3d_dilation_forward_naive(
            input_np,
            ksize=ksize,
            strides=stride,
            paddings=padding,
            dilations=dilation,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected_result,
            rtol=1e-05,
            err_msg="MaxPool3D dilation output mismatch",
        )

    def test_max_pool3d_dilation_various_params(self):
        """Test 3D dilated max pooling with various parameter combinations."""
        test_configs = [
            {
                "ksize": [2, 2, 2],
                "stride": [1, 1, 1],
                "padding": [0, 0, 0],
                "dilation": [2, 2, 2],
            },
            {
                "ksize": [2, 3, 3],
                "stride": [2, 2, 2],
                "padding": [1, 1, 1],
                "dilation": [2, 2, 3],
            },
        ]

        for config in test_configs:
            input_np = np.random.random([1, 2, 10, 12, 12]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)

            result, _ = F.max_pool3d(
                input_tensor,
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )

            expected, _ = max_pool3d_dilation_forward_naive(
                input_np,
                ksize=config["ksize"],
                strides=config["stride"],
                paddings=config["padding"],
                dilations=config["dilation"],
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool3D mismatch with config: {config}",
            )


class TestMaxPoolDilationValidation(unittest.TestCase):
    """Test parameter validation for dilation in MaxPool operations."""

    def test_negative_dilation(self):
        """Test that negative dilation raises an error."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        with self.assertRaises((ValueError, RuntimeError)):
            F.max_pool2d(
                input_tensor,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=-1,
                return_mask=True,
            )

    def test_dilation_one_no_return_mask(self):
        """Test that dilation=1 works without return_mask."""
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        # This should work fine
        result = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            return_mask=False,
        )

        expected_shape = [2, 3, 16, 16]
        self.assertEqual(list(result.shape), expected_shape)

    def test_dilation_channel_last_2d(self):
        """Test that dilation with NHWC format raises error."""
        input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(np.float32)
        input_pd = paddle.to_tensor(input_np)

        with self.assertRaises(ValueError):
            F.max_pool2d(
                input_pd,
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=2,
                ceil_mode=False,
                data_format='NHWC',
                return_mask=False,
            )

    def test_dilation_channel_last_3d(self):
        """Test that dilation with NDHWC format raises error."""
        input_np = np.random.uniform(-1, 1, [2, 3, 32, 32, 32]).astype(
            np.float32
        )
        input_pd = paddle.to_tensor(input_np)

        with self.assertRaises(ValueError):
            F.max_pool3d(
                input_pd,
                kernel_size=2,
                stride=2,
                padding=1,
                dilation=2,
                data_format='NDHWC',
                return_mask=False,
            )


class TestMaxPoolDilationGradient(unittest.TestCase):
    """Test gradient computation for dilated max pooling."""

    def test_max_pool2d_dilation_gradient(self):
        """Test that gradient can be computed for dilated max pooling."""
        input_np = np.random.random([2, 3, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)
        input_tensor.stop_gradient = False

        # Forward with dilation
        result, mask = F.max_pool2d(
            input_tensor,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2,
            return_mask=True,
        )

        # Backward
        loss = paddle.mean(result)
        loss.backward()

        # Check gradient exists and has correct shape
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(list(input_tensor.grad.shape), [2, 3, 16, 16])

    def test_max_pool3d_dilation_gradient(self):
        """Test that gradient can be computed for 3D dilated max pooling."""
        input_np = np.random.random([2, 3, 8, 8, 8]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)
        input_tensor.stop_gradient = False

        # Forward with dilation
        result, mask = F.max_pool3d(
            input_tensor,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=2,
            return_mask=True,
        )

        # Backward
        loss = paddle.mean(result)
        loss.backward()

        # Check gradient exists and has correct shape
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(list(input_tensor.grad.shape), [2, 3, 8, 8, 8])


class TestMaxPoolExtraRepr(unittest.TestCase):
    """Test extra_repr includes dilation for MaxPool layers."""

    def test_maxpool1d_extra_repr(self):
        """Test MaxPool1D extra_repr includes dilation."""
        pool = paddle.nn.MaxPool1D(
            kernel_size=3, stride=2, padding=1, dilation=2
        )
        repr_str = pool.extra_repr()
        self.assertIn('dilation', repr_str)
        self.assertIn('2', repr_str)

    def test_maxpool2d_extra_repr(self):
        """Test MaxPool2D extra_repr includes dilation."""
        pool = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, dilation=2
        )
        repr_str = pool.extra_repr()
        self.assertIn('dilation', repr_str)
        self.assertIn('2', repr_str)

    def test_maxpool3d_extra_repr(self):
        """Test MaxPool3D extra_repr includes dilation."""
        pool = paddle.nn.MaxPool3D(
            kernel_size=3, stride=2, padding=1, dilation=2
        )
        repr_str = pool.extra_repr()
        self.assertIn('dilation', repr_str)
        self.assertIn('2', repr_str)


class TestMaxPool1DLayerDilation(unittest.TestCase):
    """Test paddle.nn.MaxPool1D layer with dilation parameter."""

    def setUp(self):
        np.random.seed(42)

    def test_maxpool1d_layer_dilation_basic(self):
        """Test MaxPool1D layer with dilation parameter."""
        pool = paddle.nn.MaxPool1D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        # effective_ksize = 2 * (3 - 1) + 1 = 5
        # L_out = (32 + 2*1 - 5) / 2 + 1 = 15
        expected_shape = [2, 3, 15]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertEqual(list(mask.shape), expected_shape)

    def test_maxpool1d_layer_dilation_numerical_correctness(self):
        """Test MaxPool1D layer dilation numerical correctness."""
        input_np = np.random.random([1, 2, 16]).astype("float32")

        pool = paddle.nn.MaxPool1D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_tensor = paddle.to_tensor(input_np)
        result, _ = pool(input_tensor)

        expected, _ = max_pool1d_dilation_forward_naive(
            input_np,
            ksize=3,
            strides=2,
            paddings=1,
            dilations=2,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool1D layer dilation output mismatch",
        )

    def test_maxpool1d_layer_dilation_various_values(self):
        """Test MaxPool1D layer with various dilation values."""
        test_configs = [
            {"ksize": 2, "stride": 1, "padding": 0, "dilation": 2},
            {"ksize": 3, "stride": 2, "padding": 1, "dilation": 3},
            {"ksize": 2, "stride": 1, "padding": 1, "dilation": 4},
        ]

        for config in test_configs:
            pool = paddle.nn.MaxPool1D(
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)
            result, _ = pool(input_tensor)

            expected, _ = max_pool1d_dilation_forward_naive(
                input_np,
                ksize=config["ksize"],
                strides=config["stride"],
                paddings=config["padding"],
                dilations=config["dilation"],
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool1D layer mismatch with config: {config}",
            )

    def test_maxpool1d_layer_dilation_gradient(self):
        """Test MaxPool1D layer gradient with dilation."""
        pool = paddle.nn.MaxPool1D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)
        input_tensor.stop_gradient = False

        result, _ = pool(input_tensor)
        loss = paddle.mean(result)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(list(input_tensor.grad.shape), [2, 3, 16])

    def test_maxpool1d_layer_dilation_with_ceil_mode(self):
        """Test MaxPool1D layer with dilation and ceil_mode."""
        pool = paddle.nn.MaxPool1D(
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=True,
            return_mask=True,
        )
        input_np = np.random.random([2, 3, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        expected, _ = max_pool1d_dilation_forward_naive(
            input_np,
            ksize=3,
            strides=2,
            paddings=0,
            dilations=2,
            ceil_mode=True,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool1D layer dilation with ceil_mode mismatch",
        )


class TestMaxPool2DLayerDilation(unittest.TestCase):
    """Test paddle.nn.MaxPool2D layer with dilation parameter."""

    def setUp(self):
        np.random.seed(42)

    def test_maxpool2d_layer_dilation_basic(self):
        """Test MaxPool2D layer with dilation parameter."""
        pool = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        # effective_ksize = 2 * (3 - 1) + 1 = 5
        # H_out = (32 + 2*1 - 5) / 2 + 1 = 15
        expected_shape = [2, 3, 15, 15]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertEqual(list(mask.shape), expected_shape)

    def test_maxpool2d_layer_dilation_numerical_correctness(self):
        """Test MaxPool2D layer dilation numerical correctness."""
        input_np = np.random.random([1, 2, 16, 16]).astype("float32")

        pool = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_tensor = paddle.to_tensor(input_np)
        result, _ = pool(input_tensor)

        expected, _ = max_pool2d_dilation_forward_naive(
            input_np,
            ksize=[3, 3],
            strides=[2, 2],
            paddings=[1, 1],
            dilations=[2, 2],
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool2D layer dilation output mismatch",
        )

    def test_maxpool2d_layer_dilation_asymmetric(self):
        """Test MaxPool2D layer with asymmetric dilation values."""
        pool = paddle.nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=(2, 3),
            return_mask=True,
        )
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        # effective_kh = 2 * (3 - 1) + 1 = 5
        # effective_kw = 3 * (3 - 1) + 1 = 7
        expected_shape = [2, 3, 15, 14]
        self.assertEqual(list(result.shape), expected_shape)

        expected, _ = max_pool2d_dilation_forward_naive(
            input_np,
            ksize=[3, 3],
            strides=[2, 2],
            paddings=[1, 1],
            dilations=[2, 3],
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool2D layer asymmetric dilation mismatch",
        )

    def test_maxpool2d_layer_dilation_various_values(self):
        """Test MaxPool2D layer with various dilation values."""
        test_configs = [
            {"ksize": 2, "stride": 1, "padding": 0, "dilation": (2, 2)},
            {"ksize": 3, "stride": 2, "padding": 1, "dilation": (2, 3)},
            {"ksize": 2, "stride": 1, "padding": 1, "dilation": (3, 2)},
        ]

        for config in test_configs:
            pool = paddle.nn.MaxPool2D(
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )
            input_np = np.random.random([2, 3, 24, 24]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)
            result, _ = pool(input_tensor)

            ksize = [config["ksize"], config["ksize"]]
            stride = [config["stride"], config["stride"]]
            padding = [config["padding"], config["padding"]]
            dilation = list(config["dilation"])

            expected, _ = max_pool2d_dilation_forward_naive(
                input_np,
                ksize=ksize,
                strides=stride,
                paddings=padding,
                dilations=dilation,
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool2D layer mismatch with config: {config}",
            )

    def test_maxpool2d_layer_dilation_gradient(self):
        """Test MaxPool2D layer gradient with dilation."""
        pool = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)
        input_tensor.stop_gradient = False

        result, _ = pool(input_tensor)
        loss = paddle.mean(result)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(list(input_tensor.grad.shape), [2, 3, 16, 16])

    def test_maxpool2d_layer_dilation_with_ceil_mode(self):
        """Test MaxPool2D layer with dilation and ceil_mode."""
        pool = paddle.nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=True,
            return_mask=True,
        )
        input_np = np.random.random([2, 3, 32, 32]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        expected, _ = max_pool2d_dilation_forward_naive(
            input_np,
            ksize=[3, 3],
            strides=[2, 2],
            paddings=[0, 0],
            dilations=[2, 2],
            ceil_mode=True,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool2D layer dilation with ceil_mode mismatch",
        )


class TestMaxPool3DLayerDilation(unittest.TestCase):
    """Test paddle.nn.MaxPool3D layer with dilation parameter."""

    def setUp(self):
        np.random.seed(42)

    def test_maxpool3d_layer_dilation_basic(self):
        """Test MaxPool3D layer with dilation parameter."""
        pool = paddle.nn.MaxPool3D(
            kernel_size=2, stride=2, padding=0, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 8, 16, 16]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        # effective_ksize = 2 * (2 - 1) + 1 = 3
        # D_out = (8 - 3) / 2 + 1 = 3
        # H_out = (16 - 3) / 2 + 1 = 7
        expected_shape = [2, 3, 3, 7, 7]
        self.assertEqual(list(result.shape), expected_shape)
        self.assertEqual(list(mask.shape), expected_shape)

    def test_maxpool3d_layer_dilation_numerical_correctness(self):
        """Test MaxPool3D layer dilation numerical correctness."""
        input_np = np.random.random([1, 2, 8, 8, 8]).astype("float32")

        pool = paddle.nn.MaxPool3D(
            kernel_size=2, stride=1, padding=0, dilation=2, return_mask=True
        )
        input_tensor = paddle.to_tensor(input_np)
        result, _ = pool(input_tensor)

        expected, _ = max_pool3d_dilation_forward_naive(
            input_np,
            ksize=[2, 2, 2],
            strides=[1, 1, 1],
            paddings=[0, 0, 0],
            dilations=[2, 2, 2],
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool3D layer dilation output mismatch",
        )

    def test_maxpool3d_layer_dilation_asymmetric(self):
        """Test MaxPool3D layer with asymmetric dilation values."""
        pool = paddle.nn.MaxPool3D(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=(2, 2, 3),
            return_mask=True,
        )
        input_np = np.random.random([1, 2, 10, 12, 14]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, _ = pool(input_tensor)

        expected, _ = max_pool3d_dilation_forward_naive(
            input_np,
            ksize=[2, 2, 2],
            strides=[2, 2, 2],
            paddings=[0, 0, 0],
            dilations=[2, 2, 3],
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool3D layer asymmetric dilation mismatch",
        )

    def test_maxpool3d_layer_dilation_various_values(self):
        """Test MaxPool3D layer with various dilation values."""
        test_configs = [
            {"ksize": 2, "stride": 1, "padding": 0, "dilation": (2, 2, 2)},
            {"ksize": 2, "stride": 2, "padding": 1, "dilation": (2, 2, 3)},
        ]

        for config in test_configs:
            pool = paddle.nn.MaxPool3D(
                kernel_size=config["ksize"],
                stride=config["stride"],
                padding=config["padding"],
                dilation=config["dilation"],
                return_mask=True,
            )
            input_np = np.random.random([1, 2, 10, 12, 12]).astype("float32")
            input_tensor = paddle.to_tensor(input_np)
            result, _ = pool(input_tensor)

            ksize = [config["ksize"]] * 3
            stride = [config["stride"]] * 3
            padding = [config["padding"]] * 3
            dilation = list(config["dilation"])

            expected, _ = max_pool3d_dilation_forward_naive(
                input_np,
                ksize=ksize,
                strides=stride,
                paddings=padding,
                dilations=dilation,
            )

            np.testing.assert_allclose(
                result.numpy(),
                expected,
                rtol=1e-05,
                err_msg=f"MaxPool3D layer mismatch with config: {config}",
            )

    def test_maxpool3d_layer_dilation_gradient(self):
        """Test MaxPool3D layer gradient with dilation."""
        pool = paddle.nn.MaxPool3D(
            kernel_size=2, stride=2, padding=0, dilation=2, return_mask=True
        )
        input_np = np.random.random([2, 3, 8, 8, 8]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)
        input_tensor.stop_gradient = False

        result, _ = pool(input_tensor)
        loss = paddle.mean(result)
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(list(input_tensor.grad.shape), [2, 3, 8, 8, 8])

    def test_maxpool3d_layer_dilation_with_ceil_mode(self):
        """Test MaxPool3D layer with dilation and ceil_mode."""
        pool = paddle.nn.MaxPool3D(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=2,
            ceil_mode=True,
            return_mask=True,
        )
        input_np = np.random.random([1, 2, 8, 10, 10]).astype("float32")
        input_tensor = paddle.to_tensor(input_np)

        result, mask = pool(input_tensor)

        expected, _ = max_pool3d_dilation_forward_naive(
            input_np,
            ksize=[2, 2, 2],
            strides=[2, 2, 2],
            paddings=[0, 0, 0],
            dilations=[2, 2, 2],
            ceil_mode=True,
        )

        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1e-05,
            err_msg="MaxPool3D layer dilation with ceil_mode mismatch",
        )


# ===================== OpTest for max_pool2d_with_index with dilation =====================


def max_pool2d_with_index_dilation_wrapper(
    x,
    kernel_size=[],
    strides=[],
    paddings=[],
    dilations=[],
    global_pooling=False,
    adaptive=False,
    ceil_mode=False,
):
    return paddle._C_ops.max_pool2d_with_index(
        x,
        kernel_size,
        strides,
        paddings,
        dilations,
        global_pooling,
        adaptive,
        ceil_mode,
    )


class TestMaxPool2DWithIndexDilationOp(OpTest):
    """OpTest for max_pool2d_with_index with dilation parameter."""

    def setUp(self):
        self.op_type = "max_pool2d_with_index"
        self.python_api = max_pool2d_with_index_dilation_wrapper
        self.init_test_case()
        self.init_dtype()

        input = np.random.random(self.shape).astype(self.dtype)
        input = np.round(input * 100.0, 2)

        output, mask = max_pool2d_dilation_forward_naive(
            input,
            ksize=self.ksize,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
        )
        mask = mask.astype("int32")
        output = output.astype(self.dtype)

        self.inputs = {'X': input}
        self.outputs = {'Out': output, 'Mask': mask}
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'dilations': self.dilations,
            'global_pooling': False,
            'adaptive': False,
        }

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.shape = [2, 3, 16, 16]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [1, 1]
        self.dilations = [2, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad({'X'}, ['Out'])


class TestMaxPool2DWithIndexDilationOp2(TestMaxPool2DWithIndexDilationOp):
    """Test with different dilation values."""

    def init_test_case(self):
        self.shape = [2, 3, 24, 24]
        self.ksize = [2, 2]
        self.strides = [1, 1]
        self.paddings = [0, 0]
        self.dilations = [3, 3]


class TestMaxPool2DWithIndexDilationOp3(TestMaxPool2DWithIndexDilationOp):
    """Test with asymmetric dilation."""

    def init_test_case(self):
        self.shape = [2, 3, 20, 20]
        self.ksize = [3, 3]
        self.strides = [2, 2]
        self.paddings = [1, 1]
        self.dilations = [2, 3]


class TestMaxPool2DWithIndexDilationOp4(TestMaxPool2DWithIndexDilationOp):
    """Test with larger kernel and dilation."""

    def init_test_case(self):
        self.shape = [1, 2, 32, 32]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [2, 2]
        self.dilations = [2, 2]


class TestMaxPool2DWithIndexDilationOp5(TestMaxPool2DWithIndexDilationOp):
    """Test with asymmetric dilation."""

    def init_test_case(self):
        self.shape = [1, 2, 12, 14]
        self.ksize = [2, 2]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.dilations = [2, 3]


class TestMaxPool2DWithIndexDilationOp6(TestMaxPool2DWithIndexDilationOp):
    """Coverage with cpu branch."""

    def test_check_output(self):
        self.check_output_with_place(paddle.CPUPlace())

    def test_check_grad(self):
        self.check_grad_with_place(paddle.CPUPlace(), {'X'}, ['Out'])


# ===================== OpTest for max_pool3d_with_index with dilation =====================


def max_pool3d_with_index_dilation_wrapper(
    x,
    kernel_size=[],
    strides=[],
    paddings=[],
    dilations=[],
    global_pooling=False,
    adaptive=False,
    ceil_mode=False,
):
    return paddle._C_ops.max_pool3d_with_index(
        x,
        kernel_size,
        strides,
        paddings,
        dilations,
        global_pooling,
        adaptive,
        ceil_mode,
    )


class TestMaxPool3DWithIndexDilationOp(OpTest):
    """OpTest for max_pool3d_with_index with dilation parameter."""

    def setUp(self):
        self.op_type = "max_pool3d_with_index"
        self.python_api = max_pool3d_with_index_dilation_wrapper
        self.init_test_case()
        self.init_dtype()

        input = np.random.random(self.shape).astype(self.dtype)
        input = np.round(input * 100.0, 2)

        output, mask = max_pool3d_dilation_forward_naive(
            input,
            ksize=self.ksize,
            strides=self.strides,
            paddings=self.paddings,
            dilations=self.dilations,
        )
        mask = mask.astype("int32")
        output = output.astype(self.dtype)

        self.inputs = {'X': input}
        self.outputs = {'Out': output, 'Mask': mask}
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'dilations': self.dilations,
            'global_pooling': False,
            'adaptive': False,
        }

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.shape = [2, 3, 8, 8, 8]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.dilations = [2, 2, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad({'X'}, ['Out'])


class TestMaxPool3DWithIndexDilationOp2(TestMaxPool3DWithIndexDilationOp):
    """Test with different dilation values."""

    def init_test_case(self):
        self.shape = [1, 2, 10, 10, 10]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [1, 1, 1]
        self.dilations = [2, 2, 2]


class TestMaxPool3DWithIndexDilationOp3(TestMaxPool3DWithIndexDilationOp):
    """Test with asymmetric dilation."""

    def init_test_case(self):
        self.shape = [1, 2, 12, 12, 15]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.dilations = [2, 2, 3]


class TestMaxPool3DWithIndexDilationOp4(TestMaxPool3DWithIndexDilationOp):
    """Coverage with cpu branch."""

    def test_check_output(self):
        self.check_output_with_place(paddle.CPUPlace())

    def test_check_grad(self):
        self.check_grad_with_place(paddle.CPUPlace(), {'X'}, ['Out'])


if __name__ == '__main__':
    unittest.main()
