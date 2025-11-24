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
)

import paddle
from paddle import base
from paddle.base import core
from paddle.nn.functional import interpolate


def _bilinear_kernel_1d(x):
    """Bilinear filter kernel for anti-aliasing"""
    x = np.abs(x)
    if x < 1.0:
        return 1.0 - x
    else:
        return 0.0


def _compute_weights_and_indices_aa_bilinear(
    in_size, out_size, align_corners, scale
):
    """Compute anti-aliasing weights and indices for one dimension (bilinear)"""
    if align_corners:
        if out_size > 1:
            scale = (in_size - 1.0) / (out_size - 1.0)
        else:
            scale = 0.0
    else:
        if scale > 0:
            scale = 1.0 / scale
        else:
            scale = float(in_size) / float(out_size)

    # Filter support for bilinear is 1.0
    filter_scale = max(1.0, scale)
    support = 1.0 * filter_scale

    weights_list = []
    indices_list = []

    for out_idx in range(out_size):
        # Compute center
        if align_corners:
            center = out_idx * scale
        else:
            center = (out_idx + 0.5) * scale - 0.5

        # Compute support region
        left = int(np.floor(center - support))
        right = int(np.ceil(center + support))

        # Clip to valid range
        left = max(0, left)
        right = min(in_size - 1, right)

        # Compute weights
        weights = []
        indices = []
        total_weight = 0.0

        for i in range(left, right + 1):
            w = _bilinear_kernel_1d((i - center) / filter_scale)
            if abs(w) > 1e-10:
                weights.append(w)
                indices.append(i)
                total_weight += w

        # Normalize weights
        if abs(total_weight) > 1e-10:
            weights = [w / total_weight for w in weights]

        weights_list.append(weights)
        indices_list.append(indices)

    return weights_list, indices_list


def interp_antialias_test(
    x,
    OutSize=None,
    SizeTensor=None,
    Scale=None,
    data_format='NCHW',
    out_d=-1,
    out_h=-1,
    out_w=-1,
    scale=[],
    interp_method='bicubic',
    align_corners=True,
    align_mode=0,
):
    if isinstance(scale, (float, int)):
        scale_list = []
        for _ in range(len(x.shape) - 2):
            scale_list.append(scale)
        scale = list(map(float, scale_list))
    elif isinstance(scale, (list, tuple)):
        scale = list(map(float, scale))
    if SizeTensor is not None:
        if not isinstance(SizeTensor, list) and not isinstance(
            SizeTensor, tuple
        ):
            SizeTensor = [SizeTensor]
    return paddle._C_ops.interp_antialias(
        x,
        OutSize,
        SizeTensor,
        Scale,
        data_format,
        out_d,
        out_h,
        out_w,
        scale,
        interp_method,
        align_corners,
        align_mode,
    )


def cubic_1(x, a):
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def value_bound(input, w, h, x, y):
    access_x = int(max(min(x, w - 1), 0))
    access_y = int(max(min(y, h - 1), 0))
    return input[:, :, access_y, access_x]


def _bicubic_kernel_1d(x):
    """Bicubic filter kernel for anti-aliasing"""
    x = np.abs(x)
    a = -0.5
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    elif x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    else:
        return 0.0


def _compute_weights_and_indices_aa_bicubic(
    in_size, out_size, align_corners, scale
):
    """Compute anti-aliasing weights and indices for one dimension (bicubic)"""
    if align_corners:
        if out_size > 1:
            scale = (in_size - 1.0) / (out_size - 1.0)
        else:
            scale = 0.0
    else:
        if scale > 0:
            scale = 1.0 / scale
        else:
            scale = float(in_size) / float(out_size)

    # Filter support for bicubic is 2.0
    filter_scale = max(1.0, scale)
    support = 2.0 * filter_scale

    weights_list = []
    indices_list = []

    for out_idx in range(out_size):
        # Compute center
        if align_corners:
            center = out_idx * scale
        else:
            center = (out_idx + 0.5) * scale - 0.5

        # Compute support region
        left = int(np.floor(center - support))
        right = int(np.ceil(center + support))

        # Clip to valid range
        left = max(0, left)
        right = min(in_size - 1, right)

        # Compute weights
        weights = []
        indices = []
        total_weight = 0.0

        for i in range(left, right + 1):
            w = _bicubic_kernel_1d((i - center) / filter_scale)
            if abs(w) > 1e-10:
                weights.append(w)
                indices.append(i)
                total_weight += w

        # Normalize weights
        if abs(total_weight) > 1e-10:
            weights = [w / total_weight for w in weights]

        weights_list.append(weights)
        indices_list.append(indices)

    return weights_list, indices_list


def interp_antialias_np(
    input,
    out_h,
    out_w,
    scale_h=0,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    data_format='NCHW',
    interp_method='bicubic',
):
    """
    bilinear/bicubic interpolation implement in shape [N, C, H, W]
    with anti-aliasing support for downsampling
    """
    if data_format == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    # Use anti-aliasing implementation if requested and downsampling
    if out_h < in_h or out_w < in_w:
        out = np.zeros((batch_size, channel, out_h, out_w), dtype=input.dtype)

        # Choose the appropriate weight computation function based on interpolation method
        if interp_method == 'bilinear':
            compute_weights_fn = _compute_weights_and_indices_aa_bilinear
        else:  # bicubic
            compute_weights_fn = _compute_weights_and_indices_aa_bicubic

        # Compute weights for height
        h_weights, h_indices = compute_weights_fn(
            in_h, out_h, align_corners, max(0, scale_h)
        )

        # Compute weights for width
        w_weights, w_indices = compute_weights_fn(
            in_w, out_w, align_corners, max(0, scale_w)
        )

        # Apply separable convolution
        for b in range(batch_size):
            for c in range(channel):
                # First interpolate along width
                temp = np.zeros((in_h, out_w), dtype=input.dtype)
                for j in range(out_w):
                    for in_y in range(in_h):
                        val = 0.0
                        for w, idx in zip(w_weights[j], w_indices[j]):
                            val += input[b, c, in_y, idx] * w
                        temp[in_y, j] = val

                # Then interpolate along height
                for i in range(out_h):
                    for j in range(out_w):
                        val = 0.0
                        for w, idx in zip(h_weights[i], h_indices[i]):
                            val += temp[idx, j] * w
                        out[b, c, i, j] = val

        if data_format == "NHWC":
            out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

        return out.astype(input.dtype)

    # Standard interpolation (no anti-aliasing) for upsampling or same size
    ratio_h = ratio_w = 0.0
    if align_corners:
        if out_h > 1:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        if out_w > 1:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
    else:
        if scale_h > 0:
            ratio_h = 1.0 / scale_h
        else:
            ratio_h = 1.0 * in_h / out_h
        if scale_w > 0:
            ratio_w = 1.0 / scale_w
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    if interp_method == 'bilinear':
        # Standard bilinear interpolation
        for i in range(out_h):
            if align_corners:
                h = ratio_h * i
            else:
                h = ratio_h * (i + 0.5) - 0.5
            h = max(0, h)
            y_low = int(h)
            y_high = y_low + 1

            if y_low >= in_h - 1:
                y_high = y_low = in_h - 1
                y = float(y_low)
            else:
                y = float(y_low)

            y_high = min(y_high, in_h - 1)
            wy_h = h - y

            for j in range(out_w):
                if align_corners:
                    w = ratio_w * j
                else:
                    w = ratio_w * (j + 0.5) - 0.5
                w = max(0, w)
                x_low = int(w)
                x_high = x_low + 1

                if x_low >= in_w - 1:
                    x_high = x_low = in_w - 1
                    x = float(x_low)
                else:
                    x = float(x_low)

                x_high = min(x_high, in_w - 1)
                wy_w = w - x

                for b in range(batch_size):
                    for c in range(channel):
                        out[b, c, i, j] = (
                            input[b, c, y_low, x_low]
                            * (1.0 - wy_h)
                            * (1.0 - wy_w)
                            + input[b, c, y_low, x_high] * (1.0 - wy_h) * wy_w
                            + input[b, c, y_high, x_low] * wy_h * (1.0 - wy_w)
                            + input[b, c, y_high, x_high] * wy_h * wy_w
                        )
    else:  # bicubic
        # Standard bicubic interpolation
        for k in range(out_h):
            if align_corners:
                h = ratio_h * k
            else:
                h = ratio_h * (k + 0.5) - 0.5
            input_y = np.floor(h)
            y_t = h - input_y
            for l in range(out_w):
                if align_corners:
                    w = ratio_w * l
                else:
                    w = ratio_w * (l + 0.5) - 0.5
                input_x = np.floor(w)
                x_t = w - input_x
                for i in range(batch_size):
                    for j in range(channel):
                        coefficients = [0, 0, 0, 0]
                        for ii in range(4):
                            access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                            access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                            access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                            access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                            access_y = int(
                                max(min(input_y - 1 + ii, in_h - 1), 0)
                            )

                            coefficients[ii] = cubic_interp1d(
                                input[i, j, access_y, access_x_0],
                                input[i, j, access_y, access_x_1],
                                input[i, j, access_y, access_x_2],
                                input[i, j, access_y, access_x_3],
                                x_t,
                            )
                        out[i, j, k, l] = cubic_interp1d(
                            coefficients[0],
                            coefficients[1],
                            coefficients[2],
                            coefficients[3],
                            y_t,
                        )

    if data_format == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(input.dtype)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestInterpAntiAliasAlignment(unittest.TestCase):
    def test_antialias_vs_no_antialias(self):
        """Compare with and without anti-aliasing"""
        place = core.CUDAPlace(0)
        with base.dygraph.guard(place):
            input_data = np.random.random((1, 3, 64, 64)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            out_no_aa = interpolate(
                input_x,
                size=(32, 32),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )

            out_aa = interpolate(
                input_x,
                size=(32, 32),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # Both should have same shape
            self.assertEqual(out_aa.shape, out_no_aa.shape)
            # Both should be valid
            self.assertFalse(np.isnan(out_aa.numpy()).any())
            self.assertFalse(np.isnan(out_no_aa.numpy()).any())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasAlignment(unittest.TestCase):
    def test_antialias_vs_no_antialias(self):
        """Compare with and without anti-aliasing for bilinear"""
        place = core.CUDAPlace(0)
        with base.dygraph.guard(place):
            input_data = np.random.random((1, 3, 64, 64)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            out_no_aa = interpolate(
                input_x,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )

            out_aa = interpolate(
                input_x,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # Both should have same shape
            self.assertEqual(out_aa.shape, out_no_aa.shape)
            # Both should be valid
            self.assertFalse(np.isnan(out_aa.numpy()).any())
            self.assertFalse(np.isnan(out_no_aa.numpy()).any())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestInterpAntiAlias(OpTest):
    def setUp(self):
        self.python_api = interp_antialias_test
        self.op_type = "interp_antialias"
        self.interp_method = 'bicubic'
        self.input_shape = (2, 3, 10, 10)
        self.data_format = 'NCHW'
        self.dtype = np.float64
        self.out_h = 5
        self.out_w = 5
        self.scale_h = 0
        self.scale_w = 0
        self.align_corners = False
        self.init_test_case()
        input_np = np.random.random(self.input_shape).astype(self.dtype)

        output_np = interp_antialias_np(
            input_np,
            self.out_h,
            self.out_w,
            self.scale_h,
            self.scale_w,
            None,
            None,
            self.align_corners,
            self.data_format,
            self.interp_method,
        )
        self.inputs = {'x': input_np}
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'data_format': self.data_format,
        }
        self.outputs = {'output': output_np}

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=True)

    def test_check_grad(self):
        self.check_grad(['x'], 'output', in_place=False, check_pir=False)

    def init_test_case(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestInterpAntiAliasCase1(TestInterpAntiAlias):
    def init_test_case(self):
        self.scale_h = 0.5
        self.scale_w = 0.5


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestInterpAntiAliasCase2(TestInterpAntiAlias):
    def init_test_case(self):
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad(['x'], 'output', in_place=False, check_pir=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase1(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase2(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.scale_h = 0.5
        self.scale_w = 0.5


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase3(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = (2, 3, 8, 8)
        self.out_h = 16
        self.out_w = 16
        self.scale_h = 2.0
        self.scale_w = 2.0


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase4(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.dtype = np.float16

    def test_check_grad(self):
        self.check_grad(['x'], 'output', in_place=False, check_pir=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase5(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.align_corners = True


# @unittest.skipIf(
#     not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
# )
# class TestBilinearInterpAntiAliasCase6(TestInterpAntiAlias):
#     def init_test_case(self) -> None:
#         self.interp_method = 'bilinear'
#         self.data_format = 'NHWC'
#         self.input_shape = (2, 10, 10, 3)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasDifferentSizes(unittest.TestCase):
    def test_various_downsample_ratios(self):
        """Test bilinear antialias with various downsample ratios"""
        place = core.CUDAPlace(0)
        with base.dygraph.guard(place):
            input_data = np.random.random((1, 3, 64, 64)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            # Test different output sizes
            test_sizes = [(32, 32), (16, 16), (8, 8), (48, 48), (64, 32)]

            for size in test_sizes:
                with self.subTest(size=size):
                    out_aa = interpolate(
                        input_x,
                        size=size,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )

                    # Check output shape
                    self.assertEqual(out_aa.shape, (1, 3, size[0], size[1]))
                    # Check no NaN values
                    self.assertFalse(np.isnan(out_aa.numpy()).any())


if __name__ == "__main__":
    unittest.main()
