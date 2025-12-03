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


def _bilinear_filter(x):
    x = np.abs(x)
    if x < 1.0:
        return 1.0 - x
    return 0.0


def _bicubic_filter(x):
    x = np.abs(x)
    a = -0.5
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    elif x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _compute_weights_span(out_idx, in_size, scale, support):
    center = scale * (out_idx + 0.5)
    xmin = max(int(center - support + 0.5), 0)
    xsize = min(int(center + support + 0.5), in_size) - xmin
    return xmin, xsize, center


def _compute_weights(filter_fn, filter_size, scale, xmin, xsize, xmin_m_center):
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    weights = np.zeros(xsize, dtype=np.float64)
    total_w = 0.0

    for j in range(xsize):
        w = filter_fn((j + xmin_m_center + 0.5) * invscale)
        weights[j] = w
        total_w += w

    if total_w != 0.0:
        weights /= total_w

    return weights


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


def _area_pixel_compute_scale(input_size, output_size, align_corners, scale):
    if align_corners:
        if output_size > 1:
            return float(input_size - 1) / (output_size - 1)
        return 0.0
    else:
        if scale > 0:
            return 1.0 / scale
        if output_size > 0:
            return float(input_size) / output_size
    return 0.0


def _interpolate_aa_single_dim(src, weights, size):
    output = 0.0
    for j in range(size):
        output += src[j] * weights[j]
    return output


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
    is_nhwc = data_format == "NHWC"
    if is_nhwc:
        input = np.transpose(input, (0, 3, 1, 2))
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    filter_fn = (
        _bilinear_filter if interp_method == 'bilinear' else _bicubic_filter
    )
    filter_size = 2 if interp_method == 'bilinear' else 4

    ratio_h = _area_pixel_compute_scale(in_h, out_h, align_corners, scale_h)
    ratio_w = _area_pixel_compute_scale(in_w, out_w, align_corners, scale_w)

    scale_h_val = ratio_h
    scale_w_val = ratio_w
    half = 0.5
    support_h = (
        (filter_size * half) * scale_h_val
        if scale_h_val >= 1.0
        else filter_size * half
    )
    support_w = (
        (filter_size * half) * scale_w_val
        if scale_w_val >= 1.0
        else filter_size * half
    )

    out = np.zeros((batch_size, channel, out_h, out_w), dtype=input.dtype)

    for out_img_idy in range(out_h):
        for out_img_idx in range(out_w):
            ymin, ysize, ycenter = _compute_weights_span(
                out_img_idy, in_h, scale_h_val, support_h
            )
            xmin, xsize, xcenter = _compute_weights_span(
                out_img_idx, in_w, scale_w_val, support_w
            )

            wy = _compute_weights(
                filter_fn, filter_size, scale_h_val, ymin, ysize, ymin - ycenter
            )
            wx = _compute_weights(
                filter_fn, filter_size, scale_w_val, xmin, xsize, xmin - xcenter
            )

            for nc_id in range(batch_size * channel):
                buffer2 = np.zeros(ysize, dtype=input.dtype)
                for y in range(ysize):
                    buffer1 = input[
                        nc_id // channel,
                        nc_id % channel,
                        ymin + y,
                        xmin : xmin + xsize,
                    ]
                    buffer2[y] = _interpolate_aa_single_dim(buffer1, wx, xsize)

                out[
                    nc_id // channel, nc_id % channel, out_img_idy, out_img_idx
                ] = _interpolate_aa_single_dim(buffer2, wy, ysize)

    if is_nhwc:
        out = np.transpose(out, (0, 2, 3, 1))
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


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase5(TestInterpAntiAlias):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.align_corners = True


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Antialias only supported on GPU"
)
class TestBilinearInterpAntiAliasCase6(TestInterpAntiAlias):
    def init_test_case(self) -> None:
        self.interp_method = 'bilinear'
        self.data_format = 'NHWC'
        self.input_shape = (2, 10, 10, 3)


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
