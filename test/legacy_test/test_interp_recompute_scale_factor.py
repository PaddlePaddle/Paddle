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

import math
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.nn import Upsample
from paddle.nn.functional import interpolate


def bilinear_interp_np(
    input,
    out_h,
    out_w,
    scale_w=0,
    scale_h=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    align_mode=0,
    data_layout='NCHW',
):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if align_mode == 0 and not align_corners:
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if align_mode == 0 and not align_corners:
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            if align_mode == 0 and not align_corners:
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
            if align_mode == 0 and not align_corners:
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, :, i, j] = h2lambda * (
                w2lambda * input[:, :, h, w]
                + w1lambda * input[:, :, h, w + wid]
            ) + h1lambda * (
                w2lambda * input[:, :, h + hid, w]
                + w1lambda * input[:, :, h + hid, w + wid]
            )

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


def nearest_neighbor_interp3d_np(
    X,
    out_d,
    out_h,
    out_w,
    scale_d=0,
    scale_h=0,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    data_layout='NCHW',
):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    if out_size is not None:
        out_d = out_size[0]
        out_h = out_size[1]
        out_w = out_size[2]
    if actual_shape is not None:
        out_d = actual_shape[0]
        out_h = actual_shape[1]
        out_w = actual_shape[2]
    n, c, in_d, in_h, in_w = X.shape

    ratio_d = ratio_h = ratio_w = 0.0
    if out_d > 1:
        if align_corners:
            ratio_d = (in_d - 1.0) / (out_d - 1.0)
        else:
            if scale_d > 0:
                ratio_d = 1.0 / scale_d
            else:
                ratio_d = 1.0 * in_d / out_d
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w
    out = np.zeros((n, c, out_d, out_h, out_w))

    if align_corners:
        for d in range(out_d):
            in_d = int(ratio_d * d + 0.5)
            for i in range(out_h):
                in_i = int(ratio_h * i + 0.5)
                for j in range(out_w):
                    in_j = int(ratio_w * j + 0.5)
                    out[:, :, d, i, j] = X[:, :, in_d, in_i, in_j]
    else:
        for d in range(out_d):
            in_d = int(ratio_d * d)
            for i in range(out_h):
                in_i = int(ratio_h * i)
                for j in range(out_w):
                    in_j = int(ratio_w * j)
                    out[:, :, d, i, j] = X[:, :, in_d, in_i, in_j]

    if data_layout == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC
    return out.astype(X.dtype)


def linear_interp_np(
    input,
    out_w,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    align_mode=0,
    data_layout='NCHW',
):
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 2, 1))  # NHWC => NCHW
    if out_size is not None:
        out_w = out_size[0]
    if actual_shape is not None:
        out_w = actual_shape[0]
    batch_size, channel, in_w = input.shape

    ratio_w = 0.0
    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_w))

    for j in range(out_w):
        if align_mode == 0 and not align_corners:
            w = int(ratio_w * (j + 0.5) - 0.5)
        else:
            w = int(ratio_w * j)
        w = max(0, w)
        wid = 1 if w < in_w - 1 else 0

        if align_mode == 0 and not align_corners:
            idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
            w1lambda = idx_src_w - w
        else:
            w1lambda = ratio_w * j - w
        w2lambda = 1.0 - w1lambda

        out[:, :, j] = (
            w2lambda * input[:, :, w] + w1lambda * input[:, :, w + wid]
        )

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestBilinearInterpOpAPI_RecomputeScaleFactor(unittest.TestCase):
    def test_case(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 7, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            scale_factor = 1.6

            in_h, in_w = input_data.shape[2], input_data.shape[3]
            expected_out_h = math.floor(in_h * scale_factor)
            expected_out_w = math.floor(in_w * scale_factor)

            # Calculate expected result
            expect_res = bilinear_interp_np(
                input_data,
                out_h=expected_out_h,
                out_w=expected_out_w,
                align_corners=False,
            )

            # Test with scalar scale_factor and recompute_scale_factor=True
            out1 = interpolate(
                x=input_x,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out1.numpy(), expect_res, rtol=1e-05)

            assert out1.shape[2] == expected_out_h
            assert out1.shape[3] == expected_out_w


class TestBilinearInterpOpAPI_RecomputeScaleFactorList(unittest.TestCase):
    def test_case(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 9, 6)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            scale_h, scale_w = 2.3, 0.7

            in_h, in_w = input_data.shape[2], input_data.shape[3]
            expected_out_h = math.floor(in_h * scale_h)
            expected_out_w = math.floor(in_w * scale_w)

            # Calculate expected result
            expect_res = bilinear_interp_np(
                input_data,
                out_h=expected_out_h,
                out_w=expected_out_w,
                align_corners=True,
            )

            # Test with list scale_factor and recompute_scale_factor=True
            scale_list = [scale_h, scale_w]
            out = interpolate(
                x=input_x,
                scale_factor=scale_list,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)

            assert out.shape[2] == expected_out_h
            assert out.shape[3] == expected_out_w


class TestBilinearInterpOpAPI_RecomputeScaleFactorDifferentTensors(
    unittest.TestCase
):
    def test_case(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 9, 6)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            scale_h, scale_w = 2.3, 0.7

            scale_tensor = paddle.to_tensor([scale_h, scale_w], dtype="float32")

            # Calculate expected output size with floor
            in_h, in_w = input_data.shape[2], input_data.shape[3]
            expected_out_h = math.floor(in_h * scale_h)
            expected_out_w = math.floor(in_w * scale_w)

            # Calculate expected result
            expect_res = bilinear_interp_np(
                input_data,
                out_h=expected_out_h,
                out_w=expected_out_w,
                align_corners=True,
            )

            # Test with tensor scale_factor and recompute_scale_factor=True
            out = interpolate(
                x=input_x,
                scale_factor=scale_tensor,
                mode="bilinear",
                align_corners=True,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)

            assert out.shape[2] == expected_out_h
            assert out.shape[3] == expected_out_w


class TestBilinearInterpOpAPI_RecomputeScaleFactorScalarTensor(
    unittest.TestCase
):
    def test_case(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 7, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            # Create a scalar tensor with empty shape []
            scale_value = 1.6
            scale_tensor = paddle.to_tensor(scale_value, dtype="float32")

            in_h, in_w = input_data.shape[2], input_data.shape[3]
            expected_out_h = math.floor(in_h * scale_value)
            expected_out_w = math.floor(in_w * scale_value)

            # Calculate expected result
            expect_res = bilinear_interp_np(
                input_data,
                out_h=expected_out_h,
                out_w=expected_out_w,
                align_corners=False,
            )

            # Test with tensor scale_factor and recompute_scale_factor=True
            out = interpolate(
                x=input_x,
                scale_factor=scale_tensor,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)

            assert out.shape[2] == expected_out_h
            assert out.shape[3] == expected_out_w


class TestNearestInterpOpAPI_RecomputeScaleFactor(unittest.TestCase):
    def test_case(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 4, 7, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            scale_factor = 1.6

            in_d, in_h, in_w = (
                input_data.shape[2],
                input_data.shape[3],
                input_data.shape[4],
            )
            expected_out_d = math.floor(in_d * scale_factor)
            expected_out_h = math.floor(in_h * scale_factor)
            expected_out_w = math.floor(in_w * scale_factor)

            # Calculate expected result
            expect_res = nearest_neighbor_interp3d_np(
                input_data,
                out_d=expected_out_d,
                out_h=expected_out_h,
                out_w=expected_out_w,
                align_corners=False,
            )

            # Test with scalar scale_factor and recompute_scale_factor=True
            out = interpolate(
                x=input_x,
                scale_factor=scale_factor,
                mode="nearest",
                align_corners=False,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)

            assert out.shape[2] == expected_out_d
            assert out.shape[3] == expected_out_h
            assert out.shape[4] == expected_out_w


class TestLinearInterpOpAPI_RecomputeScaleFactor(unittest.TestCase):
    def test_case(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create 3D input data
            input_data = np.random.random((2, 3, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            scale_factor = 1.6

            in_w = input_data.shape[2]
            expected_out_w = math.floor(in_w * scale_factor)

            # Calculate expected result
            expect_res = linear_interp_np(
                input_data,
                out_w=expected_out_w,
                align_corners=False,
            )

            # Test with scalar scale_factor and recompute_scale_factor=True
            out = interpolate(
                x=input_x,
                scale_factor=scale_factor,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=True,
            )

            # Verify results match
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)

            assert out.shape[2] == expected_out_w


class TestInterpRecomputeScaleFactorError(unittest.TestCase):
    def test_size_and_recompute_scale_factor_error(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data
            input_data = np.random.random((2, 3, 7, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            def test_invalid_params():
                out = interpolate(
                    x=input_x,
                    size=[14, 16],
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )

            self.assertRaises(ValueError, test_invalid_params)

            def test_invalid_params_upsample():
                upsample = Upsample(
                    size=[14, 16],
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )
                out = upsample(input_x)

            self.assertRaises(ValueError, test_invalid_params_upsample)


class TestInterpRecomputeScaleFactorScaleShapeError(unittest.TestCase):
    def test_incorrect_scale_shape(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with base.dygraph.guard(place):
            # Create input data - 4D tensor (N, C, H, W)
            input_data = np.random.random((2, 3, 7, 8)).astype("float32")
            input_x = paddle.to_tensor(input_data)

            # For a 4D tensor, dim = len(x.shape) - 2 = 2, so scale_factor should be of length 2
            # Providing a scale_factor of length 3 should trigger the error
            scale_list = [1.5, 2.0, 0.5]

            def test_invalid_scale_shape():
                out = interpolate(
                    x=input_x,
                    scale_factor=scale_list,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=True,
                )

            self.assertRaises(ValueError, test_invalid_scale_shape)

            # TTest with a 5D tensor
            input_data_5d = np.random.random((2, 3, 4, 7, 8)).astype("float32")
            input_x_5d = paddle.to_tensor(input_data_5d)

            # For a 5D tensor, dim = len(x.shape) - 2 = 3, so scale_factor should be of length 3
            # Providing a scale_factor of length 2 should trigger the error
            scale_list_5d = [1.5, 2.0]

            def test_invalid_scale_shape_5d():
                out = interpolate(
                    x=input_x_5d,
                    scale_factor=scale_list_5d,
                    mode="nearest",
                    align_corners=False,
                    recompute_scale_factor=True,
                )

            self.assertRaises(ValueError, test_invalid_scale_shape_5d)


if __name__ == "__main__":
    unittest.main()
