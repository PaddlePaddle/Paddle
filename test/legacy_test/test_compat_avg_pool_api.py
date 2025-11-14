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

import numpy as np
from op_test import get_places

import paddle


def avg_pool1d_forward_naive(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad=True,
):
    N, C, L_in = x.shape
    L_out = (
        (L_in + 2 * padding[0] - kernel_size[0] + stride[0] - 1) // stride[0]
        + 1
        if ceil_mode
        else (L_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    )

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        l_start = i * stride[0] - padding[0]
        l_end = l_start + kernel_size[0]

        x_masked = np.zeros((N, C, kernel_size[0]))
        for n in range(N):
            for c in range(C):
                for l in range(kernel_size[0]):
                    if l_start + l >= 0 and l_start + l < L_in:
                        x_masked[n, c, l] = x[n, c, l_start + l]

        if count_include_pad:
            field_size = kernel_size[0]
        else:
            field_size = np.sum(
                (np.arange(l_start, l_end) >= 0)
                & (np.arange(l_start, l_end) < L_in)
            )
            if field_size == 0:
                field_size = 1

        out[:, :, i] = np.sum(x_masked, axis=2) / field_size
    return out


def avg_pool2d_forward_naive(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    N, C, H_in, W_in = x.shape
    H_out = (
        (H_in + 2 * padding[0] - kernel_size[0] + stride[0] - 1) // stride[0]
        + 1
        if ceil_mode
        else (H_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    )
    W_out = (
        (W_in + 2 * padding[1] - kernel_size[1] + stride[1] - 1) // stride[1]
        + 1
        if ceil_mode
        else (W_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    )

    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride[0] - padding[0]
            h_end = h_start + kernel_size[0]
            w_start = j * stride[1] - padding[1]
            w_end = w_start + kernel_size[1]

            x_masked = np.zeros((N, C, kernel_size[0], kernel_size[1]))
            for n in range(N):
                for c in range(C):
                    for h in range(kernel_size[0]):
                        for w in range(kernel_size[1]):
                            if (
                                h_start + h >= 0
                                and h_start + h < H_in
                                and w_start + w >= 0
                                and w_start + w < W_in
                            ):
                                x_masked[n, c, h, w] = x[
                                    n, c, h_start + h, w_start + w
                                ]
            if divisor_override is not None:
                field_size = divisor_override
            elif count_include_pad:
                field_size = kernel_size[0] * kernel_size[1]
            else:
                field_size = np.sum(
                    (np.arange(h_start, h_end)[:, None] >= 0)
                    & (np.arange(h_start, h_end)[:, None] < H_in)
                    & (np.arange(w_start, w_end) >= 0)
                    & (np.arange(w_start, w_end) < W_in)
                )
                if field_size == 0:
                    field_size = 1

            out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / field_size
    return out


def avg_pool3d_forward_naive(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    N, C, D_in, H_in, W_in = x.shape
    D_out = (
        (D_in + 2 * padding[0] - kernel_size[0] + stride[0] - 1) // stride[0]
        + 1
        if ceil_mode
        else (D_in + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    )
    H_out = (
        (H_in + 2 * padding[1] - kernel_size[1] + stride[1] - 1) // stride[1]
        + 1
        if ceil_mode
        else (H_in + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    )
    W_out = (
        (W_in + 2 * padding[2] - kernel_size[2] + stride[2] - 1) // stride[2]
        + 1
        if ceil_mode
        else (W_in + 2 * padding[2] - kernel_size[2]) // stride[2] + 1
    )

    out = np.zeros((N, C, D_out, H_out, W_out))
    for i in range(D_out):
        for j in range(H_out):
            for k in range(W_out):
                d_start = i * stride[0] - padding[0]
                d_end = d_start + kernel_size[0]
                h_start = j * stride[1] - padding[1]
                h_end = h_start + kernel_size[1]
                w_start = k * stride[2] - padding[2]
                w_end = w_start + kernel_size[2]

                x_masked = np.zeros(
                    (N, C, kernel_size[0], kernel_size[1], kernel_size[2])
                )
                for n in range(N):
                    for c in range(C):
                        for d in range(kernel_size[0]):
                            for h in range(kernel_size[1]):
                                for w in range(kernel_size[2]):
                                    if (
                                        d_start + d >= 0
                                        and d_start + d < D_in
                                        and h_start + h >= 0
                                        and h_start + h < H_in
                                        and w_start + w >= 0
                                        and w_start + w < W_in
                                    ):
                                        x_masked[n, c, d, h, w] = x[
                                            n,
                                            c,
                                            d_start + d,
                                            h_start + h,
                                            w_start + w,
                                        ]

                if divisor_override is not None:
                    field_size = divisor_override
                elif count_include_pad:
                    field_size = (
                        kernel_size[0] * kernel_size[1] * kernel_size[2]
                    )
                else:
                    field_size = np.sum(
                        (np.arange(d_start, d_end)[:, None, None] >= 0)
                        & (np.arange(d_start, d_end)[:, None, None] < D_in)
                        & (np.arange(h_start, h_end)[:, None] >= 0)
                        & (np.arange(h_start, h_end)[:, None] < H_in)
                        & (np.arange(w_start, w_end) >= 0)
                        & (np.arange(w_start, w_end) < W_in)
                    )
                    if field_size == 0:
                        field_size = 1

                out[:, :, i, j, k] = (
                    np.sum(x_masked, axis=(2, 3, 4)) / field_size
                )
    return out


class TestCompatAvgPool1DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 32]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool1D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride]
            if isinstance(padding, int):
                padding = [padding]

            result_np = avg_pool1d_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
            )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True)
        self.run_test_case(3, 1, 1, False, True)
        self.run_test_case(3, 2, 0, True, True)
        self.run_test_case(3, 2, 1, True, False)
        self.run_test_case(3, None, 0, False, True)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool1D(2, exclusive=False, name="test")


class TestCompatAvgPool2DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 32, 32]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool2D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride, stride]
            if isinstance(padding, int):
                padding = [padding, padding]

            result_np = avg_pool2d_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True, None)
        self.run_test_case((3, 3), (1, 1), (1, 1), False, True, None)
        self.run_test_case(3, 2, 0, True, True, None)
        self.run_test_case(3, 2, 1, True, False, None)
        self.run_test_case(3, None, 0, False, True, None)
        self.run_test_case(3, 2, 1, False, False, 5)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool2D(
                2, exclusive=True, data_format="NHWC", name="test"
            )


class TestCompatAvgPool3DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 16, 16, 16]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool3D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size, kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride, stride, stride]
            if isinstance(padding, int):
                padding = [padding, padding, padding]

            result_np = avg_pool3d_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override,
            )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True, None)
        self.run_test_case((3, 3, 3), (1, 1, 1), (1, 1, 1), False, True, None)
        self.run_test_case(3, 2, 0, True, True, None)
        self.run_test_case(3, 2, 1, True, False, None)
        self.run_test_case(3, None, 0, False, True, None)
        self.run_test_case(3, 2, 1, False, False, 5)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool3D(
                2, exclusive=True, data_format="NDHWC", name="test"
            )


if __name__ == '__main__':
    unittest.main()
