#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import check_out_dtype

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


def fractional_rational_u(u, alpha, input, output, pool_size=0):
    if pool_size > 0:
        return u

    base = input // output

    u_max1 = (base + 2) / alpha - 1
    u_max2 = (input + 1 - base) / alpha - (output - 1)
    max_u = min(u_max1, u_max2)

    return u * max_u


def fractional_start_index(idx, alpha, u, pool_size=0):
    return int((idx + u) * alpha) - int(u * alpha)


def fractional_end_index(idx, alpha, u, pool_size=0):
    if pool_size > 0:
        return int((idx + u) * alpha) - int(u * alpha) + pool_size
    return int((idx + 1 + u) * alpha) - int(u * alpha)


def fractional_pool3d_forward(
    x,
    output_size,
    kernel_size=None,
    random_u=None,
    data_format='NCDHW',
    pool_type='max',
):
    N = x.shape[0]
    C, D, H, W = (
        [x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        if data_format == 'NCDHW'
        else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]
    )

    if kernel_size is None:
        pool_depth = 0
        pool_height = 0
        pool_width = 0
    elif isinstance(kernel_size, int):
        pool_depth = kernel_size
        pool_height = kernel_size
        pool_width = kernel_size
    else:
        pool_depth, pool_height, pool_width = kernel_size

    if isinstance(output_size, int):
        H_out = output_size
        W_out = output_size
        D_out = output_size
        output_size = [D_out, H_out, W_out]
    else:
        D_out, H_out, W_out = output_size

    if output_size[0] is None:
        output_size[0] = D
        D_out = D
    if output_size[1] is None:
        output_size[1] = H
        H_out = H
    if output_size[2] is None:
        output_size[2] = W
        W_out = W

    out = (
        np.zeros((N, C, D_out, H_out, W_out))
        if data_format == 'NCDHW'
        else np.zeros((N, D_out, H_out, W_out, C))
    )

    u = random_u

    alpha_depth = (D - pool_depth) / (D_out - (1 if pool_depth > 0 else 0))
    alpha_height = (H - pool_height) / (H_out - (1 if pool_height > 0 else 0))
    alpha_width = (W - pool_width) / (W_out - (1 if pool_width > 0 else 0))

    u_depth = fractional_rational_u(u, alpha_depth, D, D_out, pool_depth)
    u_height = fractional_rational_u(u, alpha_height, H, H_out, pool_height)
    u_width = fractional_rational_u(u, alpha_width, W, W_out, pool_width)

    for k in range(D_out):
        d_start = fractional_start_index(k, alpha_depth, u_depth, pool_depth)
        d_end = fractional_end_index(k, alpha_depth, u_depth, pool_depth)
        d_start = max(d_start, 0)
        d_end = min(d_end, D)

        for i in range(H_out):
            h_start = fractional_start_index(
                i, alpha_height, u_height, pool_height
            )
            h_end = fractional_end_index(i, alpha_height, u_height, pool_height)
            h_start = max(h_start, 0)
            h_end = min(h_end, H)

            for j in range(W_out):
                w_start = fractional_start_index(
                    j, alpha_width, u_width, pool_width
                )
                w_end = fractional_end_index(
                    j, alpha_width, u_width, pool_width
                )
                w_start = max(w_start, 0)
                w_end = min(w_end, W)

                if data_format == 'NCDHW':
                    x_masked = x[
                        :, :, d_start:d_end, h_start:h_end, w_start:w_end
                    ]
                    if pool_type == 'avg':
                        field_size = (
                            (d_end - d_start)
                            * (h_end - h_start)
                            * (w_end - w_start)
                        )
                        out[:, :, k, i, j] = (
                            np.sum(x_masked, axis=(2, 3, 4)) / field_size
                        )
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
                    x_masked = x[
                        :, d_start:d_end, h_start:h_end, w_start:w_end, :
                    ]
                    if pool_type == 'avg':
                        field_size = (
                            (d_end - d_start)
                            * (h_end - h_start)
                            * (w_end - w_start)
                        )
                        out[:, k, i, j, :] = (
                            np.sum(x_masked, axis=(1, 2, 3)) / field_size
                        )
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))
    return out


class TestFractionalMaxPool3DAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.x_np = np.random.random([2, 3, 7, 7, 7]).astype("float32")
        self.res_1_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], random_u=0.3
        )

        self.res_2_np = fractional_pool3d_forward(
            x=self.x_np, output_size=5, random_u=0.5
        )

        self.res_3_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], random_u=0.7
        )

        self.res_4_np = fractional_pool3d_forward(
            x=self.x_np,
            kernel_size=2,
            output_size=[3, 3, 3],
            random_u=0.6,
        )

        self.res_5_np = fractional_pool3d_forward(
            x=self.x_np,
            kernel_size=[2, 2, 2],
            output_size=[3, 3, 3],
            random_u=0.6,
        )

        self.res_6_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[None, 3, 3], random_u=0.6
        )

        self.res_7_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[3, None, 3], random_u=0.6
        )

        self.res_8_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[3, 3, None], random_u=0.6
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 7, 7, 7], dtype="float32"
            )

            out_1 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, 3], random_u=0.3
            )

            out_2 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=5, random_u=0.5
            )

            out_3 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[2, 3, 5], random_u=0.7
            )

            out_4 = paddle.nn.functional.fractional_max_pool3d(
                x=x, kernel_size=2, output_size=[3, 3, 3], random_u=0.6
            )

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x, kernel_size=[2, 2, 2], output_size=[3, 3, 3], random_u=0.6
            )

            out_6 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, 3], random_u=0.6
            )

            out_7 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, None, 3], random_u=0.6
            )

            out_8 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, None], random_u=0.6
            )

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8] = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[
                    out_1,
                    out_2,
                    out_3,
                    out_4,
                    out_5,
                    out_6,
                    out_7,
                    out_8,
                ],
            )

            np.testing.assert_allclose(res_1, self.res_1_np)
            np.testing.assert_allclose(res_2, self.res_2_np)
            np.testing.assert_allclose(res_3, self.res_3_np)
            np.testing.assert_allclose(res_4, self.res_4_np)
            np.testing.assert_allclose(res_5, self.res_5_np)
            np.testing.assert_allclose(res_6, self.res_6_np)
            np.testing.assert_allclose(res_7, self.res_7_np)
            np.testing.assert_allclose(res_8, self.res_8_np)

    def test_static_graph_return_mask(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 7, 7, 7], dtype="float32"
            )

            out_1 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, 3], return_mask=True, random_u=0.3
            )

            out_2 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=5, return_mask=True, random_u=0.5
            )

            out_3 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[2, 3, 5], return_mask=True, random_u=0.7
            )

            out_4 = paddle.nn.functional.fractional_max_pool3d(
                x=x,
                kernel_size=2,
                output_size=[3, 3, 3],
                return_mask=True,
                random_u=0.6,
            )

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x,
                kernel_size=[2, 2, 2],
                output_size=[3, 3, 3],
                return_mask=True,
                random_u=0.6,
            )

            out_6 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, 3], return_mask=True, random_u=0.6
            )

            out_7 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, None, 3], return_mask=True, random_u=0.6
            )

            out_8 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, None], return_mask=True, random_u=0.6
            )

            exe = paddle.static.Executor(place=place)
            [
                res_1,
                mask_1,
                res_2,
                mask_2,
                res_3,
                mask_3,
                res_4,
                mask_4,
                res_5,
                mask_5,
                res_6,
                mask_6,
                res_7,
                mask_7,
                res_8,
                mask_8,
            ] = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[
                    out_1,
                    out_2,
                    out_3,
                    out_4,
                    out_5,
                    out_6,
                    out_7,
                    out_8,
                ],
            )

            self.assertEqual(res_1.shape, mask_1.shape)
            self.assertEqual(res_2.shape, mask_2.shape)
            self.assertEqual(res_3.shape, mask_3.shape)
            self.assertEqual(res_4.shape, mask_4.shape)
            self.assertEqual(res_5.shape, mask_5.shape)
            self.assertEqual(res_6.shape, mask_6.shape)
            self.assertEqual(res_7.shape, mask_7.shape)
            self.assertEqual(res_8.shape, mask_8.shape)

    def test_dynamic_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            x = paddle.to_tensor(self.x_np)

            out_1 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, 3], random_u=0.3
            )

            out_2 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=5, random_u=0.5
            )

            out_3 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[2, 3, 5], random_u=0.7
            )

            out_4 = paddle.nn.functional.fractional_max_pool3d(
                x=x, kernel_size=2, output_size=[3, 3, 3], random_u=0.6
            )

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x, kernel_size=[2, 2, 2], output_size=[3, 3, 3], random_u=0.6
            )

            out_6 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, 3], random_u=0.6
            )

            out_7 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, None, 3], random_u=0.6
            )

            out_8 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[3, 3, None], random_u=0.6
            )

            np.testing.assert_allclose(out_1.numpy(), self.res_1_np)
            np.testing.assert_allclose(out_2.numpy(), self.res_2_np)
            np.testing.assert_allclose(out_3.numpy(), self.res_3_np)
            np.testing.assert_allclose(out_4.numpy(), self.res_4_np)
            np.testing.assert_allclose(out_5.numpy(), self.res_5_np)
            np.testing.assert_allclose(out_6.numpy(), self.res_6_np)
            np.testing.assert_allclose(out_7.numpy(), self.res_7_np)
            np.testing.assert_allclose(out_8.numpy(), self.res_8_np)


class TestFractionalMaxPool3DClassAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.x_np = np.random.random([2, 3, 7, 7, 7]).astype("float32")
        self.res_1_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], random_u=0.3
        )

        self.res_2_np = fractional_pool3d_forward(
            x=self.x_np, output_size=5, random_u=0.5
        )

        self.res_3_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], random_u=0.7
        )

        self.res_4_np = fractional_pool3d_forward(
            x=self.x_np, kernel_size=2, output_size=[3, 3, 3], random_u=0.6
        )

        self.res_5_np = fractional_pool3d_forward(
            x=self.x_np,
            kernel_size=[2, 2, 2],
            output_size=[3, 3, 3],
            random_u=0.6,
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 7, 7, 7], dtype="float32"
            )

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[3, 3, 3], random_u=0.3
            )
            out_1 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=5, random_u=0.5
            )
            out_2 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[2, 3, 5], random_u=0.7
            )
            out_3 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                kernel_size=2, output_size=[3, 3, 3], random_u=0.6
            )
            out_4 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                kernel_size=[2, 2, 2], output_size=[3, 3, 3], random_u=0.6
            )
            out_5 = fractional_max_pool(x=x)

            exe = paddle.static.Executor(place=place)
            res = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_4, out_5],
            )

            np.testing.assert_allclose(res[0], self.res_1_np)
            np.testing.assert_allclose(res[1], self.res_2_np)
            np.testing.assert_allclose(res[2], self.res_3_np)
            np.testing.assert_allclose(res[3], self.res_4_np)
            np.testing.assert_allclose(res[4], self.res_5_np)

    def test_dynamic_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            x = paddle.to_tensor(self.x_np)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[3, 3, 3], random_u=0.3
            )
            out_1 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=5, random_u=0.5
            )
            out_2 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[2, 3, 5], random_u=0.7
            )
            out_3 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                kernel_size=2, output_size=[3, 3, 3], random_u=0.6
            )
            out_4 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                kernel_size=[2, 2, 2], output_size=[3, 3, 3], random_u=0.6
            )
            out_5 = fractional_max_pool(x=x)

            np.testing.assert_allclose(out_1.numpy(), self.res_1_np)
            np.testing.assert_allclose(out_2.numpy(), self.res_2_np)
            np.testing.assert_allclose(out_3.numpy(), self.res_3_np)
            np.testing.assert_allclose(out_4.numpy(), self.res_4_np)
            np.testing.assert_allclose(out_5.numpy(), self.res_5_np)


class TestOutDtype(unittest.TestCase):
    def test_max_pool(self):
        api_fn = F.fractional_max_pool3d
        shape = [1, 3, 32, 32, 32]
        check_out_dtype(
            api_fn,
            in_specs=[(shape,)],
            expect_dtypes=['uint16', 'float16', 'float32', 'float64'],
            output_size=16,
        )


class TestFractionalMaxPool3DAPIDtype(unittest.TestCase):
    def test_dtypes(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            dtypes = ['float32', 'float64']

            if core.is_float16_supported(place):
                dtypes += ['float16']

            if use_cuda and core.is_bfloat16_supported(place):
                dtypes += ['uint16']

            for dtype in dtypes:
                np.random.seed(2023)
                x_np = np.random.random([2, 3, 7, 7, 7]).astype(dtype)
                res_np = fractional_pool3d_forward(
                    x=x_np, output_size=[3, 3, 3], random_u=0.3
                )

                x_paddle = paddle.to_tensor(x_np)
                out = paddle.nn.functional.fractional_max_pool3d(
                    x=x_paddle, output_size=[3, 3, 3], random_u=0.3
                )

                np.testing.assert_allclose(out.numpy(), res_np)


class TestFractionalMaxPool3DAPIRandomU(unittest.TestCase):
    def test_none_random_u(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            np.random.seed(2023)
            x_np = paddle.to_tensor(np.random.random([2, 3, 7, 7, 7]))

            res_np = paddle.nn.functional.fractional_max_pool3d(
                x=x_np, output_size=[3, 3, 3], random_u=None
            )

            self.assertTrue(list(res_np.shape) == [2, 3, 3, 3, 3])

    def test_error_random_u(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            np.random.seed(2023)
            x_np = paddle.to_tensor(np.random.random([2, 3, 7, 7, 7]))

            # error random_u of `<0`
            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, output_size=[3, 3, 3], random_u=-0.2
                )

            # error random_u of `0`
            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, output_size=[3, 3, 3], random_u=0
                )

            # error random_u of `1`
            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, output_size=[3, 3, 3], random_u=1
                )

            # error random_u of `>1`
            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, output_size=[3, 3, 3], random_u=1.2
                )


class TestFractionalMaxPool3DAPIErrorOutputSize(unittest.TestCase):
    def test_error_output_size(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place, device = (
                (paddle.CUDAPlace(0), 'gpu')
                if use_cuda
                else (paddle.CPUPlace(), 'cpu')
            )
            paddle.disable_static(place=place)
            paddle.set_device(device)

            np.random.seed(2023)
            x_np = np.random.random([2, 3, 7, 7, 7])

            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, output_size=[7, 7, 7], random_u=0.2
                )

            with self.assertRaises(ValueError):
                res_np = paddle.nn.functional.fractional_max_pool3d(
                    x=x_np, kernel_size=2, output_size=[6, 6, 6], random_u=0.2
                )


if __name__ == '__main__':
    unittest.main()
