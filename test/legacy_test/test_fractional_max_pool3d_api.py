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


def fractional_rational_u(u, alpha, input, output):
    base = input // output

    u_max1 = (base + 2) / alpha - 1
    u_max2 = (input + 1 - base) / alpha - (output - 1)
    max_u = min(u_max1, u_max2)

    return u * max_u


def fractional_start_index(idx, alpha, u):
    return int(np.ceil(alpha * (idx + u) - 1))


def fractional_end_index(idx, alpha, u):
    return int(np.ceil(alpha * (idx + 1 + u) - 1))


def fractional_pool3d_forward(
    x, output_size, random_u=None, data_format='NCDHW', pool_type='max'
):
    N = x.shape[0]
    C, D, H, W = (
        [x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        if data_format == 'NCDHW'
        else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]
    )

    if isinstance(output_size, int) or output_size is None:
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

    alpha_depth = D / D_out
    alpha_height = H / H_out
    alpha_width = W / W_out

    u_depth = fractional_rational_u(u, alpha_depth, D, D_out)
    u_height = fractional_rational_u(u, alpha_height, H, H_out)
    u_width = fractional_rational_u(u, alpha_width, W, W_out)

    for k in range(D_out):
        d_start = fractional_start_index(k, alpha_depth, u_depth)
        d_end = fractional_end_index(k, alpha_depth, u_depth)
        d_start = max(d_start, 0)
        d_end = min(d_end, D)

        for i in range(H_out):
            h_start = fractional_start_index(i, alpha_height, u_height)
            h_end = fractional_end_index(i, alpha_height, u_height)
            h_start = max(h_start, 0)
            h_end = min(h_end, H)

            for j in range(W_out):
                w_start = fractional_start_index(j, alpha_width, u_width)
                w_end = fractional_end_index(j, alpha_width, u_width)
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
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
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
            output_size=[3, 3, 3],
            pool_type="max",
            data_format="NDHWC",
            random_u=0.1,
        )

        self.res_5_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], random_u=0.6
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 5, 7, 7], dtype="float32"
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

            # out_4 = paddle.nn.functional.fractional_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC", random_u=0.1)

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, None], random_u=0.6
            )

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )

            np.testing.assert_allclose(res_1, self.res_1_np)

            np.testing.assert_allclose(res_2, self.res_2_np)

            np.testing.assert_allclose(res_3, self.res_3_np)

            # np.testing.assert_allclose(res_4, self.res_4_np)

            np.testing.assert_allclose(res_5, self.res_5_np)

    def test_static_graph_return_mask(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 5, 7, 7], dtype="float32"
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

            # out_4 = paddle.nn.functional.fractional_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NHWC", return_mask=True, random_u=0.1)

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, None], return_mask=True, random_u=0.6
            )

            exe = paddle.static.Executor(place=place)
            [
                res_1,
                mask_1,
                res_2,
                mask_2,
                res_3,
                mask_3,
                res_5,
                mask_5,
            ] = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )

            self.assertEqual(res_1.shape, mask_1.shape)

            self.assertEqual(res_2.shape, mask_2.shape)

            self.assertEqual(res_3.shape, mask_3.shape)

            # self.assertEqual(res_4.shape, mask_4.shape)

            self.assertEqual(res_5.shape, mask_5.shape)

    def test_dynamic_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
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

            # out_4 = paddle.nn.functional.fractional_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC", random_u=0.1)

            out_5 = paddle.nn.functional.fractional_max_pool3d(
                x=x, output_size=[None, 3, None], random_u=0.6
            )

            np.testing.assert_allclose(out_1.numpy(), self.res_1_np)

            np.testing.assert_allclose(out_2.numpy(), self.res_2_np)

            np.testing.assert_allclose(out_3.numpy(), self.res_3_np)

            # np.testing.assert_allclose(out_4.numpy(), self.res_4_np)

            np.testing.assert_allclose(out_5.numpy(), self.res_5_np)


class TestFractionalMaxPool3DClassAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], random_u=0.3
        )

        self.res_2_np = fractional_pool3d_forward(
            x=self.x_np, output_size=5, random_u=0.5
        )

        self.res_3_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], random_u=0.7
        )

        # self.res_4_np = fractional_pool3d_forward(
        #     x=self.x_np,
        #     output_size=[3, 3, 3],
        #     pool_type="max",
        #     data_format="NDHWC",
        #     random_u=0.1
        # )

        self.res_5_np = fractional_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], random_u=0.6
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.static.data(
                name="x", shape=[2, 3, 5, 7, 7], dtype="float32"
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

            #     fractional_max_pool = paddle.nn.FractionalMaxPool3D(
            #         output_size=[3, 3, 3], data_format="NDHWC", random_u=0.1)
            #     out_4 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[None, 3, None], random_u=0.6
            )
            out_5 = fractional_max_pool(x=x)

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                base.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )

            np.testing.assert_allclose(res_1, self.res_1_np)

            np.testing.assert_allclose(res_2, self.res_2_np)

            np.testing.assert_allclose(res_3, self.res_3_np)

            #     assert np.allclose(res_4, self.res_4_np)

            np.testing.assert_allclose(res_5, self.res_5_np)

    def test_dynamic_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
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

            #     fractional_max_pool = paddle.nn.FractionalMaxPool3D(
            #         output_size=[3, 3, 3], data_format="NDHWC", random_u=0.1)
            #     out_4 = fractional_max_pool(x=x)

            fractional_max_pool = paddle.nn.FractionalMaxPool3D(
                output_size=[None, 3, None], random_u=0.6
            )
            out_5 = fractional_max_pool(x=x)

            np.testing.assert_allclose(out_1.numpy(), self.res_1_np)

            np.testing.assert_allclose(out_2.numpy(), self.res_2_np)

            np.testing.assert_allclose(out_3.numpy(), self.res_3_np)

            #     assert np.allclose(out_4.numpy(), self.res_4_np)

            np.testing.assert_allclose(out_5.numpy(), self.res_5_np)


class TestOutDtype(unittest.TestCase):
    def test_max_pool(self):
        api_fn = F.fractional_max_pool3d
        shape = [1, 3, 32, 32, 32]
        check_out_dtype(
            api_fn,
            in_specs=[(shape,)],
            expect_dtypes=['float32', 'float64'],
            output_size=16,
        )


if __name__ == '__main__':
    unittest.main()
