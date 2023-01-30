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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import check_out_dtype

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
=======
from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

import paddle.fluid.core as core
from op_test import OpTest, check_out_dtype
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle.nn.functional as F


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


<<<<<<< HEAD
def adaptive_pool3d_forward(
    x, output_size, adaptive=True, data_format='NCDHW', pool_type='max'
):

    N = x.shape[0]
    C, D, H, W = (
        [x.shape[1], x.shape[2], x.shape[3], x.shape[4]]
        if data_format == 'NCDHW'
        else [x.shape[4], x.shape[1], x.shape[2], x.shape[3]]
    )

    if isinstance(output_size, int) or output_size is None:
=======
def adaptive_pool3d_forward(x,
                            output_size,
                            adaptive=True,
                            data_format='NCDHW',
                            pool_type='max'):

    N = x.shape[0]
    C, D, H, W = [x.shape[1], x.shape[2], x.shape[3], x.shape[4]] \
        if data_format == 'NCDHW' else [x.shape[4], x.shape[1], x.shape[2],x.shape[3]]

    if (isinstance(output_size, int) or output_size == None):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        H_out = output_size
        W_out = output_size
        D_out = output_size
        output_size = [D_out, H_out, W_out]
    else:
        D_out, H_out, W_out = output_size

<<<<<<< HEAD
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
=======
    if output_size[0] == None:
        output_size[0] = D
        D_out = D
    if output_size[1] == None:
        output_size[1] = H
        H_out = H
    if output_size[2] == None:
        output_size[2] = W
        W_out = W

    out = np.zeros((N, C, D_out, H_out, W_out)) if data_format=='NCDHW' \
        else np.zeros((N, D_out, H_out, W_out, C))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    for k in range(D_out):
        d_start = adaptive_start_index(k, D, output_size[0])
        d_end = adaptive_end_index(k, D, output_size[0])

        for i in range(H_out):
            h_start = adaptive_start_index(i, H, output_size[1])
            h_end = adaptive_end_index(i, H, output_size[1])

            for j in range(W_out):
                w_start = adaptive_start_index(j, W, output_size[2])
                w_end = adaptive_end_index(j, W, output_size[2])

                if data_format == 'NCDHW':
<<<<<<< HEAD
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
=======
                    x_masked = x[:, :, d_start:d_end, h_start:h_end,
                                 w_start:w_end]
                    if pool_type == 'avg':
                        field_size = (d_end - d_start) * (h_end - h_start) * (
                            w_end - w_start)
                        out[:, :, k, i,
                            j] = np.sum(x_masked, axis=(2, 3, 4)) / field_size
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
<<<<<<< HEAD
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
=======
                    x_masked = x[:, d_start:d_end, h_start:h_end,
                                 w_start:w_end, :]
                    if pool_type == 'avg':
                        field_size = (d_end - d_start) * (h_end - h_start) * (
                            w_end - w_start)
                        out[:, k, i, j, :] = np.sum(x_masked,
                                                    axis=(1, 2, 3)) / field_size
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))
    return out


class TestAdaptiveMaxPool3DAPI(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], pool_type="max"
        )

        self.res_2_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=5, pool_type="max"
        )

        self.res_3_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], pool_type="max"
        )

        self.res_4_np = adaptive_pool3d_forward(
            x=self.x_np,
            output_size=[3, 3, 3],
            pool_type="max",
            data_format="NDHWC",
        )

        self.res_5_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], pool_type="max"
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(
                name="x", shape=[2, 3, 5, 7, 7], dtype="float32"
            )

            out_1 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[3, 3, 3]
            )
=======

    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[3, 3, 3],
                                                pool_type="max")

        self.res_2_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=5,
                                                pool_type="max")

        self.res_3_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[2, 3, 5],
                                                pool_type="max")

        self.res_4_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[3, 3, 3],
                                                pool_type="max",
                                                data_format="NDHWC")

        self.res_5_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[None, 3, None],
                                                pool_type="max")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x",
                                  shape=[2, 3, 5, 7, 7],
                                  dtype="float32")

            out_1 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[3, 3, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            out_2 = paddle.nn.functional.adaptive_max_pool3d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool3d(
<<<<<<< HEAD
                x=x, output_size=[2, 3, 5]
            )

            # out_4 = paddle.nn.functional.adaptive_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[None, 3, None]
            )

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )
=======
                x=x, output_size=[2, 3, 5])

            #out_4 = paddle.nn.functional.adaptive_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[None, 3, None])

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3,
             res_5] = exe.run(fluid.default_main_program(),
                              feed={"x": self.x_np},
                              fetch_list=[out_1, out_2, out_3, out_5])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(res_1, self.res_1_np)

            assert np.allclose(res_2, self.res_2_np)

            assert np.allclose(res_3, self.res_3_np)

<<<<<<< HEAD
            # assert np.allclose(res_4, self.res_4_np)

            assert np.allclose(res_5, self.res_5_np)

    def test_dynamic_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
=======
            #assert np.allclose(res_4, self.res_4_np)

            assert np.allclose(res_5, self.res_5_np)

    def func_dynamic_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            x = paddle.to_tensor(self.x_np)

            out_1 = paddle.nn.functional.adaptive_max_pool3d(
<<<<<<< HEAD
                x=x, output_size=[3, 3, 3]
            )
=======
                x=x, output_size=[3, 3, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            out_2 = paddle.nn.functional.adaptive_max_pool3d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool3d(
<<<<<<< HEAD
                x=x, output_size=[2, 3, 5]
            )

            # out_4 = paddle.nn.functional.adaptive_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[None, 3, None]
            )
=======
                x=x, output_size=[2, 3, 5])

            #out_4 = paddle.nn.functional.adaptive_max_pool3d(
            #    x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool3d(
                x=x, output_size=[None, 3, None])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

<<<<<<< HEAD
            # assert np.allclose(out_4.numpy(), self.res_4_np)

            assert np.allclose(out_5.numpy(), self.res_5_np)


class TestAdaptiveMaxPool3DClassAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], pool_type="max"
        )

        self.res_2_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=5, pool_type="max"
        )

        self.res_3_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], pool_type="max"
        )
=======
            #assert np.allclose(out_4.numpy(), self.res_4_np)

            assert np.allclose(out_5.numpy(), self.res_5_np)

    def test_dynamic_graph(self):
        with paddle.fluid.framework._test_eager_guard():
            self.func_dynamic_graph()
        self.func_dynamic_graph()


class TestAdaptiveMaxPool3DClassAPI(unittest.TestCase):

    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[3, 3, 3],
                                                pool_type="max")

        self.res_2_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=5,
                                                pool_type="max")

        self.res_3_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[2, 3, 5],
                                                pool_type="max")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # self.res_4_np = adaptive_pool3d_forward(
        #     x=self.x_np,
        #     output_size=[3, 3, 3],
        #     pool_type="max",
        #     data_format="NDHWC")

<<<<<<< HEAD
        self.res_5_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], pool_type="max"
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(
                name="x", shape=[2, 3, 5, 7, 7], dtype="float32"
            )

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
                output_size=[3, 3, 3]
            )
=======
        self.res_5_np = adaptive_pool3d_forward(x=self.x_np,
                                                output_size=[None, 3, None],
                                                pool_type="max")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x",
                                  shape=[2, 3, 5, 7, 7],
                                  dtype="float32")

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
                output_size=[3, 3, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_1 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(output_size=5)
            out_2 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
<<<<<<< HEAD
                output_size=[2, 3, 5]
            )
=======
                output_size=[2, 3, 5])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_3 = adaptive_max_pool(x=x)

            #     adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
            #         output_size=[3, 3, 3], data_format="NDHWC")
            #     out_4 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
<<<<<<< HEAD
                output_size=[None, 3, None]
            )
            out_5 = adaptive_max_pool(x=x)

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )
=======
                output_size=[None, 3, None])
            out_5 = adaptive_max_pool(x=x)

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3,
             res_5] = exe.run(fluid.default_main_program(),
                              feed={"x": self.x_np},
                              fetch_list=[out_1, out_2, out_3, out_5])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(res_1, self.res_1_np)

            assert np.allclose(res_2, self.res_2_np)

            assert np.allclose(res_3, self.res_3_np)

            #     assert np.allclose(res_4, self.res_4_np)

            assert np.allclose(res_5, self.res_5_np)

    def test_dynamic_graph(self):
<<<<<<< HEAD
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
=======
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            x = paddle.to_tensor(self.x_np)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
<<<<<<< HEAD
                output_size=[3, 3, 3]
            )
=======
                output_size=[3, 3, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_1 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(output_size=5)
            out_2 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
<<<<<<< HEAD
                output_size=[2, 3, 5]
            )
=======
                output_size=[2, 3, 5])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_3 = adaptive_max_pool(x=x)

            #     adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
            #         output_size=[3, 3, 3], data_format="NDHWC")
            #     out_4 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool3D(
<<<<<<< HEAD
                output_size=[None, 3, None]
            )
=======
                output_size=[None, 3, None])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_5 = adaptive_max_pool(x=x)

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

            #     assert np.allclose(out_4.numpy(), self.res_4_np)

            assert np.allclose(out_5.numpy(), self.res_5_np)


class TestOutDtype(unittest.TestCase):
<<<<<<< HEAD
    def test_max_pool(self):
        api_fn = F.adaptive_max_pool3d
        shape = [1, 3, 32, 32, 32]
        check_out_dtype(
            api_fn,
            in_specs=[(shape,)],
            expect_dtypes=['float32', 'float64'],
            output_size=16,
        )
=======

    def test_max_pool(self):
        api_fn = F.adaptive_max_pool3d
        shape = [1, 3, 32, 32, 32]
        check_out_dtype(api_fn,
                        in_specs=[(shape, )],
                        expect_dtypes=['float32', 'float64'],
                        output_size=16)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
