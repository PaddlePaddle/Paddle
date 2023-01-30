# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
def adaptive_pool2d_forward(
    x, output_size, data_format='NCHW', pool_type="max"
):

    N = x.shape[0]
    C, H, W = (
        [x.shape[1], x.shape[2], x.shape[3]]
        if data_format == 'NCHW'
        else [x.shape[3], x.shape[1], x.shape[2]]
    )

    if isinstance(output_size, int) or output_size is None:
=======
def adaptive_pool2d_forward(x,
                            output_size,
                            data_format='NCHW',
                            pool_type="max"):

    N = x.shape[0]
    C, H, W = [x.shape[1], x.shape[2], x.shape[3]] if data_format == 'NCHW' \
        else [x.shape[3], x.shape[1], x.shape[2]]

    if (isinstance(output_size, int) or output_size == None):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        H_out = output_size
        W_out = output_size
        output_size = [H_out, W_out]
    else:
        H_out, W_out = output_size

<<<<<<< HEAD
    if output_size[0] is None:
        output_size[0] = H
        H_out = H
    if output_size[1] is None:
        output_size[1] = W
        W_out = W

    out = (
        np.zeros((N, C, H_out, W_out))
        if data_format == 'NCHW'
        else np.zeros((N, H_out, W_out, C))
    )
=======
    if output_size[0] == None:
        output_size[0] = H
        H_out = H
    if output_size[1] == None:
        output_size[1] = W
        W_out = W

    out = np.zeros((N, C, H_out, W_out)) if data_format=='NCHW' \
        else np.zeros((N, H_out, W_out, C))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    for i in range(H_out):
        in_h_start = adaptive_start_index(i, H, output_size[0])
        in_h_end = adaptive_end_index(i, H, output_size[0])

        for j in range(W_out):
            in_w_start = adaptive_start_index(j, W, output_size[1])
            in_w_end = adaptive_end_index(j, W, output_size[1])

            if data_format == 'NCHW':
                x_masked = x[:, :, in_h_start:in_h_end, in_w_start:in_w_end]
                if pool_type == 'avg':
<<<<<<< HEAD
                    field_size = (in_h_end - in_h_start) * (
                        in_w_end - in_w_start
                    )
=======
                    field_size = ((in_h_end - in_h_start) *
                                  (in_w_end - in_w_start))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / field_size
                elif pool_type == 'max':
                    out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
            elif data_format == 'NHWC':
                x_masked = x[:, in_h_start:in_h_end, in_w_start:in_w_end, :]
                if pool_type == 'avg':
<<<<<<< HEAD
                    field_size = (in_h_end - in_h_start) * (
                        in_w_end - in_w_start
                    )
=======
                    field_size = ((in_h_end - in_h_start) *
                                  (in_w_end - in_w_start))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    out[:, i, j, :] = np.sum(x_masked, axis=(1, 2)) / field_size
                elif pool_type == 'max':
                    out[:, i, j, :] = np.max(x_masked, axis=(1, 2))
    return out


class TestAdaptiveMaxPool2DAPI(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.x_np = np.random.random([2, 3, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[3, 3], pool_type="max"
        )

        self.res_2_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=5, pool_type="max"
        )

        self.res_3_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[2, 5], pool_type="max"
        )
=======

    def setUp(self):
        self.x_np = np.random.random([2, 3, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[3, 3],
                                                pool_type="max")

        self.res_2_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=5,
                                                pool_type="max")

        self.res_3_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[2, 5],
                                                pool_type="max")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        """
        self.res_4_np = adaptive_pool2d_forward(
            x=self.x_np,
            output_size=[3, 3],
            pool_type="max",
            data_format="NHWC")
        """
<<<<<<< HEAD
        self.res_5_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[None, 3], pool_type="max"
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
=======
        self.res_5_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[None, 3],
                                                pool_type="max")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x", shape=[2, 3, 7, 7], dtype="float32")

<<<<<<< HEAD
            out_1 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[3, 3]
            )

            out_2 = paddle.nn.functional.adaptive_max_pool2d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[2, 5]
            )

            # out_4 = paddle.nn.functional.adaptive_max_pool2d(
            #    x=x, output_size=[3, 3], data_format="NHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[None, 3]
            )

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )
=======
            out_1 = paddle.nn.functional.adaptive_max_pool2d(x=x,
                                                             output_size=[3, 3])

            out_2 = paddle.nn.functional.adaptive_max_pool2d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool2d(x=x,
                                                             output_size=[2, 5])

            #out_4 = paddle.nn.functional.adaptive_max_pool2d(
            #    x=x, output_size=[3, 3], data_format="NHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[None, 3])

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
=======
            #assert np.allclose(res_4, self.res_4_np)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

<<<<<<< HEAD
            out_1 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, return_mask=False, output_size=[3, 3]
            )

            out_2 = paddle.nn.functional.adaptive_max_pool2d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[2, 5]
            )

            # out_4 = paddle.nn.functional.adaptive_max_pool2d(
            #    x=x, output_size=[3, 3], data_format="NHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[None, 3]
            )
=======
            out_1 = paddle.nn.functional.adaptive_max_pool2d(x=x,
                                                             return_mask=False,
                                                             output_size=[3, 3])

            out_2 = paddle.nn.functional.adaptive_max_pool2d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_max_pool2d(x=x,
                                                             output_size=[2, 5])

            #out_4 = paddle.nn.functional.adaptive_max_pool2d(
            #    x=x, output_size=[3, 3], data_format="NHWC")

            out_5 = paddle.nn.functional.adaptive_max_pool2d(
                x=x, output_size=[None, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

<<<<<<< HEAD
            # assert np.allclose(out_4.numpy(), self.res_4_np)
=======
            #assert np.allclose(out_4.numpy(), self.res_4_np)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(out_5.numpy(), self.res_5_np)


class TestAdaptiveMaxPool2DClassAPI(unittest.TestCase):
<<<<<<< HEAD
    def setUp(self):
        self.x_np = np.random.random([2, 3, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[3, 3], pool_type="max"
        )

        self.res_2_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=5, pool_type="max"
        )

        self.res_3_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[2, 5], pool_type="max"
        )

        # self.res_4_np = adaptive_pool2d_forward(
=======

    def setUp(self):
        self.x_np = np.random.random([2, 3, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[3, 3],
                                                pool_type="max")

        self.res_2_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=5,
                                                pool_type="max")

        self.res_3_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[2, 5],
                                                pool_type="max")

        #self.res_4_np = adaptive_pool2d_forward(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        #    x=self.x_np,
        #    output_size=[3, 3],
        #    pool_type="max",
        #    data_format="NHWC")

<<<<<<< HEAD
        self.res_5_np = adaptive_pool2d_forward(
            x=self.x_np, output_size=[None, 3], pool_type="max"
        )

    def test_static_graph(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
=======
        self.res_5_np = adaptive_pool2d_forward(x=self.x_np,
                                                output_size=[None, 3],
                                                pool_type="max")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x", shape=[2, 3, 7, 7], dtype="float32")

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=[3, 3])
            out_1 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=5)
            out_2 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=[2, 5])
            out_3 = adaptive_max_pool(x=x)

            #    adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
            #        output_size=[3, 3], data_format="NHWC")
            #    out_4 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
<<<<<<< HEAD
                output_size=[None, 3]
            )
            out_5 = adaptive_max_pool(x=x)

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_5],
            )
=======
                output_size=[None, 3])
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

<<<<<<< HEAD
            # assert np.allclose(res_4, self.res_4_np)
=======
            #assert np.allclose(res_4, self.res_4_np)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=[3, 3])
            out_1 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=5)
            out_2 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=[2, 5])
            out_3 = adaptive_max_pool(x=x)

<<<<<<< HEAD
            # adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
            #    output_size=[3, 3], data_format="NHWC")
            # out_4 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
                output_size=[None, 3]
            )
=======
            #adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
            #    output_size=[3, 3], data_format="NHWC")
            #out_4 = adaptive_max_pool(x=x)

            adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(
                output_size=[None, 3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            out_5 = adaptive_max_pool(x=x)

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

<<<<<<< HEAD
            # assert np.allclose(out_4.numpy(), self.res_4_np)
=======
            #assert np.allclose(out_4.numpy(), self.res_4_np)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            assert np.allclose(out_5.numpy(), self.res_5_np)


class TestOutDtype(unittest.TestCase):
<<<<<<< HEAD
    def test_max_pool(self):
        api_fn = F.adaptive_max_pool2d
        shape = [1, 3, 32, 32]
        check_out_dtype(
            api_fn,
            in_specs=[(shape,)],
            expect_dtypes=['float32', 'float64'],
            output_size=16,
        )
=======

    def test_max_pool(self):
        api_fn = F.adaptive_max_pool2d
        shape = [1, 3, 32, 32]
        check_out_dtype(api_fn,
                        in_specs=[(shape, )],
                        expect_dtypes=['float32', 'float64'],
                        output_size=16)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
