#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.nn.functional as F

paddle.enable_static()
paddle.seed(2024)


def _unpool_output_size(x, kernel_size, stride, padding, output_size):
    input_size = x.shape
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append(
            (input_size[-len(kernel_size) + d] - 1) * stride[d]
            + kernel_size[d]
            - 2 * padding[d]
        )
    if output_size is None:
        ret = default_size
    else:
        ret = output_size
    return ret


def unpool1dmax_forward_naive(
    input, indices, ksize, strides, paddings, output_size
):
    s0, s1, s2 = input.shape
    output_size = _unpool_output_size(
        input, ksize, strides, paddings, output_size
    )
    out_lsize = output_size[0]
    out = np.zeros((s0, s1, out_lsize))
    for nidx in range(s0):
        for cidx in range(s1):
            for l in range(s2):
                index = indices[nidx, cidx, l]
                lidx = index % out_lsize
                out[nidx, cidx, lidx] = input[nidx, cidx, l]

    return out


def unpool2dmax_forward_naive(
    input, indices, ksize, strides, paddings, output_size
):
    s0, s1, s2, s3 = input.shape
    output_size = _unpool_output_size(
        input, ksize, strides, paddings, output_size
    )
    out_hsize = output_size[0]
    out_wsize = output_size[1]
    out = np.zeros((s0, s1, out_hsize, out_wsize))
    for nidx in range(s0):
        for cidx in range(s1):
            for h in range(s2):
                for w in range(s3):
                    index = indices[nidx, cidx, h, w]
                    hidx = (index - index % out_wsize) // out_wsize
                    widx = index % out_wsize
                    out[nidx, cidx, hidx, widx] = input[nidx, cidx, h, w]

    return out


def unpool3dmax_forward_naive(
    input, indices, ksize, strides, paddings, output_size
):
    s0, s1, s2, s3, s4 = input.shape
    output_size = _unpool_output_size(
        input, ksize, strides, paddings, output_size
    )
    out_dsize = output_size[0]
    out_hsize = output_size[1]
    out_wsize = output_size[2]
    out = np.zeros((s0, s1, out_dsize, out_hsize, out_wsize))
    for nidx in range(s0):
        for cidx in range(s1):
            for d in range(s2):
                for h in range(s3):
                    for w in range(s4):
                        index = indices[nidx, cidx, d, h, w]
                        didx = index // (out_wsize * out_hsize)
                        hidx = (
                            index - didx * out_hsize * out_wsize
                        ) // out_wsize
                        widx = (
                            index - didx * out_hsize * out_wsize
                        ) % out_wsize
                        out[nidx, cidx, didx, hidx, widx] = input[
                            nidx, cidx, d, h, w
                        ]

    return out


def max_unpool2d_wrapper(
    x,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
    data_format="NCHW",
    name=None,
):
    out = paddle.nn.functional.max_unpool2d(
        x,
        indices,
        kernel_size,
        stride=stride,
        padding=padding,
        data_format=data_format,
        output_size=output_size,
        name=name,
    )
    return out


def max_unpool3d_wrapper(
    x,
    indices,
    kernel_size,
    stride=None,
    padding=0,
    output_size=None,
    data_format="NCDHW",
    name=None,
):
    out = paddle.nn.functional.max_unpool3d(
        x,
        indices,
        kernel_size,
        stride=stride,
        padding=padding,
        data_format=data_format,
        output_size=output_size,
        name=name,
    )
    return out


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "unpool"
        self.python_api = max_unpool2d_wrapper
        self.indices_dtype = "int64"
        self.init_test_case()
        input = np.random.randint(0, 100, self.shape)
        nsize, csize, hsize, wsize = input.shape
        self.output_size = _unpool_output_size(
            input, self.ksize, self.strides, self.paddings, self.output_size
        )
        indices = np.random.permutation(
            np.arange(0, self.output_size[0] * self.output_size[1])
        )[: hsize * wsize]
        indices = np.reshape(indices, [hsize, wsize])
        idx_list = []
        for n in range(nsize):
            c_list = []
            for c in range(csize):
                c_list.append(indices.tolist())
            idx_list.append(c_list)
        indices = np.array(idx_list)

        output = self.unpool2d_forward_naive(
            input,
            indices,
            self.ksize,
            self.strides,
            self.paddings,
            self.output_size,
        ).astype("float64")

        self.inputs = {
            'X': input.astype('float64'),
            'Indices': indices.astype(self.indices_dtype),
        }
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'unpooling_type': self.unpooling_type,
            'output_size': self.output_size,
        }
        self.outputs = {'Out': output.astype('float64')}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [2, 4, 7, 8]
        self.ksize = [2, 2]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = None


class TestUnpoolOpcase1(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = None


class TestUnpoolOpOutputsize(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [12, 12]


class TestUnpoolOpOutput(TestUnpoolOp):
    def init_test_case(self):
        self.unpool2d_forward_naive = unpool2dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [3, 2, 5, 5]
        self.ksize = [4, 4]
        self.strides = [2, 2]
        self.paddings = [0, 0]
        self.output_size = [12, 12]


class TestUnpool3DOp(OpTest):
    def setUp(self):
        self.op_type = "unpool3d"
        self.python_api = max_unpool3d_wrapper
        self.indices_dtype = "int64"
        self.init_test_case()
        inputs = np.random.randint(0, 100, self.shape)
        nsize, csize, dsize, hsize, wsize = inputs.shape
        self.output_size = _unpool_output_size(
            inputs, self.ksize, self.strides, self.paddings, self.output_size
        )
        indices = np.random.permutation(
            np.arange(
                0,
                self.output_size[0] * self.output_size[1] * self.output_size[2],
            )
        )[: dsize * hsize * wsize]
        indices = np.reshape(indices, [dsize, hsize, wsize])
        idx_list = []
        for n in range(nsize):
            c_list = []
            for c in range(csize):
                c_list.append(indices.tolist())
            idx_list.append(c_list)
        indices = np.array(idx_list)

        output = self.unpool3d_forward_naive(
            inputs,
            indices,
            self.ksize,
            self.strides,
            self.paddings,
            self.output_size,
        ).astype("float64")

        self.inputs = {
            'X': inputs.astype('float64'),
            'Indices': indices.astype(self.indices_dtype),
        }
        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'unpooling_type': self.unpooling_type,
            'output_size': self.output_size,
        }
        self.outputs = {'Out': output.astype('float64')}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)

    def init_test_case(self):
        self.unpool3d_forward_naive = unpool3dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [1, 1, 4, 5, 6]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.output_size = None


class TestUnpool3DOpcase1(TestUnpool3DOp):
    def init_test_case(self):
        self.unpool3d_forward_naive = unpool3dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [1, 3, 4, 5, 6]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.output_size = None


class TestUnpool3DOpcase2(TestUnpool3DOp):
    def init_test_case(self):
        self.unpool3d_forward_naive = unpool3dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [1, 3, 4, 5, 6]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.output_size = None
        self.indices_dtype = "int64"


class TestUnpool3DOpOutput(TestUnpool3DOp):
    def init_test_case(self):
        self.unpool3d_forward_naive = unpool3dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [1, 3, 4, 5, 6]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.output_size = [7, 9, 11]


class TestUnpool1DAPI_dy(unittest.TestCase):
    def test_case(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            paddle.disable_static(place)
            input_data = np.arange(3 * 16).reshape([1, 3, 16]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool1d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool1d(
                output,
                indices.astype("int64"),
                kernel_size=2,
                stride=2,
                output_size=input_x.shape,
            )
            expected_output_unpool = unpool1dmax_forward_naive(
                output.numpy(), indices.numpy(), [2], [2], [0], [16]
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool1DAPI_st(unittest.TestCase):

    def test_case(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_data = np.array(
                    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]
                ).astype("float32")
                x = paddle.static.data(
                    name='x', shape=[1, 3, 4], dtype='float32'
                )
                output, indices = F.max_pool1d(
                    x, kernel_size=2, stride=2, return_mask=True
                )
                output_unpool = F.max_unpool1d(
                    output, indices.astype("int64"), kernel_size=2, stride=None
                )

                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    feed={"x": input_data},
                    fetch_list=[output_unpool],
                    return_numpy=True,
                )
                pool1d_out_np = np.array(
                    [[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]
                ).astype("float32")
                indices_np = np.array([[[1, 3], [1, 3], [1, 3]]]).astype(
                    "int64"
                )
                expected_output_unpool = unpool1dmax_forward_naive(
                    pool1d_out_np, indices_np, [2], [2], [0], [4]
                )
                np.testing.assert_allclose(
                    fetches[0], expected_output_unpool, rtol=1e-05
                )


class TestUnpool2DAPI_dy(unittest.TestCase):
    def test_case(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            paddle.disable_static(place)
            input_data = np.array(
                [
                    [
                        [
                            [1, 2, 3, 4, 5],
                            [6, 7, 8, 9, 10],
                            [11, 12, 13, 14, 15],
                            [16, 17, 18, 19, 20],
                        ]
                    ]
                ]
            ).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool2d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            out_pp = F.max_unpool2d(
                output,
                indices.astype("int64"),
                kernel_size=2,
                stride=None,
                output_size=input_x.shape,
            )
            output_np = output.numpy()
            indices_np = indices.numpy()
            expect_res = unpool2dmax_forward_naive(
                output_np, indices_np, [2, 2], [2, 2], [0, 0], [4, 5]
            ).astype("float64")
            np.testing.assert_allclose(out_pp.numpy(), expect_res, rtol=1e-05)

        paddle.enable_static()


class TestUnpool2DAPI_st(unittest.TestCase):

    def test_case(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_data = np.array(
                    [
                        [
                            [
                                [1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12],
                                [13, 14, 15, 16],
                            ]
                        ]
                    ]
                ).astype("float32")

                x = paddle.static.data(
                    name="x", shape=[1, 1, 4, 4], dtype="float32"
                )
                output, indices = F.max_pool2d(
                    x, kernel_size=2, stride=2, return_mask=True
                )
                unpool_out = F.max_unpool2d(
                    output,
                    indices.astype("int64"),
                    kernel_size=2,
                    stride=None,
                    output_size=(5, 5),
                )
                exe = paddle.static.Executor(place)

                results = exe.run(
                    feed={"x": input_data},
                    fetch_list=[unpool_out],
                    return_numpy=True,
                )

                pool_out_np = np.array([[[[6.0, 8.0], [14.0, 16.0]]]]).astype(
                    "float32"
                )
                indices_np = np.array([[[[5, 7], [13, 15]]]]).astype("int64")
                expect_res = unpool2dmax_forward_naive(
                    pool_out_np, indices_np, [2, 2], [2, 2], [0, 0], [5, 5]
                ).astype("float64")
                np.testing.assert_allclose(results[0], expect_res, rtol=1e-05)


class TestUnpool3DAPI_dy(unittest.TestCase):
    def test_case(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            paddle.disable_static(place)
            input_data = (
                np.arange(3 * 4 * 4 * 6)
                .reshape([1, 3, 4, 4, 6])
                .astype("float32")
            )
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool3d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool3d(
                output,
                indices.astype("int64"),
                kernel_size=2,
                stride=2,
                output_size=input_x.shape,
            )
            expected_output_unpool = unpool3dmax_forward_naive(
                output.numpy(),
                indices.numpy(),
                [2, 2, 2],
                [2, 2, 2],
                [0, 0, 0],
                [4, 4, 6],
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool3DAPI_st2(unittest.TestCase):

    def test_case(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.base.core.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.base.core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input_data = np.array(
                    [
                        [
                            [
                                [
                                    [1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16],
                                ],
                                [
                                    [1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16],
                                ],
                            ]
                        ]
                    ]
                ).astype("float32")
                x = paddle.static.data(
                    name='x', shape=[1, 1, 2, 4, 4], dtype='float32'
                )
                output, indices = F.max_pool3d(
                    x, kernel_size=2, stride=2, return_mask=True
                )
                output_unpool = F.max_unpool3d(
                    output, indices.astype("int64"), kernel_size=2, stride=None
                )

                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    feed={"x": input_data},
                    fetch_list=[output_unpool],
                    return_numpy=True,
                )
                pool3d_out_np = np.array(
                    [[[[[6.0, 8.0], [14.0, 16.0]]]]]
                ).astype("float32")
                indices_np = np.array([[[[[5, 7], [13, 15]]]]]).astype("int64")
                expected_output_unpool = unpool3dmax_forward_naive(
                    pool3d_out_np,
                    indices_np,
                    [2, 2, 2],
                    [2, 2, 2],
                    [0, 0, 0],
                    [2, 4, 4],
                )
                np.testing.assert_allclose(
                    fetches[0], expected_output_unpool, rtol=1e-05
                )


if __name__ == '__main__':
    unittest.main()
