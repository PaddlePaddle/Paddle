#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core

paddle.enable_static()
paddle.seed(2022)


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


class TestUnpool3DOp(OpTest):
    def setUp(self):
        self.op_type = "unpool3d"
        self.python_api = max_unpool3d_wrapper
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
            'Indices': indices.astype('int32'),
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


class TestUnpool3DOpOutput(TestUnpool3DOp):
    def init_test_case(self):
        self.unpool3d_forward_naive = unpool3dmax_forward_naive
        self.unpooling_type = "max"
        self.shape = [1, 3, 4, 5, 6]
        self.ksize = [2, 2, 2]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]
        self.output_size = [7, 9, 11]


class TestUnpool3DOpException(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_exception(self):
        def indices_size_error():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 36), shape=[1, 1, 3, 3, 4]
            ).astype("int32")
            F.max_unpool3d(data, indices, kernel_size=2, stride=2)

        def x_rank_error():
            data = paddle.rand(shape=[1, 1, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 27), shape=[1, 1, 3, 3, 3]
            ).astype("int32")
            F.max_unpool3d(data, indices, kernel_size=2, stride=2)

        def indices_rank_error():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 27), shape=[1, 3, 3, 3]
            ).astype("int32")
            F.max_unpool3d(data, indices, kernel_size=2, stride=2)

        def indices_value_error():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(195, 222), shape=[1, 1, 3, 3, 3]
            ).astype("int32")
            F.max_unpool3d(data, indices, kernel_size=2, stride=2)

        def data_format_error():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 27), shape=[1, 1, 3, 3, 3]
            ).astype("int32")
            F.max_unpool3d(
                data, indices, kernel_size=2, stride=2, data_format="NDHWC"
            )

        def data_outputsize_error():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 27), shape=[1, 1, 3, 3, 3]
            ).astype("int32")
            F.max_unpool3d(
                data,
                indices,
                kernel_size=2,
                stride=2,
                output_size=[2, 2, 3, 4, 5],
            )

        def data_outputsize_error2():
            data = paddle.rand(shape=[1, 1, 3, 3, 3])
            indices = paddle.reshape(
                paddle.arange(0, 27), shape=[1, 1, 3, 3, 3]
            )
            F.max_unpool3d(
                data, indices, kernel_size=2, stride=2, output_size=[10, 10, 10]
            )

        self.assertRaisesRegex(
            ValueError,
            r"The dimensions of Input\(X\) must equal to",
            indices_size_error,
        )
        self.assertRaisesRegex(
            ValueError,
            r"The x should have \[N, C, D, H, W\] format",
            x_rank_error,
        )
        self.assertRaisesRegex(
            ValueError,
            r"The indices should have \[N, C, D, H, W\] format",
            indices_rank_error,
        )
        if not core.is_compiled_with_cuda():
            self.assertRaisesRegex(
                ValueError,
                r"index should less than output",
                indices_value_error,
            )
        self.assertRaisesRegex(
            ValueError,
            r"Attr\(data_format\) should be 'NCDHW'",
            data_format_error,
        )
        self.assertRaisesRegex(
            ValueError, r"invalid output_size", data_outputsize_error
        )
        self.assertRaisesRegex(
            ValueError, r"invalid output_size", data_outputsize_error2
        )


class TestUnpool3DOpAPI_dygraph(unittest.TestCase):
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
            paddle.disable_static()
            input_data = np.random.rand(1, 3, 4, 4, 6)
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool3d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool3d(
                output, indices, kernel_size=2, stride=2
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


class TestUnpool3DOpAPI_dygraph2(unittest.TestCase):
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
            paddle.disable_static()
            input_data = np.random.rand(1, 3, 4, 4, 6)
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool3d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool3d(
                output, indices, kernel_size=2, stride=None
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


class TestUnpool3DOpAPI_dygraph3(unittest.TestCase):
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
            paddle.disable_static()
            input_data = np.random.rand(1, 3, 4, 4, 6)
            input_x = paddle.to_tensor(input_data)
            Pool3d = paddle.nn.MaxPool3D(
                kernel_size=2, stride=2, return_mask=True
            )
            UnPool3d = paddle.nn.MaxUnPool3D(kernel_size=2, stride=2)

            output, indices = Pool3d(input_x)
            output_unpool = UnPool3d(output, indices)
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


class TestUnpool3DOpAPI_dygraph4(unittest.TestCase):
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
            paddle.disable_static()
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
                output.astype("int64"),
                indices,
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


class TestUnpool3DOpAPI_static(unittest.TestCase):

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
                    output, indices, kernel_size=2, stride=None
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
                indices_np = np.array([[[[[5, 7], [13, 15]]]]]).astype("int32")
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
