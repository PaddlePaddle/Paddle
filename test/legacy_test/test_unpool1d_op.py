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

import paddle
import paddle.nn.functional as F

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


class TestUnpool1DOpAPI_dygraph(unittest.TestCase):
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
            input_data = np.random.rand(1, 3, 16)
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool1d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool1d(
                output, indices, kernel_size=2, stride=2
            )
            expected_output_unpool = unpool1dmax_forward_naive(
                output.numpy(), indices.numpy(), [2], [2], [0], [16]
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool1DOpAPI_dygraph2(unittest.TestCase):
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
            input_data = np.random.rand(1, 3, 16)
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool1d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool1d(
                output, indices, kernel_size=2, stride=None
            )
            expected_output_unpool = unpool1dmax_forward_naive(
                output.numpy(), indices.numpy(), [2], [2], [0], [16]
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool1DOpAPI_dygraph3(unittest.TestCase):
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
            input_data = np.random.rand(1, 3, 16)
            input_x = paddle.to_tensor(input_data)
            Pool1d = paddle.nn.MaxPool1D(
                kernel_size=2, stride=2, return_mask=True
            )
            UnPool1d = paddle.nn.MaxUnPool1D(kernel_size=2, stride=2)

            output, indices = Pool1d(input_x)
            output_unpool = UnPool1d(output, indices)
            expected_output_unpool = unpool1dmax_forward_naive(
                output.numpy(), indices.numpy(), [2], [2], [0], [16]
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool1DOpAPI_dygraph4(unittest.TestCase):
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
            input_data = np.arange(3 * 16).reshape([1, 3, 16]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool1d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool1d(
                output.astype("int64"),
                indices,
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


class TestUnpool1DOpAPI_dygraph5(unittest.TestCase):
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
            input_data = np.arange(3 * 16).reshape([1, 3, 16]).astype("float32")
            input_x = paddle.to_tensor(input_data)
            output, indices = F.max_pool1d(
                input_x, kernel_size=2, stride=2, return_mask=True
            )
            output_unpool = F.max_unpool1d(
                output.astype("int64"),
                indices,
                kernel_size=2,
                stride=2,
                output_size=tuple(input_x.shape),
            )
            expected_output_unpool = unpool1dmax_forward_naive(
                output.numpy(), indices.numpy(), [2], [2], [0], [16]
            )
            np.testing.assert_allclose(
                output_unpool.numpy(), expected_output_unpool, rtol=1e-05
            )

        paddle.enable_static()


class TestUnpool1DOpAPI_static(unittest.TestCase):

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
                    output, indices, kernel_size=2, stride=None
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
                    "int32"
                )
                expected_output_unpool = unpool1dmax_forward_naive(
                    pool1d_out_np, indices_np, [2], [2], [0], [4]
                )
                np.testing.assert_allclose(
                    fetches[0], expected_output_unpool, rtol=1e-05
                )


if __name__ == '__main__':
    unittest.main()
