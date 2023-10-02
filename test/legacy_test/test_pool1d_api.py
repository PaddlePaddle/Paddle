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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool1D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=False,
    adaptive=False,
    data_type=np.float64,
):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (
            (L - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (L - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        out[:, :, i] = np.max(x_masked, axis=(2))
    return out


def avg_pool1D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=0,
    ceil_mode=False,
    exclusive=False,
    adaptive=False,
    data_type=np.float64,
):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (
            (L - ksize[0] + 2 * paddings[0] + strides[0] - 1) // strides[0] + 1
            if ceil_mode
            else (L - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        )

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        field_size = (
            (r_end - r_start) if (exclusive or adaptive) else (ksize[0])
        )
        if data_type == np.int8 or data_type == np.uint8:
            out[:, :, i] = (
                np.rint(np.sum(x_masked, axis=(2, 3)) / field_size)
            ).astype(data_type)
        else:
            out[:, :, i] = (np.sum(x_masked, axis=(2)) / field_size).astype(
                data_type
            )
    return out


class TestPool1D_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_avg_static_results(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32], dtype="float32"
            )
            result = F.avg_pool1d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0], ceil_mode=False
            )

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_avg_static_results_fp16(self, place):
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32], dtype="float16"
            )
            result = F.avg_pool1d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32]).astype("float16")
            result_np = avg_pool1D_forward_naive(
                input_np,
                ksize=[2],
                strides=[2],
                paddings=[0],
                ceil_mode=False,
            )

            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )
                np.testing.assert_allclose(fetches[0], result_np, rtol=1e-03)

    def check_avg_dygraph_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = F.avg_pool1d(input, kernel_size=2, stride=2, padding=[0])

            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool1d_dg = paddle.nn.layer.AvgPool1D(
                kernel_size=2, stride=None, padding=0
            )
            result = avg_pool1d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_padding_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = F.avg_pool1d(
                input, kernel_size=2, stride=2, padding=[1], exclusive=True
            )

            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[1], exclusive=False
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool1d_dg = paddle.nn.AvgPool1D(
                kernel_size=2, stride=None, padding=1, exclusive=True
            )

            result = avg_pool1d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_static_results(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32], dtype="float32"
            )
            result = F.max_pool1d(input, kernel_size=2, stride=2, padding=[0])

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_max_dygraph_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = F.max_pool1d(input, kernel_size=2, stride=2, padding=0)

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool1d_dg = paddle.nn.layer.MaxPool1D(
                kernel_size=2, stride=None, padding=0
            )
            result = max_pool1d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_return_index_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result, index = F.max_pool1d(
                input, kernel_size=2, stride=2, padding=0, return_mask=True
            )

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool1d_dg = paddle.nn.layer.MaxPool1D(
                kernel_size=2, stride=None, padding=0
            )
            result = max_pool1d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_padding_same(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = F.max_pool1d(
                input, kernel_size=2, stride=2, padding="SAME"
            )

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_padding_same(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = F.avg_pool1d(
                input, kernel_size=2, stride=2, padding="SAME"
            )

            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0]
            )

            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def test_pool1d(self):
        for place in self.places:
            self.check_max_dygraph_results(place)
            self.check_avg_dygraph_results(place)
            self.check_max_static_results(place)
            self.check_avg_static_results(place)
            self.check_max_dygraph_padding_same(place)
            self.check_avg_dygraph_padding_same(place)
            self.check_max_dygraph_return_index_results(place)
            self.check_avg_static_results_fp16(place)


class TestPool1DError_API(unittest.TestCase):
    def test_error_api(self):
        def run1():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[2]]
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding
                )

        self.assertRaises(ValueError, run1)

        def run2():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[2]]
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding
                )

        self.assertRaises(ValueError, run2)

        def run3():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "padding"
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding
                )

        self.assertRaises(ValueError, run3)

        def run4():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.max_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run4)

        def run5():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.max_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run5)

        def run6():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run6)

        def run7():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "paddle"
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run7)

        def run_kernel_out_of_range():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = 0
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=-1,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run_kernel_out_of_range)

        def run_stride_out_of_range():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = 0
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=0,
                    padding=padding,
                    ceil_mode=True,
                )

        self.assertRaises(ValueError, run_stride_out_of_range)

        def run_zero_stride():
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [1, 1, 1]), dtype='float32'
                )
                out = F.max_pool1d(
                    x, 1, stride=0, padding=1, return_mask=True, ceil_mode=True
                )

        self.assertRaises(ValueError, run_zero_stride)

        def run_zero_tuple_stride():
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [1, 1, 1]), dtype='float32'
                )
                out = F.max_pool1d(x, 1, stride=(0))

        self.assertRaises(ValueError, run_zero_tuple_stride)


if __name__ == '__main__':
    unittest.main()
