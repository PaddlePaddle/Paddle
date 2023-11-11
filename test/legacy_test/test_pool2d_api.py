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
from test_pool2d_op import (
    avg_pool2D_forward_naive,
    max_pool2D_forward_naive,
    pool2D_forward_naive,
)

import paddle
from paddle import base
from paddle.base import core
from paddle.nn.functional import avg_pool2d, max_pool2d
from paddle.pir_utils import test_with_pir_api


class TestPool2D_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_avg_static_results(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32, 32], dtype="float32"
            )
            result = avg_pool2d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg',
            )

            exe = base.Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_avg_dygraph_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = avg_pool2d(input, kernel_size=2, stride=2, padding=0)

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg',
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool2d_dg = paddle.nn.layer.AvgPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = avg_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_padding_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = avg_pool2d(
                input, kernel_size=2, stride=2, padding=1, ceil_mode=False
            )

            result_np = avg_pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[1, 1],
                ceil_mode=False,
                exclusive=False,
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool2d_dg = paddle.nn.layer.AvgPool2D(
                kernel_size=2, stride=2, padding=1, ceil_mode=False
            )
            result = avg_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_ceilmode_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = avg_pool2d(
                input, kernel_size=2, stride=2, padding=0, ceil_mode=True
            )

            result_np = avg_pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                ceil_mode=True,
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool2d_dg = paddle.nn.layer.AvgPool2D(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
            result = avg_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_static_results(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[2, 3, 32, 32], dtype="float32"
            )
            result = max_pool2d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max',
            )

            exe = base.Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"input": input_np},
                fetch_list=[result],
            )
            np.testing.assert_allclose(fetches[0], result_np, rtol=1e-05)

    def check_max_dygraph_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = max_pool2d(
                input, kernel_size=2, stride=2, padding=0, return_mask=False
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max',
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool2d_dg = paddle.nn.layer.MaxPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = max_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_nhwc_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(
                np.transpose(input_np, [0, 2, 3, 1])
            )
            result = max_pool2d(
                input,
                kernel_size=2,
                stride=2,
                padding=0,
                return_mask=False,
                data_format="NHWC",
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max',
            )
            np.testing.assert_allclose(
                np.transpose(result.numpy(), [0, 3, 1, 2]),
                result_np,
                rtol=1e-05,
            )

    def check_max_dygraph_padding_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = max_pool2d(
                input, kernel_size=2, stride=2, padding=1, ceil_mode=False
            )

            result_np = max_pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[1, 1],
                ceil_mode=False,
                exclusive=False,
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool2d_dg = paddle.nn.layer.MaxPool2D(
                kernel_size=2, stride=2, padding=1, ceil_mode=False
            )
            result = max_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_ceilmode_results(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = max_pool2d(
                input, kernel_size=2, stride=2, padding=0, ceil_mode=True
            )

            result_np = max_pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                ceil_mode=True,
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool2d_dg = paddle.nn.layer.MaxPool2D(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
            result = max_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_stride_is_none(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result, indices = max_pool2d(
                input,
                kernel_size=2,
                stride=None,
                padding="SAME",
                return_mask=True,
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max',
                padding_algorithm="SAME",
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool2d_dg = paddle.nn.layer.MaxPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = max_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_dygraph_stride_is_none(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            result = avg_pool2d(
                input, kernel_size=2, stride=None, padding="SAME"
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg',
                padding_algorithm="SAME",
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool2d_dg = paddle.nn.layer.AvgPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = avg_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_max_dygraph_padding(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
            result = max_pool2d(
                input,
                kernel_size=2,
                stride=2,
                padding=padding,
                return_mask=False,
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max',
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            max_pool2d_dg = paddle.nn.layer.MaxPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = max_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def check_avg_divisor(self, place):
        with base.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = base.dygraph.to_variable(input_np)
            padding = [[0, 0], [0, 0], [0, 0], [0, 0]]
            result = avg_pool2d(
                input,
                kernel_size=2,
                stride=2,
                padding=padding,
                divisor_override=4,
            )

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg',
            )
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

            avg_pool2d_dg = paddle.nn.layer.AvgPool2D(
                kernel_size=2, stride=2, padding=0
            )
            result = avg_pool2d_dg(input)
            np.testing.assert_allclose(result.numpy(), result_np, rtol=1e-05)

    def test_pool2d(self):
        for place in self.places:
            self.check_max_dygraph_results(place)
            self.check_avg_dygraph_results(place)
            self.check_max_dygraph_stride_is_none(place)
            self.check_avg_dygraph_stride_is_none(place)
            self.check_max_dygraph_padding(place)
            self.check_avg_divisor(place)
            self.check_max_dygraph_padding_results(place)
            self.check_max_dygraph_ceilmode_results(place)
            self.check_max_dygraph_nhwc_results(place)

    @test_with_pir_api
    def test_pool2d_static(self):
        paddle.enable_static()
        for place in self.places:
            self.check_max_static_results(place)
            self.check_avg_static_results(place)
        paddle.disable_static()


class TestPool2DError_API(unittest.TestCase):
    def test_error_api(self):
        def run1():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[0, 1], [0, 0], [0, 0], [0, 0]]
                res_pd = max_pool2d(
                    input_pd, kernel_size=2, stride=2, padding=padding
                )

        self.assertRaises(ValueError, run1)

        def run2():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = [[0, 1], [0, 0], [0, 0], [0, 0]]
                res_pd = max_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run2)

        def run3():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "padding"
                res_pd = max_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run3)

        def run3_avg():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "padding"
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run3_avg)

        def run4():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = max_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run4)

        def run4_avg():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run4_avg)

        def run5():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "padding"
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run5)

        def run6():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run6)

        def run7():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=False,
                    data_format='NNNN',
                )

        self.assertRaises(ValueError, run7)

        def run8():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = max_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=False,
                    data_format='NNNN',
                )

        self.assertRaises(ValueError, run8)

        def run9():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = max_pool2d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    ceil_mode=False,
                    data_format='NHWC',
                    return_mask=True,
                )

        self.assertRaises(ValueError, run9)

        def run_kernel_out_of_range():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=[-1, 2],
                    stride=2,
                    padding=0,
                    ceil_mode=False,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run_kernel_out_of_range)

        def run_stride_out_of_range():
            with base.dygraph.guard():
                input_np = np.random.uniform(-1, 1, [2, 3, 32, 32]).astype(
                    np.float32
                )
                input_pd = base.dygraph.to_variable(input_np)
                res_pd = avg_pool2d(
                    input_pd,
                    kernel_size=3,
                    stride=[0, 2],
                    padding=0,
                    ceil_mode=False,
                    data_format='NHWC',
                )

        self.assertRaises(ValueError, run_stride_out_of_range)

        def run_zero_stride():
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [1, 1, 1, 1]), dtype='float32'
                )
                out = max_pool2d(
                    x, 1, stride=0, padding=1, return_mask=True, ceil_mode=True
                )

        self.assertRaises(ValueError, run_zero_stride)

        def run_zero_tuple_stride():
            with base.dygraph.guard():
                array = np.array([1], dtype=np.float32)
                x = paddle.to_tensor(
                    np.reshape(array, [1, 1, 1, 1]), dtype='float32'
                )
                out = max_pool2d(
                    x, 1, stride=(0, 0), return_mask=False, data_format='NHWC'
                )

        self.assertRaises(ValueError, run_zero_tuple_stride)


if __name__ == '__main__':
    unittest.main()
