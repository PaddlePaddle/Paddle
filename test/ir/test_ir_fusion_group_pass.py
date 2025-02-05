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

import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle import base
from paddle.base import core


class FusionGroupPassTest(PassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 2)
            self.feed_vars.append(
                paddle.static.data(name="data2", shape=[128, 128], dtype=dtype)
            )

            # subgraph with only 1 op node
            tmp_0 = self.feed_vars[0] * self.feed_vars[1]
            tmp_0.stop_gradient = False
            tmp_1 = paddle.matmul(tmp_0, self.feed_vars[2])
            # subgraph with 2 op nodes
            tmp_2 = paddle.nn.functional.relu(tmp_0 + tmp_1)

        self.append_gradients(tmp_2)

        self.num_fused_ops = 2
        self.fetch_list = [tmp_2, self.grad(tmp_1)]

    def setUp(self):
        self.build_program("float32")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"

    def _prepare_feed_vars(self, shape, dtype, num_data, stop_gradient=True):
        feed_vars = []
        for i in range(num_data):
            var = paddle.static.data(
                name=("data" + str(i)), shape=shape, dtype=dtype
            )
            var.stop_gradient = stop_gradient
            feed_vars.append(var)
        return feed_vars

    def _feed_random_data(self, feed_vars):
        feeds = {}
        for var in feed_vars:
            if var.type != base.core.VarDesc.VarType.DENSE_TENSOR:
                raise TypeError(
                    "Feed data of non DenseTensor is not supported."
                )

            shape = var.shape
            if var.dtype == paddle.float32:
                dtype = "float32"
            elif var.dtype == paddle.float64:
                dtype = "float64"
            elif var.dtype == paddle.float16:
                dtype = "float16"
            else:
                raise ValueError(f"Unsupported dtype {var.dtype}")
            feeds[var.name] = np.random.random(shape).astype(dtype)
        return feeds

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.pass_attrs = {"fusion_group_pass": {"use_gpu": True}}
            self.check_output_with_place(base.CUDAPlace(0))


class FusionGroupPassComplicatedTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 64], dtype, 5, False)

            one = paddle.tensor.fill_constant(shape=[1], dtype=dtype, value=1.0)
            tmp_0 = one * self.feed_vars[0]
            # subgraph with 9 op nodes
            tmp_1 = tmp_0 * paddle.nn.functional.sigmoid(
                self.feed_vars[1]
            ) + paddle.nn.functional.sigmoid(self.feed_vars[2]) * paddle.tanh(
                self.feed_vars[3]
            )
            tmp_2 = paddle.tanh(tmp_1) + paddle.nn.functional.sigmoid(
                self.feed_vars[4]
            )

        self.append_gradients(tmp_2)

        self.num_fused_ops = 2
        self.fetch_list = [tmp_2, self.grad(tmp_0)]


class FusionGroupPassInplaceTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 3)
            self.feed_vars.append(
                paddle.static.data(name="data3", shape=[128, 32], dtype=dtype)
            )

            # subgraph with 3 op node
            tmp_0 = self.feed_vars[0] - self.feed_vars[1]
            tmp_1 = tmp_0 * self.feed_vars[2]
            tmp_2 = paddle.assign(tmp_1, output=tmp_0)
            tmp_3 = paddle.matmul(tmp_2, self.feed_vars[3])

        self.num_fused_ops = 1
        self.fetch_list = [tmp_3]


class FusionGroupPassTestFP64(FusionGroupPassTest):
    def setUp(self):
        self.build_program("float64")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"


class FusionGroupPassTestCastAndFP16(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 2)
            self.feed_vars.append(
                paddle.static.data(name="data2", shape=[128, 128], dtype=dtype)
            )

            # subgraph with 2 op nodes
            tmp_0 = self.feed_vars[0] * self.feed_vars[1]
            tmp_0.stop_gradient = False
            tmp_1 = paddle.cast(tmp_0, dtype="float16")
            zero = paddle.tensor.fill_constant(
                shape=[128], dtype="float16", value=0
            )
            # TODO(xreki): fix precision problem when using softmax of float16.
            # tmp_2 = layers.softmax(tmp_1)
            tmp_2 = paddle.add(tmp_1, zero)
            tmp_3 = paddle.matmul(tmp_0, self.feed_vars[2])
            # subgraph with 4 op nodes
            tmp_3 = paddle.cast(tmp_2, dtype="float16")
            tmp_4 = paddle.nn.functional.relu(tmp_1 + tmp_3)
            tmp_5 = paddle.cast(tmp_4, dtype=dtype)
            tmp_3 = paddle.cast(tmp_2, dtype=dtype)

        self.append_gradients(tmp_5)

        self.num_fused_ops = 4
        self.fetch_list = [tmp_5, self.grad(tmp_0)]


class FusionGroupPassSumTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 3)
            self.feed_vars.append(
                paddle.static.data(name="data3", shape=[128, 128], dtype=dtype)
            )

            # subgraph with 2 op nodes
            tmp_0 = paddle.add_n(
                [self.feed_vars[0], self.feed_vars[1], self.feed_vars[2]]
            )
            tmp_0.stop_gradient = False
            tmp_1 = paddle.sqrt(tmp_0)
            tmp_2 = paddle.matmul(tmp_0, self.feed_vars[3])
            # subgraph with 2 op nodes
            tmp_3 = paddle.square(paddle.add_n([tmp_1, tmp_2]))

        self.append_gradients(tmp_3)

        self.num_fused_ops = 3
        self.fetch_list = [tmp_3, self.grad(tmp_0)]


class FusionGroupPassCastTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([2, 2], dtype, 2)

            tmp_0 = paddle.add(self.feed_vars[0], self.feed_vars[1])
            tmp_0.stop_gradient = False
            tmp_1 = paddle.cast(tmp_0, dtype="float64")
            tmp_2 = paddle.cast(tmp_1, dtype="float32")

        self.append_gradients(tmp_2)

        self.num_fused_ops = 2
        self.fetch_list = [tmp_2, self.grad(tmp_0)]

    def setUp(self):
        self.build_program("float64")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"


class FusionGroupPassFillConstantTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with base.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([2, 2], dtype, 2)

            tmp_0 = paddle.add(self.feed_vars[0], self.feed_vars[1])
            tmp_0.stop_gradient = False
            tmp_1 = paddle.tensor.fill_constant(
                shape=[2, 2], dtype=dtype, value=2.0
            )
            tmp_2 = paddle.scale(
                tmp_1, scale=3.0, bias=1.0, bias_after_scale=True
            )
            tmp_3 = paddle.multiply(tmp_2, tmp_0)

        self.append_gradients(tmp_3)

        self.num_fused_ops = 1
        self.fetch_list = [tmp_2, self.grad(tmp_0)]


if __name__ == "__main__":
    unittest.main()
