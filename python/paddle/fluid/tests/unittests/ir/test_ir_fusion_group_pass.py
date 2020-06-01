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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core


class FusionGroupPassTest(PassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 2)
            self.feed_vars.append(
                fluid.data(
                    name="data2", shape=[128, 128], dtype=dtype))

            # subgraph with only 1 op node
            tmp_0 = self.feed_vars[0] * self.feed_vars[1]
            tmp_1 = layers.mul(tmp_0, self.feed_vars[2])
            # subgraph with 2 op nodes
            tmp_2 = layers.relu(tmp_0 + tmp_1)

        self.append_gradients(tmp_2)

        self.num_fused_ops = 2
        self.fetch_list = [tmp_2, self.grad(tmp_1)]

    def setUp(self):
        self.build_program("float32")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"

    def _prepare_feed_vars(self, shape, dtype, num_data):
        feed_vars = []
        for i in range(num_data):
            var = fluid.data(name=("data" + str(i)), shape=shape, dtype=dtype)
            feed_vars.append(var)
        return feed_vars

    def _feed_random_data(self, feed_vars):
        feeds = {}
        for var in feed_vars:
            if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                raise TypeError("Feed data of non LoDTensor is not supported.")

            shape = var.shape
            if var.dtype == fluid.core.VarDesc.VarType.FP32:
                dtype = "float32"
            elif var.dtype == fluid.core.VarDesc.VarType.FP64:
                dtype = "float64"
            elif var.dtype == fluid.core.VarDesc.VarType.FP16:
                dtype = "float16"
            else:
                raise ValueError("Unsupported dtype %s" % var.dtype)
            feeds[var.name] = np.random.random(shape).astype(dtype)
        return feeds

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.pass_attrs = {"fusion_group_pass": {"use_gpu": True}}
            self.check_output_with_place(fluid.CUDAPlace(0))


class FusionGroupPassTest1(FusionGroupPassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 5)

            tmp_0 = layers.assign(self.feed_vars[0])
            # subgraph with 9 op nodes
            tmp_1 = tmp_0 * layers.sigmoid(self.feed_vars[1]) + layers.sigmoid(
                self.feed_vars[2]) * layers.tanh(self.feed_vars[3])
            tmp_2 = layers.tanh(tmp_1) + layers.sigmoid(self.feed_vars[4])

        self.append_gradients(tmp_2)

        self.num_fused_ops = 2
        self.fetch_list = [tmp_2, self.grad(tmp_0)]


class FusionGroupPassTest2(FusionGroupPassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 3)
            self.feed_vars.append(
                fluid.data(
                    name="data3", shape=[128, 32], dtype=dtype))

            # subgraph with 3 op node
            tmp_0 = self.feed_vars[0] + self.feed_vars[1]
            tmp_1 = layers.relu(self.feed_vars[2] * tmp_0)
            # subgraph with 2 op nodes
            tmp_2 = layers.relu(layers.sigmoid(self.feed_vars[3]))
            tmp_3 = layers.mul(tmp_1, tmp_2)

        self.append_gradients(tmp_3)
        self.num_fused_ops = 2
        self.fetch_list = [tmp_3, self.grad(tmp_1)]


class FusionGroupPassTestFP64(FusionGroupPassTest):
    def setUp(self):
        self.build_program("float64")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"


class FusionGroupPassTestCastAndFP16(FusionGroupPassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 2)
            self.feed_vars.append(
                fluid.data(
                    name="data2", shape=[128, 128], dtype=dtype))

            # subgraph with 2 op nodes
            tmp_0 = self.feed_vars[0] * self.feed_vars[1]
            tmp_1 = layers.softmax(layers.cast(tmp_0, dtype="float16"))
            tmp_2 = layers.mul(tmp_0, self.feed_vars[2])
            # subgraph with 4 op nodes
            tmp_3 = layers.cast(tmp_2, dtype="float16")
            tmp_4 = layers.relu(tmp_1 + tmp_3)
            tmp_5 = layers.cast(tmp_4, dtype=dtype)

        self.append_gradients(tmp_5)

        self.num_fused_ops = 4
        self.fetch_list = [tmp_5, self.grad(tmp_0)]


class FusionGroupPassSumTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([32, 128], dtype, 3)
            self.feed_vars.append(
                fluid.data(
                    name="data3", shape=[128, 128], dtype=dtype))

            # subgraph with 2 op nodes
            tmp_0 = layers.sum(
                [self.feed_vars[0], self.feed_vars[1], self.feed_vars[2]])
            tmp_1 = layers.sqrt(tmp_0)
            tmp_2 = layers.mul(tmp_0, self.feed_vars[3])
            # subgraph with 2 op nodes
            tmp_3 = layers.square(layers.sum([tmp_1, tmp_2]))

        self.append_gradients(tmp_3)

        self.num_fused_ops = 4
        self.fetch_list = [tmp_3, self.grad(tmp_0)]


class FusionGroupPassCastTest(FusionGroupPassTest):
    def build_program(self, dtype):
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([2, 2], dtype, 2)

            tmp_0 = layers.elementwise_add(self.feed_vars[0], self.feed_vars[1])
            tmp_1 = layers.cast(tmp_0, dtype="float64")
            tmp_2 = layers.cast(tmp_1, dtype="float32")

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
        with fluid.program_guard(self.main_program, self.startup_program):
            self.feed_vars = self._prepare_feed_vars([2, 2], dtype, 2)

            tmp_0 = layers.elementwise_add(self.feed_vars[0], self.feed_vars[1])
            tmp_1 = layers.fill_constant(shape=[2, 2], dtype=dtype, value=2.0)
            tmp_2 = layers.scale(
                tmp_1, scale=3.0, bias=1.0, bias_after_scale=True)
            tmp_3 = layers.elementwise_mul(tmp_2, tmp_0)

        self.append_gradients(tmp_3)

        self.num_fused_ops = 1
        self.fetch_list = [tmp_2, self.grad(tmp_0)]

    def setUp(self):
        self.build_program("float32")
        self.feeds = self._feed_random_data(self.feed_vars)
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"


if __name__ == "__main__":
    unittest.main()
