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
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            feed_vars = self._prepare_feed_vars([32, 128], "float32", 2)
            feed_vars.append(
                fluid.data(
                    name="data2", shape=[128, 128], dtype="float32"))

            tmp_0 = feed_vars[0] * feed_vars[1]  # subgraph with only 1 op node
            tmp_1 = layers.mul(tmp_0, feed_vars[2])
            tmp_2 = layers.relu(tmp_0 + tmp_1)  # subgraph with 2 op nodes

        self.feeds = self._feed_random_data(feed_vars)
        self.fetch_list = [tmp_2]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 1

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
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            feed_vars = self._prepare_feed_vars([32, 128], "float32", 5)

            tmp_0 = layers.assign(feed_vars[0])
            # subgraph with 9 op nodes
            tmp_1 = tmp_0 * layers.sigmoid(feed_vars[1]) + layers.sigmoid(
                feed_vars[2]) * layers.tanh(feed_vars[3])
            tmp_2 = layers.tanh(tmp_1) + layers.sigmoid(feed_vars[4])

        self.feeds = self._feed_random_data(feed_vars)
        self.fetch_list = [tmp_1, tmp_2]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 1


class FusionGroupPassTest2(FusionGroupPassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            feed_vars = self._prepare_feed_vars([32, 128], "float32", 3)
            feed_vars.append(
                fluid.data(
                    name="data3", shape=[128, 32], dtype="float32"))

            # subgraph with 3 op nodes
            tmp_1 = layers.relu((feed_vars[0] - feed_vars[1]) * feed_vars[2])
            # subgraph with 2 op nodes
            tmp_2 = layers.relu(layers.sigmoid(feed_vars[3]))
            tmp_3 = layers.mul(tmp_1, tmp_2)

        self.feeds = self._feed_random_data(feed_vars)
        self.fetch_list = [tmp_1, tmp_2, tmp_3]
        self.pass_names = "fusion_group_pass"
        self.fused_op_type = "fusion_group"
        self.num_fused_ops = 2


if __name__ == "__main__":
    unittest.main()
