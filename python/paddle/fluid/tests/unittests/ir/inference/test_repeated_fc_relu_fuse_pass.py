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
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker


class RepeatedFcReluFusePass3Test(InferencePassTest):
    def setUp(self):
        fc_num = 3
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                bias_attr=param_attr,
                act=None)
            fc_outs = []
            fc_outs.append(
                fluid.layers.fc(input=[conv_out], act="relu", size=1000))
            for i in range(1, fc_num):
                fc_outs.append(
                    fluid.layers.fc(
                        input=[fc_outs[i - 1]], act="relu", size=1000))
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [fc_outs[fc_num - 1]]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

        self.assertTrue(
            PassVersionChecker.IsCompatible('repeated_fc_relu_fuse_pass'))


class RepeatedFcReluFusePass9Test(InferencePassTest):
    def setUp(self):
        fc_num = 9
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                bias_attr=param_attr,
                act=None)
            fc_outs = []
            fc_outs.append(
                fluid.layers.fc(input=[conv_out], act="relu", size=1000))
            for i in range(1, fc_num):
                fc_outs.append(
                    fluid.layers.fc(
                        input=[fc_outs[i - 1]], act="relu", size=1000))
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.fetch_list = [fc_outs[fc_num - 1]]

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

        self.assertTrue(
            PassVersionChecker.IsCompatible('repeated_fc_relu_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
