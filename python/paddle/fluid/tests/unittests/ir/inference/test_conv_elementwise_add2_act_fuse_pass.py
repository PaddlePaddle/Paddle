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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
"""Test for fusion of conv, elementwise_add and 2 act."""


class ConvElementwiseAdd2ActFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32")
            add_y2 = fluid.data(
                name="add_y2", shape=[1, 3, 98, 98], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data, num_filters=3, filter_size=3, bias_attr=None)
            add1_out = fluid.layers.elementwise_add(
                add_y2, conv_out, act="relu")

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32"),
            "add_y2": np.random.random((1, 3, 98, 98)).astype("float32")
        }
        self.fetch_list = [add1_out]
        self.enable_mkldnn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_output_with_option([True])


if __name__ == "__main__":
    unittest.main()
