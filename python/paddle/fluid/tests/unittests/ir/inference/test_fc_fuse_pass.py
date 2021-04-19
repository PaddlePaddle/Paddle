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
from paddle.fluid.core import PassVersionChecker


class FcFusePassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            data_y = fluid.data(name="y", shape=[-1, 128, 768], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=3072,
                                      num_flatten_dims=2,
                                      act="relu")
            fc_out2 = fluid.layers.fc(input=fc_out1,
                                      size=768,
                                      num_flatten_dims=2)

        self.feeds = {"data": np.random.random((4, 128, 768)).astype("float32")}
        self.fetch_list = [fc_out2]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])

        self.assertTrue(PassVersionChecker.IsCompatible('fc_fuse_pass'))


if __name__ == "__main__":
    unittest.main()
