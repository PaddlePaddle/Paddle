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

import paddle
import paddle.fluid as fluid
import paddle.static.nn as nn
from paddle.fluid.core import AnalysisConfig, PassVersionChecker


class ShuffleChannelFuseTRTPassTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32"
            )
            reshape1 = paddle.reshape(x=data, shape=[-1, 2, 3, 64, 64])
            trans = paddle.transpose(x=reshape1, perm=[0, 2, 1, 3, 4])
            reshape2 = paddle.reshape(x=trans, shape=[-1, 6, 64, 64])
            out = nn.batch_norm(reshape2, is_test=True)

        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = ShuffleChannelFuseTRTPassTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.fetch_list = [out]

    def test_check_output(self):

        self.check_output()

        self.assertTrue(
            PassVersionChecker.IsCompatible('shuffle_channel_detect_pass')
        )


if __name__ == "__main__":
    unittest.main()
