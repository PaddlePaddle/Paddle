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
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTAnchorGeneratorTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=[-1, self.channel, self.height, self.width],
                dtype='float32')
            anchor, var = fluid.layers.detection.anchor_generator(
                data,
                anchor_sizes=self.anchor_sizes,
                aspect_ratios=self.aspect_ratios,
                variance=self.variance,
                stride=self.stride)
            out = fluid.layers.batch_norm(var, is_test=True)

        self.feeds = {
            "data":
            np.random.random([self.bs, self.channel, self.height,
                              self.width]).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTAnchorGeneratorTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out, var]

    def set_params(self):
        self.bs = 1
        self.channel = 32
        self.height = 64
        self.width = 64
        self.anchor_sizes = [64., 128., 256., 512.]
        self.aspect_ratios = [.5, 1., 2.]
        self.variance = [.1, .1, .2, .2]
        self.stride = [8., 8.]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
