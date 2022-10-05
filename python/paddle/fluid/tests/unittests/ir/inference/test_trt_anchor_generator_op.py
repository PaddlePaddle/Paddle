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
import itertools
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTAnchorGeneratorBaseTest(InferencePassTest):

    def setUp(self):
        self.bs = 1
        self.channel = 16
        self.height = 32
        self.width = 32
        self.anchor_sizes = [64., 128., 256., 512.]
        self.aspect_ratios = [.5, 1., 2.]
        self.variance = [.1, .1, .2, .2]
        self.stride = [8., 8.]
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.enable_trt = True
        self.feeds = {
            'data':
            np.random.random([self.bs, self.channel, self.height,
                              self.width]).astype('float32'),
        }

    def build(self):
        min_graph_size = 3 if self.dynamic_shape_params is not None else 2
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, self.bs, min_graph_size, self.precision, self.serialize,
            False)
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data',
                              shape=[-1, self.channel, self.height, self.width],
                              dtype='float32')
            anchor, var = fluid.layers.detection.anchor_generator(
                data,
                anchor_sizes=self.anchor_sizes,
                aspect_ratios=self.aspect_ratios,
                variance=self.variance,
                stride=self.stride)
            if self.dynamic_shape_params is not None:
                anchor = fluid.layers.transpose(anchor, [2, 3, 0, 1])
            out = fluid.layers.batch_norm(anchor, is_test=True)

        self.fetch_list = [out, var]

    def run_test(self):
        self.build()
        self.check_output()

    def set_dynamic(self):
        self.dynamic_shape_params = InferencePassTest.DynamicShapeParam(
            {
                'data':
                [self.bs, self.channel, self.height // 2, self.width // 2]
            }, {'data': [self.bs, self.channel, self.height, self.width]},
            {'data': [self.bs, self.channel, self.height, self.width]}, False)

    def test_base(self):
        self.run_test()

    def test_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        self.serialize = True
        self.run_test()

    def test_dynamic(self):
        self.set_dynamic()
        self.run_test()

    def test_dynamic_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.set_dynamic()
        self.run_test()

    def test_dynamic_serialize(self):
        self.serialize = True
        self.set_dynamic()
        self.run_test()

    def test_dynamic_fp16_serialize(self):
        self.serialize = True
        self.precision = AnalysisConfig.Precision.Half
        self.set_dynamic()
        self.run_test()

    def check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 1e-3
            self.check_output_with_option(use_gpu, atol, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
