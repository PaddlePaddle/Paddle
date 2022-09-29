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


class TRTAffineChannelTest(InferencePassTest):

    def setUp(self):
        self.bs = 2
        self.channel = 8
        self.height = 16
        self.width = 16
        self.data_layout = 'NCHW'
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.enable_trt = True

    def build(self):
        # set min_graph_size to 2,
        # because affine channel doesn't support nhwc format
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, self.bs, 2, self.precision, self.serialize, False)

        with fluid.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [-1, self.channel, self.height, self.width]
            else:
                shape = [-1, self.height, self.width, self.channel]

            data = fluid.data(name='in', shape=shape, dtype='float32')
            # set scale, bias by constant
            scale = fluid.layers.create_parameter(
                shape=[self.channel],
                dtype='float32',
                default_initializer=fluid.initializer.Constant(2.))
            bias = fluid.layers.create_parameter(
                shape=[self.channel],
                dtype='float32',
                default_initializer=fluid.initializer.Constant(.5))
            affine_channel_out = fluid.layers.affine_channel(
                data, scale=scale, bias=bias, data_layout=self.data_layout)
            out = fluid.layers.batch_norm(affine_channel_out, is_test=True)

        shape[0] = self.bs
        self.feeds = {
            'in': np.random.random(shape).astype('float32'),
        }
        self.fetch_list = [out]

    def check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 2e-2
            self.check_output_with_option(use_gpu, atol, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self):
        self.build()
        self.check_output()

    def run_test_all(self):
        precision_opt = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_opt = [False, True]

        if self.data_layout == 'NCHW':
            min_shape = [
                self.bs, self.channel, self.height // 2, self.width // 2
            ]
            max_shape = [self.bs, self.channel, self.height * 2, self.width * 2]
            opt_shape = [self.bs, self.channel, self.height, self.width]

        if self.data_layout == 'NHWC':
            min_shape = [
                self.bs, self.height // 2, self.width // 2, self.channel
            ]
            max_shape = [self.bs, self.height * 2, self.width * 2, self.channel]
            opt_shape = [self.bs, self.height, self.width, self.channel]

        dynamic_shape_profile = InferencePassTest.DynamicShapeParam(
            {'in': min_shape}, {'in': max_shape}, {'in': opt_shape}, False)
        dynamic_shape_opt = [None, dynamic_shape_profile]

        for precision, serialize, dynamic_shape in itertools.product(
                precision_opt, serialize_opt, dynamic_shape_opt):
            self.precision = precision
            self.serialize = serialize
            self.dynamic_shape_params = dynamic_shape
            self.run_test()

    def test_base(self):
        self.run_test()

    def test_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        self.serialize = True
        self.run_test()

    def test_dynamic(self):
        self.dynamic_shape_params = InferencePassTest.DynamicShapeParam(
            {'in': [self.bs, self.channel, self.height // 2, self.width // 2]},
            {'in': [self.bs, self.channel, self.height * 2, self.width * 2]},
            {'in': [self.bs, self.channel, self.height, self.width]}, False)
        self.run_test()

    def test_nchw_all(self):
        self.run_test_all()

    def test_nhwc(self):
        self.data_layout = 'NHWC'
        self.run_test_all()


if __name__ == "__main__":
    unittest.main()
