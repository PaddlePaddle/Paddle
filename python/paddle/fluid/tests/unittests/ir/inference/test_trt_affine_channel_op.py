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


class TRTAffineChannelTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            if self.data_layout == 'NCHW':
                shape = [-1, self.channel, self.hw[0], self.hw[1]]
            else:
                shape = [-1, self.hw[0], self.hw[1], self.channel]

            data = fluid.data(name='data', shape=shape, dtype='float32')
            scale = fluid.data(
                name='scale', shape=[self.channel], dtype='float32')
            bias = fluid.data(
                name='bias', shape=[self.channel], dtype='float32')
            affine_channel_out = self.append_affine_channel(data, scale, bias)
            out = fluid.layers.batch_norm(affine_channel_out, is_test=True)

        shape[0] = self.bs
        self.feeds = {
            'data': np.random.random(shape).astype('float32'),
            'scale': np.random.random([self.channel]).astype('float32'),
            'bias': np.random.random([self.channel]).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTAffineChannelTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.bs = 4
        self.channel = 16
        self.hw = (32, 32)
        self.data_layout = 'NCHW'

    def append_affine_channel(self, data, scale, bias):
        return fluid.layers.affine_channel(
            data, scale=scale, bias=bias, data_layout=self.data_layout)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTAffineChannelTest1(TRTAffineChannelTest):
    def set_params(self):
        self.bs = 4
        self.channel = 16
        self.hw = (32, 32)
        self.data_layout = 'NHWC'


if __name__ == "__main__":
    unittest.main()
