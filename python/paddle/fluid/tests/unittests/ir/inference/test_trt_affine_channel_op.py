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
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[-1, 3, 64, 64], dtype='float32')
            scale = fluid.data(name='scale', shape=[3], dtype='float32')
            bias = fluid.data(name='bias', shape=[3], dtype='float32')
            affine_channel_out = self.append_affine_channel(data, scale, bias)
            out = fluid.layers.batch_norm(affine_channel_out, is_test=True)

        self.feeds = {
            'data': np.random.random([1, 3, 64, 64]).astype('float32'),
            'scale': np.random.random([3]).astype('float32'),
            'bias': np.random.random([3]).astype('float32'),
        }
        self.enable_trt = True
        self.trt_parameters = TRTAffineChannelTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_affine_channel(self, data, scale, bias):
        return fluid.layers.affine_channel(data, scale=scale, bias=bias)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
