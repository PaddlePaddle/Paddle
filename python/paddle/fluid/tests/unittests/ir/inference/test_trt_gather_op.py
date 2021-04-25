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


class TRTGatherTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name='data', shape=[-1, 512], dtype='float32')
            index = fluid.data(name='index', shape=[-1], dtype='int32')
            scale_out = self.append_gather(data, index)
            out = fluid.layers.batch_norm(scale_out, is_test=True)

        index = np.arange(self.num_gather, dtype='int32')
        np.random.shuffle(index)

        self.feeds = {
            "data": np.random.random([self.bs, 512]).astype("float32"),
            "index": index,
        }

        self.enable_trt = True
        self.trt_parameters = TRTGatherTest.TensorRTParam(
            1 << 30, self.bs, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.num_gather = 16
        self.bs = 32

    def append_gather(self, data, index):
        return fluid.layers.gather(data, index=index)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTGatherTest1(TRTGatherTest):
    def set_params(self):
        self.num_gather = 32
        self.bs = 32


if __name__ == "__main__":
    unittest.main()
