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

import os
import shutil
import unittest
import itertools
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTInstanceNormTest(InferencePassTest):

    def setUp(self):
        self.bs = 4
        self.channel = 4
        self.height = 8
        self.width = 8
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.enable_trt = True

    def build(self):
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, self.bs, 2, self.precision, self.serialize, False)

        with fluid.program_guard(self.main_program, self.startup_program):
            shape = [-1, self.channel, self.height, self.width]
            data = fluid.data(name='in', shape=shape, dtype='float32')
            instance_norm_out = fluid.layers.instance_norm(data)
            out = fluid.layers.batch_norm(instance_norm_out, is_test=True)

        shape[0] = self.bs
        self.feeds = {
            'in': np.random.random(shape).astype('float32'),
        }
        self.fetch_list = [out]

    def check_output(self, remove_cache=False):
        if remove_cache and os.path.exists(self.path + "_opt_cache"):
            shutil.rmtree(self.path + "_opt_cache")
        if core.is_compiled_with_cuda():
            use_gpu = True
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 2e-2
            self.check_output_with_option(use_gpu, atol, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def run_test(self, remove_cache=False):
        self.build()
        self.check_output(remove_cache)

    def run_all_tests(self):
        precision_opt = [
            AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half
        ]
        serialize_opt = [False, True]

        for precision, serialize in itertools.product(precision_opt,
                                                      serialize_opt):
            self.precision = precision
            self.serialize = serialize
            self.run_test()

    def test_base(self):
        self.run_test()

    def test_fp16(self):
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        self.serialize = True
        self.run_test(remove_cache=True)

    def test_all(self):
        self.run_all_tests()


if __name__ == "__main__":
    unittest.main()
