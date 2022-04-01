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


class TRTGroupNormTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 512, 12, 12], dtype="float32")
            out = self.append_group_norm(data)

        self.feeds = {
            "data": np.random.random([1, 512, 12, 12]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTGroupNormTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TRTGroupNormTest.DynamicShapeParam({
            'data': [1, 512, 12, 12]
        }, {'data': [1, 512, 12, 12]}, {'data': [1, 512, 12, 12]}, False)
        self.fetch_list = [out]

    def append_group_norm(self, data):
        param_attr = fluid.ParamAttr(
            name='group_norm_scale',
            initializer=fluid.initializer.Constant(value=1.0))
        bias_attr = fluid.ParamAttr(
            name='group_norm_bias',
            initializer=fluid.initializer.Constant(value=0.0))
        return fluid.layers.group_norm(
            data,
            groups=32,
            epsilon=0.000009999999747378752,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
