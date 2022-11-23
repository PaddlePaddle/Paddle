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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig


#normal starts && ends
class SlicePluginTRTDynamicTest(InferencePassTest):

    def setUpSliceParams(self):
        self.params_axes = [1, 3]
        self.params_starts = [0, 1]
        self.params_ends = [2, 3]

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTDynamicTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.enable_trt = True
        self.dynamic_shape_params = SlicePluginTRTDynamicTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]}, {'data': [8, 8, 8, 8]},
            {'data': [8, 8, 8, 8]}, False)

    def setUp(self):
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="float32")
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = fluid.layers.slice(data,
                                           axes=axes,
                                           starts=starts,
                                           ends=ends)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        self.fetch_list = [slice_out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 1e-3
            self.check_output_with_option(use_gpu[i], atol)


class SlicePluginTRTDynamicBoundTest(SlicePluginTRTDynamicTest):

    def setUpSliceParams(self):
        self.params_axes = [1, 3]
        self.params_starts = [0, 1]
        self.params_ends = [2, 1000]

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTDynamicBoundTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.enable_trt = True
        self.dynamic_shape_params = SlicePluginTRTDynamicBoundTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]}, {'data': [8, 8, 8, 8]},
            {'data': [8, 8, 8, 8]}, False)


class SlicePluginTRTDynamicNegativeBoundTest(SlicePluginTRTDynamicTest):

    def setUpSliceParams(self):
        self.params_axes = [1, 3]
        self.params_starts = [-5, 1]
        self.params_ends = [2, 1000]

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTDynamicNegativeBoundTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.enable_trt = True
        self.dynamic_shape_params = SlicePluginTRTDynamicNegativeBoundTest.DynamicShapeParam(
            {'data': [1, 1, 1, 1]}, {'data': [8, 8, 8, 8]},
            {'data': [8, 8, 8, 8]}, False)


if __name__ == "__main__":
    unittest.main()
