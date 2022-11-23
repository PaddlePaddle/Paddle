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
class SlicePluginTRTTest(InferencePassTest):

    def setUpSliceParams(self):
        self.params_axes = [1, 3]
        self.params_starts = [0, 1]
        self.params_ends = [2, 3]

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.enable_trt = True

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
            out = fluid.layers.batch_norm(slice_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            atol = 1e-5
            if self.trt_parameters.precision == AnalysisConfig.Precision.Half:
                atol = 1e-3
            self.check_output_with_option(use_gpu[i], atol)


#negative starts && ends
class SlicePluginTRTTestNegativeStartsAndEnds(SlicePluginTRTTest):

    def setUpSliceParams(self):
        self.params_axes = [2, 3]
        self.params_starts = [-3, -2]
        self.params_ends = [-1, 3]


#exceeded bound starts && ends
class SlicePluginTRTTestStartsAndEndsBoundCheck(SlicePluginTRTTest):

    def setUpSliceParams(self):
        self.params_axes = [2, 3]
        self.params_starts = [-5, -2]
        self.params_ends = [-1, 8]


#fp16
class SlicePluginTRTTestFp16(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.enable_trt = True


class StaticSlicePluginTRTTestFp16(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, True, False)
        self.enable_trt = True


class StaticSlicePluginTRTTestFp32(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, True, False)
        self.enable_trt = True


class SlicePluginTRTTestInt32(SlicePluginTRTTest):

    def setUp(self):
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="int32")
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = fluid.layers.slice(data,
                                           axes=axes,
                                           starts=starts,
                                           ends=ends)
            cast_out = fluid.layers.cast(slice_out, 'float32')
            out = fluid.layers.batch_norm(cast_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("int32"),
        }
        self.fetch_list = [out]


class StaticSlicePluginTRTTestInt32(SlicePluginTRTTest):

    def setUpTensorRTParams(self):
        self.trt_parameters = SlicePluginTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, True, False)
        self.enable_trt = True

    def setUp(self):
        self.setUpSliceParams()
        self.setUpTensorRTParams()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="int32")
            axes = self.params_axes
            starts = self.params_starts
            ends = self.params_ends
            slice_out = fluid.layers.slice(data,
                                           axes=axes,
                                           starts=starts,
                                           ends=ends)
            cast_out = fluid.layers.cast(slice_out, 'float32')
            out = fluid.layers.batch_norm(cast_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("int32"),
        }
        self.fetch_list = [out]


if __name__ == "__main__":
    unittest.main()
