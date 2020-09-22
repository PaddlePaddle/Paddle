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
from paddle.fluid.core import AnalysisConfig


#normal starts && ends
class SlicePluginTRTTest1(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="float32")
            axes = [1, 3]
            starts = [0, 1]
            ends = [2, 3]
            slice_out = fluid.layers.slice(
                data, axes=axes, starts=starts, ends=ends)
            out = fluid.layers.batch_norm(slice_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        # Diff occurred between GPU and TRT. 
        # In order to provide TRT CI ASAP, this test for trt part 
        # is disabled temporarily. 
        self.enable_trt = True
        self.trt_parameters = SlicePluginTRTTest1.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


#negative starts && ends
class SlicePluginTRTTest2(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="float32")
            axes = [2, 3]
            starts = [-3, -2]
            ends = [-1, 3]
            slice_out = fluid.layers.slice(
                data, axes=axes, starts=starts, ends=ends)
            out = fluid.layers.batch_norm(slice_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        # Diff occurred between GPU and TRT. 
        # In order to provide TRT CI ASAP, this test for trt part 
        # is disabled temporarily. 
        self.enable_trt = True
        self.trt_parameters = SlicePluginTRTTest2.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


#exceeded bound starts && ends
class SlicePluginTRTTest3(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="float32")
            axes = [2, 3]
            starts = [-5, -2]
            ends = [-1, 8]
            slice_out = fluid.layers.slice(
                data, axes=axes, starts=starts, ends=ends)
            out = fluid.layers.batch_norm(slice_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        # Diff occurred between GPU and TRT. 
        # In order to provide TRT CI ASAP, this test for trt part 
        # is disabled temporarily. 
        self.enable_trt = True
        self.trt_parameters = SlicePluginTRTTest3.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


#fp16
class SlicePluginTRTTest4(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[3, 3, 3, 3], dtype="float32")
            axes = [2, 3]
            starts = [-5, -2]
            ends = [-1, 8]
            slice_out = fluid.layers.slice(
                data, axes=axes, starts=starts, ends=ends)
            out = fluid.layers.batch_norm(slice_out, is_test=True)

        self.feeds = {
            "data": np.random.random((3, 3, 3, 3)).astype("float32"),
        }
        # Diff occurred between GPU and TRT. 
        # In order to provide TRT CI ASAP, this test for trt part 
        # is disabled temporarily. 
        self.enable_trt = True
        self.trt_parameters = SlicePluginTRTTest3.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


if __name__ == "__main__":
    unittest.main()
