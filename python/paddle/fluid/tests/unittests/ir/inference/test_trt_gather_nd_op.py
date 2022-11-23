# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTGatherNdTest(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[-1, 3, 4], dtype="float32")
            index = fluid.data(name="index", shape=[-1, 2, 2], dtype="int32")
            gather_nd = fluid.layers.gather_nd(data, index)
            out = fluid.layers.batch_norm(gather_nd, is_test=True)

        self.feeds = {
            "data": np.random.random([2, 3, 4]).astype("float32"),
            "index": np.array([[[0, 1], [1, 0]], [[1, 2],
                                                  [0, 1]]]).astype("int32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTGatherNdTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTGatherNdTest.DynamicShapeParam(
            {
                'data': [1, 3, 4],
                'index': [1, 2, 2]
            }, {
                'data': [3, 3, 4],
                'index': [3, 2, 2]
            }, {
                'data': [3, 3, 4],
                'index': [3, 2, 2]
            }, False)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTGatherNdFp16Test(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 1280, 192],
                              dtype="float32")
            index = fluid.data(name="index", shape=[-1, 1028, 2], dtype="int32")
            gather_nd = fluid.layers.gather_nd(data, index)
            out = fluid.layers.batch_norm(gather_nd, is_test=True)

        index_data = np.zeros((1, 1028, 2), dtype='int32')
        self.feeds = {
            "data": np.random.random([1, 1280, 192]).astype("float32"),
            "index": index_data,
        }
        self.enable_trt = True
        self.trt_parameters = TRTGatherNdFp16Test.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Half, False, False)
        self.fetch_list = [out]
        self.dynamic_shape_params = TRTGatherNdFp16Test.DynamicShapeParam(
            {
                'data': [1, 1280, 192],
                'index': [1, 1028, 2]
            }, {
                'data': [3, 1280, 192],
                'index': [3, 1028, 2]
            }, {
                'data': [3, 1280, 192],
                'index': [3, 1028, 2]
            }, False)

    def test_check_output(self, atol=1e-3):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
