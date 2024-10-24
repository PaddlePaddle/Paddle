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

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.static import nn


class TensorRTSubgraphPassElementwiseBroadcastTest(InferencePassTest):
    def setUp(self):
        with base.program_guard(self.main_program, self.startup_program):
            data1 = paddle.static.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32"
            )
            data2 = paddle.static.data(
                name="data2", shape=[-1, 3, 64, 1], dtype="float32"
            )
            eltwise_out = self.append_eltwise(data1, data2)
            out = nn.batch_norm(eltwise_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([1, 3, 64, 1]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = (
            TensorRTSubgraphPassElementwiseBroadcastTest.TensorRTParam(
                1 << 30, 32, 0, AnalysisConfig.Precision.Float32, True, False
            )
        )
        self.dynamic_shape_params = (
            TensorRTSubgraphPassElementwiseBroadcastTest.DynamicShapeParam(
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 1]},
                {'data1': [32, 3, 64, 64], 'data2': [32, 3, 64, 1]},
                {'data1': [1, 3, 64, 64], 'data2': [1, 3, 64, 1]},
                False,
            )
        )
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        return paddle.tensor.math.add(x=data1, y=data2)

    def test_check_output(self):
        opt_path = os.path.join(self.path, '_opt_cache')
        if os.path.exists(opt_path):
            shutil.rmtree(opt_path)
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass')
            )


class TensorRTSubgraphPassElementwiseBroadcastTest1(
    TensorRTSubgraphPassElementwiseBroadcastTest
):
    def append_eltwise(self, data1, data2):
        return paddle.tensor.math.subtract(x=data1, y=data2)


class TensorRTSubgraphPassElementwiseBroadcastTest2(
    TensorRTSubgraphPassElementwiseBroadcastTest
):
    def append_eltwise(self, data1, data2):
        return paddle.tensor.math.multiply(x=data1, y=data2)


class TensorRTSubgraphPassElementwiseBroadcastTest3(
    TensorRTSubgraphPassElementwiseBroadcastTest
):
    def append_eltwise(self, data1, data2):
        return paddle.tensor.math.divide(x=data1, y=data2)


if __name__ == "__main__":
    unittest.main()
