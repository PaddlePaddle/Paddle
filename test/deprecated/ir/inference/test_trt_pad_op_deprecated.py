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

import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig
from paddle.static import nn


class PadOpTRTTest(InferencePassTest):
    def setUp(self):
        paddle.enable_static()
        with base.program_guard(self.main_program, self.startup_program):
            data = paddle.static.data(
                name="data", shape=[1, 3, 128, 128], dtype="float32"
            )
            pad_out = paddle.nn.functional.pad(
                x=data, pad=[0, 0, 0, 0, 0, 1, 1, 2], value=0.0
            )
            out = nn.batch_norm(pad_out, is_test=True)

        self.feeds = {
            "data": np.random.random((1, 3, 128, 128)).astype("float32")
        }
        self.enable_trt = True
        self.trt_parameters = PadOpTRTTest.TensorRTParam(
            1 << 30, 32, 1, AnalysisConfig.Precision.Float32, False, False
        )
        self.fetch_list = [out]

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)

        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


if __name__ == "__main__":
    unittest.main()
