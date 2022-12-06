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

import subprocess
import sys
import unittest

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig


class TensorRTInspectorTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[1, 16, 16], dtype="float32")
            matmul_out = paddle.matmul(
                x=data,
                y=data,
                transpose_x=self.transpose_x,
                transpose_y=self.transpose_y,
            )
            matmul_out = paddle.scale(matmul_out, scale=self.alpha)
            out = paddle.static.nn.batch_norm(matmul_out, is_test=True)

        self.feeds = {
            "data": np.ones([1, 16, 16]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = InferencePassTest.TensorRTParam(
            1 << 30, 1, 0, AnalysisConfig.Precision.Float32, False, False, True
        )
        self.fetch_list = [out]

    def set_params(self):
        self.transpose_x = True
        self.transpose_y = True
        self.alpha = 2.0

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            build_engine = subprocess.run(
                [sys.executable, 'test_trt_inspector.py', '--build-engine'],
                stderr=subprocess.PIPE,
            )
            engine_info = build_engine.stderr.decode('ascii')
            trt_compile_version = paddle.inference.get_trt_compile_version()
            trt_runtime_version = paddle.inference.get_trt_runtime_version()
            valid_version = (8, 2, 0)
            if (
                trt_compile_version >= valid_version
                and trt_runtime_version >= valid_version
            ):
                self.assertTrue('====== engine info ======' in engine_info)
                self.assertTrue('====== engine info end ======' in engine_info)
                self.assertTrue('matmul' in engine_info)
                self.assertTrue('LayerType: Scale' in engine_info)
                self.assertTrue('batch_norm' in engine_info)
            else:
                self.assertTrue(
                    'Inspector needs TensorRT version 8.2 and after.'
                    in engine_info
                )


if __name__ == "__main__":
    if '--build-engine' in sys.argv:
        test = TensorRTInspectorTest()
        test.setUp()
        use_gpu = True
        test.check_output_with_option(use_gpu)
    else:
        unittest.main()
