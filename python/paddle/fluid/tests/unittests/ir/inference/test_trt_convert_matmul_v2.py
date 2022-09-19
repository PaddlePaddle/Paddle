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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import os


class TrtConvertMatmulTest_dynamic(TrtLayerAutoScanTest):

    def sample_program_configs(self):

        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for batch in [10, 11, 12, 13, 14, 15]:
            for trans_x in [False]:
                for trans_y in [False]:
                    input1_shape = [batch, 64, 350, 75]
                    input2_shape = [75, 25]
                    dics = [{
                        "trans_x": trans_x,
                        "trans_y": trans_y,
                    }]
                    ops_config = [{
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["input1_data"],
                            "Y": ["input2_data"]
                        },
                        "op_outputs": {
                            "Out": ["output_data"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input1_data":
                            TensorConfig(
                                data_gen=partial(generate_input, input1_shape)),
                            "input2_data":
                            TensorConfig(
                                data_gen=partial(generate_input, input2_shape))
                        },
                        outputs=["output_data"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input1_data": [10, 64, 350, 75],
                "input2_data": [75, 25]
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [100, 64, 350, 75],
                "input2_data": [75, 25]
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [15, 64, 350, 75],
                "input2_data": [75, 25]
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # The output has little diff between gpu and trt in CI-Windows-Inference
        tol_fp32 = 1e-5
        tol_half = 1e-5
        if (os.name == 'nt'):
            tol_fp32 = 1e-3
            tol_half = 1e-3
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), tol_fp32
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), tol_half

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertMatmulTest_dynamic2(TrtLayerAutoScanTest):

    def sample_program_configs(self):

        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for batch in [10, 11, 12, 13, 14, 15]:
            for trans_x in [False]:
                for trans_y in [False]:
                    input1_shape = [60, 40]
                    input2_shape = [batch, 40, 90]
                    dics = [{
                        "trans_x": trans_x,
                        "trans_y": trans_y,
                    }]
                    ops_config = [{
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["input1_data"],
                            "Y": ["input2_data"]
                        },
                        "op_outputs": {
                            "Out": ["output_data"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input1_data":
                            TensorConfig(
                                data_gen=partial(generate_input, input1_shape)),
                            "input2_data":
                            TensorConfig(
                                data_gen=partial(generate_input, input2_shape))
                        },
                        outputs=["output_data"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input1_data": [60, 40],
                "input2_data": [10, 40, 90]
            }
            self.dynamic_shape.max_input_shape = {
                "input1_data": [60, 40],
                "input2_data": [20, 40, 90]
            }
            self.dynamic_shape.opt_input_shape = {
                "input1_data": [60, 40],
                "input2_data": [15, 40, 90]
            }

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # The output has little diff between gpu and trt in CI-Windows-Inference
        tol_fp32 = 1e-5
        tol_half = 1e-5
        if (os.name == 'nt'):
            tol_fp32 = 1e-3
            tol_half = 1e-3
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), tol_fp32
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), tol_half

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
