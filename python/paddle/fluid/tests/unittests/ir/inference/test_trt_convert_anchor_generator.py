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
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertAnchorGeneratorTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.random.random([batch, 3, 64, 64]).astype(np.float32)

        for batch in [1, 2, 4]:
            for anchor_sizes in [[64.0, 128.0, 256.0, 512.0]]:
                for aspect_ratios in [[0.5, 1, 2], [0.4, 1.2, 3]]:
                    for variances in [[1.0, 1.0, 1.0, 1.0],
                                      [0.5, 1.0, 0.5, 1.0]]:
                        for stride in [[16.0, 16.0], [16.0, 32.0]]:
                            for offset in [0.5, 0.8]:
                                dics = [{
                                    "anchor_sizes": anchor_sizes,
                                    "aspect_ratios": aspect_ratios,
                                    "variances": variances,
                                    "stride": stride,
                                    "offset": offset
                                }]

                                ops_config = [{
                                    "op_type": "anchor_generator",
                                    "op_inputs": {
                                        "Input": ["input_data"]
                                    },
                                    "op_outputs": {
                                        "Anchors": ["output_anchors"],
                                        "Variances": ["output_variances"]
                                    },
                                    "op_attrs": dics[0]
                                }]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(generate_input1,
                                                             batch, dics))
                                    },
                                    outputs=[
                                        "output_anchors", "output_variances"
                                    ])

                                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 3
            else:
                return 0, 4

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
