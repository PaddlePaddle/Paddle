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


class TrtConvertGroupNormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(attrs: List[Dict[str, Any]], batch):
            if attrs[0]['data_layout'] == 'NCHW':
                return np.random.random([batch, 32, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch, 64, 64, 32]).astype(np.float32)

        def generate_scale():
            return np.random.randn(32).astype(np.float32)

        def generate_bias():
            return np.random.randn(32).astype(np.float32)

        for batch in [1, 2, 4]:
            for group in [1, 4, 32]:
                for epsilon in [0.1, 0.7]:
                    for data_layout in ['NCHW', 'NHWC']:
                        for i in [0, 1]:
                            dics = [{
                                "epsilon": epsilon,
                                "groups": group,
                                "data_layout": data_layout
                            }, {
                                "groups": group,
                                "data_layout": data_layout
                            }]
                            ops_config = [{
                                "op_type": "group_norm",
                                "op_inputs": {
                                    "X": ["input_data"],
                                    "Scale": ["scale_weight"],
                                    "Bias": ["bias_weight"]
                                },
                                "op_outputs": {
                                    "Y": ["y_output"],
                                    "Mean": ["mean_output"],
                                    "Variance": ["variance_output"]
                                },
                                "op_attrs": dics[i]
                            }]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={
                                    "scale_weight": TensorConfig(
                                        data_gen=partial(generate_scale)),
                                    "bias_weight": TensorConfig(
                                        data_gen=partial(generate_bias))
                                },
                                inputs={
                                    "input_data": TensorConfig(data_gen=partial(
                                        generate_input, dics, batch))
                                },
                                outputs=["y_output"])

                            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 16, 32, 32]}
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 64, 128, 64]
            }
            self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if len(attrs[0]) == 3:
                if dynamic_shape:
                    return 1, 2
                else:
                    return 0, 3
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(self.dynamic_shape.min_input_shape) != 0:
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The goup_norm plugin will check dim not -1 failed when dynamic fp16 mode."
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
