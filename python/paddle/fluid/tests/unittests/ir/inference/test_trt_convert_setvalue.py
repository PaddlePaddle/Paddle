# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertSetValue(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.random.random([1, 6, 3]).astype(np.float32)

        def generate_input2():
            return np.random.random([1, 6, 1]).astype(np.float32)

        ops_config = [
            {
                "op_type": "set_value",
                "op_inputs": {
                    "Input": ["input_data"],
                    "ValueTensor": ["update_data"],
                },
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {
                    "axes": [2],
                    "starts": [0],
                    "ends": [1],
                    "steps": [1],
                },
            }
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "update_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 6, 3],
                "update_data": [1, 6, 1],
                "output_data": [1, 6, 3],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1, 6, 3],
                "update_data": [1, 6, 1],
                "output_data": [1, 6, 3],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 6, 3],
                "update_data": [1, 6, 1],
                "output_data": [1, 6, 3],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-3, 1e-3)

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        # tensorrt op num is not consistent with paddle
        return True

    def test(self):
        self.run_test()
