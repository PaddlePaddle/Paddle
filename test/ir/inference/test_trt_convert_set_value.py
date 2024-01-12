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

import unittest
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
            return np.random.random([2, 3, 3]).astype(np.float32)

        def generate_input2():
            return np.random.random([2, 2, 3]).astype(np.float32)

        for update_scalar in [True, False]:
            self.update_scalar = update_scalar
            set_value_inputs = {}
            if update_scalar:
                set_value_inputs = {
                    "Input": ["input_data"],
                }
            else:
                set_value_inputs = {
                    "Input": ["input_data"],
                    "ValueTensor": ["update_data"],
                }
            ops_config = [
                {
                    "op_type": "set_value",
                    "op_inputs": set_value_inputs,
                    "op_outputs": {"Out": ["input_data"]},
                    "op_attrs": {
                        "axes": [1],
                        "starts": [0],
                        "ends": [2],
                        "steps": [1],
                        "decrease_axes": [],
                        "values": [0.0],
                    },
                },
                {
                    "op_type": "relu",
                    "op_inputs": {
                        "X": ["input_data"],
                    },
                    "op_outputs": {"Out": ["output_data"]},
                    "op_attrs": {},
                },
            ]

            ops = self.generate_op_config(ops_config)
            if update_scalar:
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1)
                        ),
                    },
                    outputs=["output_data"],
                )
            else:
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1)
                        ),
                        "update_data": TensorConfig(
                            data_gen=partial(generate_input2)
                        ),
                    },
                    outputs=["output_data"],
                )

            yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            if self.update_scalar:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 3, 3],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [3, 3, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [3, 3, 3],
                }
            else:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [2, 3, 3],
                    "update_data": [2, 2, 3],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [3, 3, 4],
                    "update_data": [3, 2, 4],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [3, 3, 3],
                    "update_data": [3, 2, 3],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if self.update_scalar:
                    if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
                        return 1, 3
                    return 1, 2
                else:
                    if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
                        return 1, 4
                    return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-4)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
