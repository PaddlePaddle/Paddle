# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertQuantizeDequantizeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        # only TRT > 8.0 has quantize / dequantize layers
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 8517:
            return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_add(shape):
            return np.ones(shape).astype(np.float32)

        def generate_scale():
            return np.ones([1]).astype(np.float32) + 2.521234002

        def generate_zeropoint():
            return np.zeros([1]).astype(np.float32)

        desc = [{"quant_axis": -1}]
        ops_config = [
            {
                "op_type": "quantize_linear",
                "op_inputs": {
                    "X": ["input_data_1"],
                    "Scale": ["scale_data_1"],
                    "ZeroPoint": ["zeropoint_data_1"],
                },
                "op_outputs": {
                    "Y": ["y_data_1"],
                },
                "op_attrs": desc[0],
            },
            {
                "op_type": "dequantize_linear",
                "op_inputs": {
                    "X": ["y_data_1"],
                    "Scale": ["scale_data_2"],
                    "ZeroPoint": ["zeropoint_data_2"],
                },
                "op_outputs": {
                    "Y": ["y_data_2"],
                },
                "op_attrs": desc[0],
            },
            {
                "op_type": "elementwise_add",
                "op_inputs": {
                    "X": ["y_data_2"],
                    "Y": ["add"],
                },
                "op_outputs": {
                    "Out": ["y_data_3"],
                },
                "op_attrs": {"axis": -1},
                "outputs_dtype": {"output_data": np.float32},
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "scale_data_1": TensorConfig(data_gen=partial(generate_scale)),
                "zeropoint_data_1": TensorConfig(
                    data_gen=partial(generate_zeropoint)
                ),
                "scale_data_2": TensorConfig(data_gen=partial(generate_scale)),
                "zeropoint_data_2": TensorConfig(
                    data_gen=partial(generate_zeropoint)
                ),
                "add": TensorConfig(
                    data_gen=partial(generate_add, [1, 8, 32, 32])
                ),
            },
            inputs={
                "input_data_1": TensorConfig(
                    data_gen=partial(generate_input1, [1, 8, 32, 32])
                )
            },
            outputs=["y_data_3"],
        )

        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data_1": [1, 8, 32, 32],
                "add": [1, 8, 32, 32],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data_1": [16, 8, 32, 32],
                "add": [16, 8, 32, 32],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data_1": [16, 8, 32, 32],
                "add": [16, 8, 32, 32],
            }

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)

        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-2, 1e-2)

    def test(self):
        self.run_test(quant=False, explicit=True)


if __name__ == "__main__":
    unittest.main()
