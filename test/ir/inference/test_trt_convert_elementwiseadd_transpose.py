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


class TrtConvertElementwiseAddTransposeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def conv_filter_datagen(dics):
            c = dics["c"]
            x = (np.random.randn(c, c, 1, 1)) * np.sqrt(2 / c) * 0.1
            return x.astype(np.float32)

        def conv_elementwise_bias_datagen(dics):
            c = dics["c"]
            x = np.random.random([dics["c"]]) * 0.01
            return x.astype(np.float32)

        def ele1_input_datagen(dics):
            x = np.random.random(
                [dics["batch"], dics["h"] * dics["w"], dics["c"]]
            )
            x = (x - np.mean(x)) / (np.std(x))
            return x.astype(np.float32)

        def ele2_input_datagen(dics):
            x = np.random.random(
                [dics["batch"], dics["h"] * dics["w"], dics["c"]]
            )
            x = (x - np.mean(x)) / (np.std(x))
            return x.astype(np.float32)

        for batch in [2]:
            for h in [32, 64]:
                for w in [32, 64]:
                    for c in [128, 320, 255, 133]:
                        dics = {"batch": batch, "h": h, "w": w, "c": c}
                        ops_config = [
                            {
                                "op_type": "elementwise_add",
                                "op_inputs": {
                                    "X": ["ele_input_1"],
                                    "Y": ["ele_input_2"],
                                },
                                "op_outputs": {"Out": ["elementwise_out"]},
                                "op_attrs": {"axis": -1},
                            },
                            {
                                "op_type": "reshape",
                                "op_inputs": {"X": ["elementwise_out"]},
                                "op_outputs": {
                                    "Out": ["reshape_out"],
                                },
                                "op_attrs": {"shape": [-1, h, w, c]},
                            },
                            {
                                "op_type": "transpose2",
                                "op_inputs": {
                                    "X": ["reshape_out"],
                                },
                                "op_outputs": {
                                    "Out": ["transpose2_out"],
                                },
                                "op_attrs": {"axis": [0, 3, 1, 2]},
                            },
                            {
                                "op_type": "conv2d",
                                "op_inputs": {
                                    "Input": ["transpose2_out"],
                                    "Filter": ["conv2d_filter"],
                                },
                                "op_outputs": {
                                    "Output": ["conv2d_output"],
                                },
                                "op_attrs": {
                                    "dilations": [1, 1],
                                    "padding_algorithm": "EXPLICIT",
                                    "groups": 1,
                                    "paddings": [0, 0],
                                    "strides": [1, 1],
                                    "data_format": "NCHW",
                                },
                            },
                        ]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "conv2d_filter": TensorConfig(
                                    data_gen=partial(conv_filter_datagen, dics)
                                ),
                                "elementwise_bias": TensorConfig(
                                    data_gen=partial(
                                        conv_elementwise_bias_datagen, dics
                                    )
                                ),
                            },
                            inputs={
                                "ele_input_1": TensorConfig(
                                    data_gen=partial(ele1_input_datagen, dics)
                                ),
                                "ele_input_2": TensorConfig(
                                    data_gen=partial(ele2_input_datagen, dics)
                                ),
                            },
                            outputs=["conv2d_output"],
                        )
                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs, inputs):
            channel = inputs['ele_input_1'].shape[2]

            self.dynamic_shape.min_input_shape = {
                "ele_input_1": [1, 32 * 32, channel],
                "ele_input_2": [1, 32 * 32, channel],
            }
            self.dynamic_shape.max_input_shape = {
                "ele_input_1": [4, 64 * 64, channel],
                "ele_input_2": [4, 64 * 64, channel],
            }
            self.dynamic_shape.opt_input_shape = {
                "ele_input_1": [4, 64 * 64, channel],
                "ele_input_2": [4, 64 * 64, channel],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        inputs = program_config.inputs
        # just support dynamic_shape
        generate_dynamic_shape(attrs, inputs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (
            1e-2,
            1e-2,
        )  # tol 1e-2 for half

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
