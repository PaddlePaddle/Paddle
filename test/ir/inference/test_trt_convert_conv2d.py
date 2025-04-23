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

from __future__ import annotations

import itertools
import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertConv2dNotPersistableTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if (
            inputs['input_data'].shape[1]
            != inputs['weight_data'].shape[1] * attrs[0]['groups']
        ):
            return False

        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 8600:
            return False

        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(attrs: list[dict[str, Any]]):
            return (
                np.random.random(attrs[0]['input_shape']).astype(np.float32)
                - 0.5
            )

        def generate_data(attrs: list[dict[str, Any]]):
            return (
                np.random.random(attrs[0]['weight_shape']).astype(np.float32)
                - 0.5
            )

        input_shapes = [[1, 32, 128, 128]]
        ocs = [64]
        kernel_sizes = [[3, 3]]
        strides_options = [[2, 2]]
        paddings_options = [[1, 1]]
        groups_options = [1]
        padding_algorithm_options = ['EXPLICIT']
        dilations_options = [[1, 1]]
        data_format_options = ['NCHW']

        configurations = [
            input_shapes,
            ocs,
            kernel_sizes,
            strides_options,
            paddings_options,
            groups_options,
            padding_algorithm_options,
            dilations_options,
            data_format_options,
        ]

        for (
            input_shape,
            oc,
            kernel_size,
            strides,
            paddings,
            groups,
            padding_algorithm,
            dilations,
            data_format,
        ) in itertools.product(*configurations):
            ic = input_shape[1]
            attrs = [
                {
                    "dilations": dilations,
                    "padding_algorithm": padding_algorithm,
                    "groups": groups,
                    "paddings": paddings,
                    "strides": strides,
                    "data_format": data_format,
                    # below attrs are used for my convenience.
                    "input_shape": input_shape,
                    "weight_shape": [
                        oc,
                        ic // groups,
                        kernel_size[0],
                        kernel_size[1],
                    ],
                },
            ]

            ops_config = [
                {
                    "op_type": "conv2d",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "Filter": ["weight_data"],
                    },
                    "op_outputs": {"Output": ["conv_output_data"]},
                    "op_attrs": attrs[0],
                },
            ]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input1, attrs)
                    ),
                    "weight_data": TensorConfig(
                        data_gen=partial(generate_data, attrs)
                    ),
                },
                outputs=["conv_output_data"],
            )

        yield program_config

    def generate_dynamic_shape(self, attrs):
        self.dynamic_shape.min_input_shape = {
            "input_data": attrs[0]["input_shape"],
            "weight_data": attrs[0]["weight_shape"],
        }
        self.dynamic_shape.max_input_shape = {
            "input_data": attrs[0]["input_shape"],
            "weight_data": attrs[0]["weight_shape"],
        }
        self.dynamic_shape.opt_input_shape = {
            "input_data": attrs[0]["input_shape"],
            "weight_data": attrs[0]["weight_shape"],
        }
        return self.dynamic_shape

    def sample_predictor_configs(
        self, program_config, run_pir=False
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        self.generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-2, 1e-2)

        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-2, 1e-2)

    def test(self):
        self.run_test(run_pir=True)


if __name__ == "__main__":
    unittest.main()
