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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertPad3dTensorPadding(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        valid_version = (8, 2, 0)
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        self.assertTrue(compile_version == runtime_version)
        if compile_version < valid_version:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1():
            shape = [6, 6, 6, 64, 64]
            return np.random.uniform(low=0.1, high=1.0, size=shape).astype(
                np.float32
            )

        def generate_paddings(p):
            return np.array(p).astype(np.int32)

        for value in [0, 1.5, 2, 2.5, 3]:
            for paddings in [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 1, 2, 1, 2],
                [1, 1, 1, 1, 1, 1],
                [0, 0, -1, -1, 1, 1],
            ]:
                for pad_mode in ['constant', 'reflect', 'replicate']:
                    dics = [
                        {
                            "value": value,
                            "data_format": "NCDHW",
                            "mode": pad_mode,
                            "paddings": [],
                        },
                        {},
                    ]
                    ops_config = [
                        {
                            "op_type": "pad3d",
                            "op_inputs": {
                                "X": ["input_data"],
                                "Paddings": ["input_paddings"],
                            },
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)
                    inputs = {
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1)
                        )
                    }

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={
                            "input_paddings": TensorConfig(
                                data_gen=partial(generate_paddings, paddings)
                            )
                        },
                        inputs=inputs,
                        outputs=["output_data"],
                        no_cast_list=["input_paddings"],
                    )
                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [6, 6, 6, 64, 64],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [8, 8, 8, 66, 66],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [6, 6, 6, 64, 64],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 2
            return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def test(self):
        self.run_test()


class TrtConvertPad3dListPadding(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        valid_version = (8, 2, 0)
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        self.assertTrue(compile_version == runtime_version)
        if compile_version < valid_version:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1():
            shape = [6, 6, 6, 64, 64]
            return np.random.uniform(low=0.1, high=1.0, size=shape).astype(
                np.float32
            )

        for value in [0, 1.1, 2.3, 3]:
            for paddings in [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 1, 2, 1, 2],
                [1, 1, 1, 1, 1, 1],
                [0, 0, -1, -1, 1, 1],
            ]:
                for pad_mode in ['constant', 'reflect', 'replicate']:
                    dics = [
                        {
                            "value": value,
                            "data_format": "NCDHW",
                            "mode": pad_mode,
                            "paddings": paddings,
                        },
                        {},
                    ]
                    ops_config = [
                        {
                            "op_type": "pad3d",
                            "op_inputs": {"X": ["input_data"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]
                    ops = self.generate_op_config(ops_config)
                    inputs = {
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input1)
                        )
                    }

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs=inputs,
                        outputs=["output_data"],
                    )
                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [6, 6, 6, 64, 64],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [8, 8, 8, 66, 66],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [6, 6, 6, 64, 64],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 2
            return 0, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
