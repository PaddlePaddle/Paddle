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


class TrtConvertElementwiseTest_one_input(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight():
            return np.random.randn(32).astype(np.float32)

        for batch in [1, 2, 4]:
            for shape in [[32], [batch, 32], [batch, 64, 32],
                          [batch, 8, 16, 32]]:
                for op_type in ["elementwise_add", "elementwise_mul"]:
                    for axis in [len(shape) - 1, -1]:
                        self.dims = len(shape)
                        dics = [{"axis": axis}]
                        ops_config = [{
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["input_data"],
                                "Y": ["weight"]
                            },
                            "op_outputs": {
                                "Out": ["output_data"]
                            },
                            "op_attrs": dics[0]
                        }]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "weight":
                                TensorConfig(data_gen=partial(generate_weight))
                            },
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input, shape)),
                            },
                            outputs=["output_data"])

                        yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {"input_data": [4]}
                self.dynamic_shape.max_input_shape = {"input_data": [256]}
                self.dynamic_shape.opt_input_shape = {"input_data": [16]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 4]}
                self.dynamic_shape.max_input_shape = {"input_data": [4, 256]}
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 16]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input_data": [1, 4, 4]}
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 256, 256]
                }
                self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 16]}
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data": [1, 4, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data": [4, 256, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data": [2, 32, 32, 16]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (0, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 3), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), 1e-5

    def test(self):
        self.run_test()


class TrtConvertElementwiseTest_two_input_without_broadcast(
        TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        if len(inputs['input_data1'].shape) == 1:
            return False

        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        for shape in [[4], [4, 32], [2, 64, 32], [1, 8, 16, 32]]:
            for op_type in ["elementwise_add", "elementwise_mul"]:
                for axis in [0, -1]:
                    self.dims = len(shape)
                    dics = [{"axis": axis}]
                    ops_config = [{
                        "op_type": op_type,
                        "op_inputs": {
                            "X": ["input_data1"],
                            "Y": ["input_data2"]
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
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape))
                        },
                        outputs=["output_data"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1],
                    "input_data2": [1]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [256],
                    "input_data2": [128]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [16],
                    "input_data2": [32]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4],
                    "input_data2": [1, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 256],
                    "input_data2": [128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 16],
                    "input_data2": [32, 64]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4],
                    "input_data2": [1, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [128, 256, 128],
                    "input_data2": [128, 128, 256]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 32, 16],
                    "input_data2": [2, 64, 64]
                }
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 4, 4, 4],
                    "input_data2": [1, 4, 4, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [8, 32, 64, 64],
                    "input_data2": [8, 128, 64, 128]
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 32, 32, 16],
                    "input_data2": [2, 64, 32, 32]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5

    def test(self):
        self.run_test()


class TrtConvertElementwiseTest_two_input_with_broadcast(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        if len(inputs['input_data1'].shape) == 1 or len(inputs['input_data2']
                                                        .shape) == 1:
            return False

        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        input1_shape_list = [[4, 32], [2, 4, 32], [4, 2, 4, 32]]
        input2_shape1_list = [[32], [4, 32], [2, 4, 32]]
        input2_shape2_list = [[1, 32], [1, 1, 32], [1, 1, 1, 32]]
        input2_shape3_list = [[1, 32], [1, 4, 32], [4, 32]]
        input2_shape_list = [
            input2_shape1_list, input2_shape2_list, input2_shape3_list
        ]
        axis1_list = [[-1], [1, -1], [1, -1]]
        axis2_list = [[-1], [-1], [-1]]
        axis3_list = [[-1], [-1], [2, -1]]
        axis_list = [axis1_list, axis2_list, axis3_list]

        for i in range(3):
            input1_shape = input1_shape_list[i]
            for j in range(3):
                input2_shape = input2_shape_list[j][i]
                for op_type in ["elementwise_add", "elementwise_mul"]:
                    for axis in axis_list[j][i]:
                        self.dims1 = len(input1_shape)
                        self.dims2 = len(input2_shape)
                        dics = [{"axis": axis}]
                        ops_config = [{
                            "op_type": op_type,
                            "op_inputs": {
                                "X": ["input_data1"],
                                "Y": ["input_data2"]
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
                                "input_data1": TensorConfig(data_gen=partial(
                                    generate_input, input1_shape)),
                                "input_data2": TensorConfig(data_gen=partial(
                                    generate_input, input2_shape))
                            },
                            outputs=["output_data"])

                        yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            max_shape = [[128], [128, 128], [128, 128, 128],
                         [128, 128, 128, 128]]
            min_shape = [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]
            opt_shape = [[32], [32, 32], [32, 32, 32], [32, 32, 32, 32]]

            self.dynamic_shape.min_input_shape = {
                "input_data1": min_shape[self.dims1 - 1],
                "input_data2": min_shape[self.dims2 - 1]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": max_shape[self.dims1 - 1],
                "input_data2": max_shape[self.dims2 - 1]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": opt_shape[self.dims1 - 1],
                "input_data2": opt_shape[self.dims2 - 1]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
