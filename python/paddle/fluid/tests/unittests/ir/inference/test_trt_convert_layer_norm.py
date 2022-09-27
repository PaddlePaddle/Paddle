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


class TrtConvertLayerNormTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['epsilon'] < 0 or attrs[0]['epsilon'] > 0.001:
            return False
        if attrs[0]['begin_norm_axis'] <= 0 or attrs[0]['begin_norm_axis'] >= (
                len(inputs['input_data'].shape) - 1):
            return False

        return True

    def sample_program_configs(self):

        def generate_input1(attrs: List[Dict[str, Any]], shape_input):
            return np.ones(shape_input).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], shape_input):
            begin = attrs[0]["begin_norm_axis"]
            sum = 1
            for x in range(begin, len(shape_input)):
                sum *= shape_input[x]
            return np.ones([sum]).astype(np.float32)

        for epsilon in [0.0005, -1, 1]:
            for begin_norm_axis in [1, 0, -1, 2, 3]:
                dics = [{
                    "epsilon": epsilon,
                    "begin_norm_axis": begin_norm_axis
                }, {}]

                ops_config = [{
                    "op_type": "layer_norm",
                    "op_inputs": {
                        "X": ["input_data"],
                        "Scale": ["scale_data"],
                        "Bias": ["bias_data"]
                    },
                    "op_outputs": {
                        "Y": ["y_data"],
                        "Mean": ["saved_mean_data"],
                        "Variance": ["saved_variance_data"]
                    },
                    "op_attrs": dics[0]
                }]
                ops = self.generate_op_config(ops_config)
                shape_input = [1, 3, 64, 64]
                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "bias_data":
                        TensorConfig(data_gen=partial(generate_input2, dics,
                                                      shape_input)),
                        "scale_data":
                        TensorConfig(data_gen=partial(generate_input2, dics,
                                                      shape_input))
                    },
                    inputs={
                        "input_data":
                        TensorConfig(data_gen=partial(generate_input1, dics,
                                                      shape_input))
                    },
                    outputs=["y_data"])

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
            inputs = program_config.inputs
            #if not dynamic_shape:
            #    if attrs[0]["begin_norm_axis"] >= len(inputs["input_data"].shape) - 1:
            #        print ("iiiiiii")
            #        return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-2

    def test(self):
        self.run_test()


class TrtConvertLayerNormTest_2(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['epsilon'] < 0 or attrs[0]['epsilon'] > 0.001:
            return False
        if attrs[0]['begin_norm_axis'] <= 0 or attrs[0]['begin_norm_axis'] >= (
                len(inputs['input_data'].shape) - 1):
            return False

        return True

    def sample_program_configs(self):

        def generate_input1(attrs: List[Dict[str, Any]], shape_input):
            return np.ones(shape_input).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], shape_input):
            begin = attrs[0]["begin_norm_axis"]
            sum = 1
            for x in range(begin, len(shape_input)):
                sum *= shape_input[x]
            return np.ones([sum]).astype(np.float32)

        for epsilon in [0.0005, -1, 1]:
            for begin_norm_axis in [1, 0, -1, 2, 3]:
                dics = [{
                    "epsilon": epsilon,
                    "begin_norm_axis": begin_norm_axis
                }, {}]

                ops_config = [{
                    "op_type": "layer_norm",
                    "op_inputs": {
                        "X": ["input_data"],
                        "Scale": ["scale_data"],
                        "Bias": ["bias_data"]
                    },
                    "op_outputs": {
                        "Y": ["y_data"],
                        "Mean": ["saved_mean_data"],
                        "Variance": ["saved_variance_data"]
                    },
                    "op_attrs": dics[0]
                }]
                ops = self.generate_op_config(ops_config)
                shape_input = [2, 64, 3, 3]
                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "bias_data":
                        TensorConfig(data_gen=partial(generate_input2, dics,
                                                      shape_input)),
                        "scale_data":
                        TensorConfig(data_gen=partial(generate_input2, dics,
                                                      shape_input))
                    },
                    inputs={
                        "input_data":
                        TensorConfig(data_gen=partial(generate_input1, dics,
                                                      shape_input))
                    },
                    outputs=["y_data"])

                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 64, 3, 3]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 3, 3]}
            self.dynamic_shape.opt_input_shape = {"input_data": [2, 64, 3, 3]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            inputs = program_config.inputs
            #if not dynamic_shape:
            #    if attrs[0]["begin_norm_axis"] >= len(inputs["input_data"].shape) - 1:
            #        print ("iiiiiii")
            #        return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
