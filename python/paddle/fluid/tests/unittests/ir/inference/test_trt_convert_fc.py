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
import unittest
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertFcTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:
            if attrs[0]['padding_algorithm'] == 'SAME' and (
                    attrs[0]['strides'][0] > 1 or attrs[0]['strides'][1] > 1):
                return False
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.ones([batch, 3, 64, (int)(attrs[0]["m"]/2), 2]).astype(np.float32)

        def generate_w(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["m"], attrs[0]["n"]]).astype(np.float32)

        def generate_bias(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["n"]]).astype(np.float32)


        for batch in [1, 4]:
            for [m,n] in [[32, 23]]:
                dics = [{
                    "in_num_col_dims" : 3,
                    # for my conveinence
                    "m" : m,
                    "n" : n,
                }, {}]

                ops_config = [{
                    "op_type": "fc",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "W": ["w_data"],
                        "Bias": ["bias_data"]
                    },
                    "op_outputs": {
                        "Out": ["output_data"]
                    },
                    "op_attrs": dics[0]
                },]

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "w_data":TensorConfig(data_gen=partial(generate_w, batch, dics)),
                        "bias_data":TensorConfig(data_gen=partial(generate_bias, batch, dics))
                        },
                    inputs={
                        "input_data":TensorConfig(data_gen=partial(generate_input1, batch, dics)),
                    },
                    outputs=["output_data"])

                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 3, 32, 16, 2],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 3, 64, 32, 2],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 3, 64, 32, 2],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # # for static_shape
        # clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)

    def test(self):
        self.run_test()

    def test_quant(self):
        self.run_test(quant=True)


class TrtConvertFcTest2(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824
        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.ones([batch, 3, 64, 14]).astype(np.float32)

        def generate_w(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["m"], attrs[0]["n"]]).astype(np.float32)

        def generate_bias(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["n"]]).astype(np.float32)


        for batch in [1, 4]:
            for [m,n] in [[14, 43]]:
                dics = [{
                    "in_num_col_dims" : 3,
                    # for my conveinence
                    "m" : m,
                    "n" : n,
                }, {}]

                ops_config = [{
                    "op_type": "fc",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "W": ["w_data"],
                        "Bias": ["bias_data"]
                    },
                    "op_outputs": {
                        "Out": ["output_data"]
                    },
                    "op_attrs": dics[0]
                },]

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "w_data":TensorConfig(data_gen=partial(generate_w, batch, dics)),
                        "bias_data":TensorConfig(data_gen=partial(generate_bias, batch, dics))
                        },
                    inputs={
                        "input_data":TensorConfig(data_gen=partial(generate_input1, batch, dics)),
                    },
                    outputs=["output_data"])

                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape():
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 3, 32, 14],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 3, 64, 64],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 3, 64, 64],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        # # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)

    def test(self):
        self.run_test()

    def test_quant(self):
        self.run_test(quant=True)


class TrtConvertFcTest2(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824
        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.ones([batch, 1, 1, 14]).astype(np.float32)

        def generate_w(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["m"], attrs[0]["n"]]).astype(np.float32)

        def generate_bias(batch, attrs: List[Dict[str, Any]]):
            return np.ones([attrs[0]["n"]]).astype(np.float32)


        for batch in [1, 4]:
            for [m,n] in [[14, 43]]:
                dics = [{
                    "in_num_col_dims" : 1,
                    # for my conveinence
                    "m" : m,
                    "n" : n,
                }, {}]

                ops_config = [{
                    "op_type": "fc",
                    "op_inputs": {
                        "Input": ["input_data"],
                        "W": ["w_data"],
                        "Bias": ["bias_data"]
                    },
                    "op_outputs": {
                        "Out": ["output_data"]
                    },
                    "op_attrs": dics[0]
                },]

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "w_data":TensorConfig(data_gen=partial(generate_w, batch, dics)),
                        "bias_data":TensorConfig(data_gen=partial(generate_bias, batch, dics))
                        },
                    inputs={
                        "input_data":TensorConfig(data_gen=partial(generate_input1, batch, dics)),
                    },
                    outputs=["output_data"])

                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape():
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 1, 1, 14],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 1, 1, 64],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 1, 1, 64],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        # # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 2), (1e-5, 1e-5)

    def test(self):
        self.run_test()

    def test_quant(self):
        self.run_test(quant=True)

if __name__ == "__main__":
    unittest.main()
