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
import itertools


class TrtConvertPool2dTest(TrtLayerAutoScanTest):

    def is_paddings_valid(self, program_config: ProgramConfig) -> bool:
        exclusive = program_config.ops[0].attrs['exclusive']
        paddings = program_config.ops[0].attrs['paddings']
        ksize = program_config.ops[0].attrs['ksize']
        pooling_type = program_config.ops[0].attrs['pooling_type']
        global_pooling = program_config.ops[0].attrs['global_pooling']
        if global_pooling == False:
            if pooling_type == 'avg':
                for index in range(len(ksize)):
                    if ksize[index] <= paddings[index]:
                        return False
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:
            if program_config.ops[0].attrs['pooling_type'] == 'avg':
                return False
        return True

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return self.is_paddings_valid(program_config)

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        strides_options = [[1, 2]]
        paddings_options = [[0, 2]]
        pooling_type_options = ['max', 'avg']
        padding_algorithm_options = ['EXPLICIT', 'SAME', 'VAILD']
        ksize_options = [[2, 3], [3, 3]]
        data_format_options = ['NCHW']
        global_pooling_options = [True, False]
        exclusive_options = [True, False]
        adaptive_option = [True, False]
        ceil_mode_options = [True, False]

        configurations = [
            strides_options, paddings_options, pooling_type_options,
            padding_algorithm_options, ksize_options, data_format_options,
            global_pooling_options, exclusive_options, adaptive_option,
            ceil_mode_options
        ]

        for (strides, paddings, pooling_type, padding_algorithm, ksize,
             data_format, global_pooling, exclusive, adaptive,
             ceil_mode) in itertools.product(*configurations):

            attrs = [{
                "strides": strides,
                "paddings": paddings,
                "pooling_type": pooling_type,
                "padding_algorithm": padding_algorithm,
                "ksize": ksize,
                "data_format": data_format,
                "global_pooling": global_pooling,
                "exclusive": exclusive,
                "adaptive": adaptive,
                "ceil_mode": ceil_mode,
            }]

            ops_config = [{
                "op_type": "pool2d",
                "op_inputs": {
                    "X": ["input_data"]
                },
                "op_outputs": {
                    "Out": ["output_data"]
                },
                "op_attrs": attrs[0]
            }]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input1, attrs))
                },
                outputs=["output_data"])

            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
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
            attrs, False), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-3, 1e-3)

    def add_skip_trt_case(self):

        def teller(program_config, predictor_config):
            if program_config.ops[0].attrs['pooling_type'] == 'avg' and \
               program_config.ops[0].attrs['global_pooling'] == False and \
               program_config.ops[0].attrs['exclusive'] == True and \
               program_config.ops[0].attrs['adaptive'] == False and \
               program_config.ops[0].attrs['ceil_mode'] == True:
                return True
            return False

        self.add_skip_case(
            teller, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The results of some cases are Nan, but the results of TensorRT and GPU are the same."
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
