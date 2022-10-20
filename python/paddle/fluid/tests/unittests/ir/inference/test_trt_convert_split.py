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
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Any, Dict, List


class TrtConvertSplitTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # the dimensions of input and axis match
        if len(inputs['split_input'].shape) <= attrs[0]['axis']:
            return False

        #Sections and num cannot both be equal to 0.
        if len(attrs[0]['sections']) == 0:
            if attrs[0]['num'] == 0:
                return False

        #When sections and num are not both equal to 0, sections has higher priority.
        #The sum of sections should be equal to the input size.
        if len(attrs[0]['sections']) != 0:
            if attrs[0]['num'] != 0:
                return False
            if len(outputs) != len(attrs[0]['sections']):
                return False
            sum = 0
            for num in attrs[0]['sections']:
                sum += num
            if sum != inputs['split_input'].shape[attrs[0]['axis']]:
                return False

        #The size of num should be equal to the input dimension.
        if attrs[0]['num'] != 0:
            if len(outputs) != attrs[0]['num']:
                return False

        #Test AxisTensor and SectionsTensorList
        if self.num_input == 0:
            if self.dims == 2 and attrs[0]['sections'] == [
                    10, 14
            ] and len(outputs) == 2:
                return True
            else:
                return False

        return True

    def sample_program_configs(self):

        def generate_input1(attrs: List[Dict[str, Any]], batch):
            if self.dims == 4:
                return np.random.random([batch, 3, 3, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.random.random([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.random.random([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.random.random([24]).astype(np.float32)

        def generate_AxisTensor(attrs: List[Dict[str, Any]]):
            return np.ones([1]).astype(np.int32)

        def generate_SectionsTensorList1(attrs: List[Dict[str, Any]]):
            return np.array([10]).astype(np.int32)

        def generate_SectionsTensorList2(attrs: List[Dict[str, Any]]):
            return np.array([14]).astype(np.int32)

        for num_input in [0, 1]:
            for dims in [1, 2, 3, 4]:
                for batch in [3, 6, 9]:
                    for Out in [["output_var0", "output_var1"],
                                ["output_var0", "output_var1", "output_var2"]]:
                        for sections in [[], [1, 2], [2, 1], [10, 14],
                                         [1, 1, 1], [2, 2, 2], [3, 3, 3],
                                         [3, 7, 14]]:
                            for num in [0, 3]:
                                for axis in [0, 1, 2, 3]:
                                    self.batch = batch
                                    self.num_input = num_input
                                    self.dims = dims
                                    dics = [{
                                        "sections": sections,
                                        "num": num,
                                        "axis": axis
                                    }, {}]

                                    dics_intput = [{
                                        "X": ["split_input"],
                                        "AxisTensor": ["AxisTensor"],
                                        "SectionsTensorList": [
                                            "SectionsTensorList1",
                                            "SectionsTensorList2"
                                        ]
                                    }, {
                                        "X": ["split_input"]
                                    }]
                                    dics_intputs = [{
                                        "AxisTensor":
                                        TensorConfig(data_gen=partial(
                                            generate_AxisTensor, dics)),
                                        "SectionsTensorList1":
                                        TensorConfig(data_gen=partial(
                                            generate_SectionsTensorList1,
                                            dics)),
                                        "SectionsTensorList2":
                                        TensorConfig(data_gen=partial(
                                            generate_SectionsTensorList2, dics))
                                    }, {}]

                                    ops_config = [{
                                        "op_type":
                                        "split",
                                        "op_inputs":
                                        dics_intput[num_input],
                                        "op_outputs": {
                                            "Out": Out
                                        },
                                        "op_attrs":
                                        dics[0]
                                    }]
                                    ops = self.generate_op_config(ops_config)
                                    program_config = ProgramConfig(
                                        ops=ops,
                                        weights=dics_intputs[num_input],
                                        inputs={
                                            "split_input":
                                            TensorConfig(data_gen=partial(
                                                generate_input1, dics, batch))
                                        },
                                        outputs=Out)

                                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "split_input": [1, 3 - 1, 3 - 1, 24 - 1]
                }
                self.dynamic_shape.max_input_shape = {
                    "split_input": [9, 3 + 1, 3 + 1, 24 + 1]
                }
                self.dynamic_shape.opt_input_shape = {
                    "split_input": [1, 3, 3, 24]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "split_input": [1, 3 - 1, 24 - 1]
                }
                self.dynamic_shape.max_input_shape = {
                    "split_input": [9, 3 + 1, 24 + 1]
                }
                self.dynamic_shape.opt_input_shape = {"split_input": [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "split_input": [1, 24 - 1]
                }
                self.dynamic_shape.max_input_shape = {
                    "split_input": [9, 24 + 1]
                }
                self.dynamic_shape.opt_input_shape = {"split_input": [1, 24]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"split_input": [24 - 1]}
                self.dynamic_shape.max_input_shape = {"split_input": [24 + 1]}
                self.dynamic_shape.opt_input_shape = {"split_input": [24]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if len(program_config.outputs) == 2:
                if attrs[0]['axis'] != 0:
                    return 1, 3
                else:
                    return 0, 4
            else:
                if attrs[0]['axis'] != 0:
                    return 1, 4
                else:
                    return 0, 5

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        self.trt_param.max_batch_size = 9
        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-3

    def add_skip_trt_case(self):

        def teller1(program_config, predictor_config):
            if len(program_config.weights) == 3:
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_SUPPORT,
            "INPUT AxisTensor AND SectionsTensorList NOT SUPPORT.")

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
