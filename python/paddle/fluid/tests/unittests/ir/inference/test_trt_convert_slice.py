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


class TrtConvertSliceTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        for x in attrs[0]["decrease_axis"]:
            if x < 0:
                return False
        for x in range(len(attrs[0]["axes"])):
            start = 0
            end = 0
            if attrs[0]["starts"][x] < 0:
                start = attrs[0]["starts"][x] + inputs['input_data'].shape[
                    attrs[0]["axes"][x]]
            else:
                start = attrs[0]["starts"][x]
            if attrs[0]["ends"][x] < 0:
                end = attrs[0]["ends"][x] + inputs['input_data'].shape[attrs[0][
                    "axes"][x]]
            else:
                end = attrs[0]["ends"][x]
            start = max(0, start)
            end = max(0, end)
            if start >= end:
                return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([6, 6, 64, 64]).astype(np.float32)

        for axes in [[0, 1], [1, 3], [2, 3]]:
            for starts in [[0, 1]]:
                for ends in [[2, 2], [5, 5]]:
                    for decrease_axis in [[], [1], [2], [-1], [-100]]:
                        for infer_flags in [[-1]]:
                            dics = [{
                                "axes": axes,
                                "starts": starts,
                                "ends": ends,
                                "decrease_axis": decrease_axis,
                                "infer_flags": infer_flags
                            }]

                            ops_config = [{
                                "op_type": "slice",
                                "op_inputs": {
                                    "Input": ["input_data"]
                                },
                                "op_outputs": {
                                    "Out": ["slice_output_data"]
                                },
                                "op_attrs": dics[0]
                            }]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs={
                                    "input_data": TensorConfig(data_gen=partial(
                                        generate_input1, dics))
                                },
                                outputs=["slice_output_data"])

                            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
            self.dynamic_shape.max_input_shape = {"input_data": [8, 8, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [6, 6, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            inputs = program_config.inputs
            if dynamic_shape == True and len(attrs[0]["decrease_axis"]) == 0:
                return 1, 2
            if dynamic_shape == True and len(attrs[0]["decrease_axis"]) != 1:
                return 0, 3
            if dynamic_shape == False and len(attrs[0]["decrease_axis"]) != 0:
                return 0, 3
            if dynamic_shape:
                for i in range(len(attrs[0]["starts"])):
                    if attrs[0]["starts"][i] < 0 or attrs[0]["ends"][i] < 0:
                        return 0, 3
            if not dynamic_shape:
                for x in attrs[0]["axes"]:
                    if x == 0:
                        return 0, 3
            return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]
        self.trt_param.max_batch_size = 9
        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-4

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-4

    def test(self):
        # TODO(inference): fix.
        # trt6 and trt7.1 has bug.
        # trt7.2 deserialize has bug.
        self.run_test()
        pass


if __name__ == "__main__":
    unittest.main()
