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

import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertLinspaceTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_num_data(attrs: List[Dict[str, Any]]):
            return np.array([5]).astype(np.float32)

        def generate_start_data(attrs: List[Dict[str, Any]]):
            return np.array([0]).astype(np.float32)

        def generate_stop_data(attrs: List[Dict[str, Any]]):
            return np.array([10]).astype(np.float32)

        for start in [0]:
            for num_input in [0]:
                for dtype in [5, 2, 3]:
                    for str_value in ["0"]:
                        dics_attrs = [
                            {
                                "dtype": dtype,
                            },
                        ]
                        dics_intput = [
                            {
                                "Num": ["num_data"],
                                "Start": ["start_data"],
                                "Stop": ["stop_data"],
                            },
                        ]
                        ops_config = [
                            {
                                "op_type": "linspace",
                                "op_inputs": dics_intput[0],
                                "op_outputs": {
                                    "Out": ["out_data"],
                                },
                                "op_attrs": dics_attrs[0],
                            },
                        ]

                        def generate_input():
                            return np.random.random([1, 1]).astype(np.float32)

                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "num_data": TensorConfig(
                                    data_gen=partial(
                                        generate_num_data, dics_attrs
                                    )
                                ),
                                "start_data": TensorConfig(
                                    data_gen=partial(
                                        generate_start_data, dics_attrs
                                    )
                                ),
                                "stop_data": TensorConfig(
                                    data_gen=partial(
                                        generate_stop_data, dics_attrs
                                    )
                                ),
                            },
                            outputs=["out_data"],
                        )

                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "num_data": [1],
                "start_data": [1],
                "stop_data": [1],
            }
            self.dynamic_shape.max_input_shape = {
                "num_data": [2],
                "start_data": [2],
                "stop_data": [2],
            }
            self.dynamic_shape.opt_input_shape = {
                "num_data": [1],
                "start_data": [1],
                "stop_data": [1],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 3, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # Don't test static shape

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
