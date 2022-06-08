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

import os

os.environ["FLAGS_test_allreduce_plugin"] = "1"
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import paddle.distributed.fleet as fleet
import unittest


class TrtConvertCAllreduceTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):

        def generate_input1():
            return np.ones([3, 32]).astype(np.float32)

        for op_type in ["c_allreduce_sum"]:
            dics = [{"ring_id": 0, "use_calc_stream": True}]

            ops_config = [{
                "op_type": op_type,
                "op_inputs": {
                    "X": ["input_data"]
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
                inputs={"input_data": TensorConfig(data_gen=generate_input1)},
                outputs=["output_data"])

            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 16]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 32]}
            self.dynamic_shape.opt_input_shape = {"input_data": [3, 32]}

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
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5

    def test(self):
        # It is an operator for communication but not calculation.
        # So there is no need to check the diff between TRT's plugin and
        # Paddle's operator.
        self.run_test(skip_baseline=True)


if __name__ == "__main__":
    unittest.main()
