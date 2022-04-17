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
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertFlattenContiguousRangeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(batch):
            return np.random.random([2, batch, 4, 8, 3]).astype(np.float32)

        for batch in [1, 2, 4]:
            for start_axis in range(5):
                for stop_axis in range(start_axis, 5):
                    type = "flatten_contiguous_range"
                    op_outputs = {
                        "Out": ["output_data"],
                        "XShape": ["xshape_data"]
                    }
                    ops_config = [{
                        "op_type": type,
                        "op_inputs": {
                            "X": ["input_data"]
                        },
                        "op_outputs": op_outputs,
                        "op_attrs": {
                            "start_axis": start_axis,
                            "stop_axis": stop_axis,
                        }
                    }]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input, batch))
                        },
                        outputs=["output_data"])
                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [2, 1, 4, 8, 3]}
            self.dynamic_shape.max_input_shape = {"input_data": [2, 4, 4, 8, 3]}
            self.dynamic_shape.opt_input_shape = {"input_data": [2, 2, 4, 8, 3]}

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 >= 7000:
                if dynamic_shape:
                    return 1, 2
                else:
                    if attrs[0]['start_axis'] == 0:
                        return 0, 3
                    else:
                        return 1, 2
            else:
                return 0, 3

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
