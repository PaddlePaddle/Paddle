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


class TrtConvertNearestInterpTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if attrs[0]['scale'] <= 0 and (attrs[0]['out_h'] <= 0 or
                                       attrs[0]['out_w'] <= 0):
            return False
        if (attrs[0]['out_h'] <= 0) ^ (attrs[0]['out_w'] <= 0):
            return False

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        for data_layout in ["NCHW", "NHWC"]:
            for interp_method in ["nearest"]:
                for align_corners in [True, False]:
                    for scale in [2.0, -1.0, 0.0]:
                        for out_h in [32, 64, 128 - 32]:
                            for out_w in [32, -32]:
                                dics = [{
                                    "data_layout": data_layout,
                                    "interp_method": interp_method,
                                    "align_corners": align_corners,
                                    "scale": scale,
                                    "out_h": out_h,
                                    "out_w": out_w
                                }]

                                ops_config = [{
                                    "op_type": "nearest_interp",
                                    "op_inputs": {
                                        "X": ["input_data"]
                                    },
                                    "op_outputs": {
                                        "Out": ["nearest_interp_output_data"]
                                    },
                                    "op_attrs": dics[0]
                                }]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(generate_input1,
                                                             dics))
                                    },
                                    outputs=["nearest_interp_output_data"])

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
            return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
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
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-2

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if program_config.ops[0].attrs[
                    'scale'] <= 0 and self.dynamic_shape.min_input_shape:
                return True
            if program_config.ops[0].attrs['align_corners'] == True:
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
            "NOT Implemented: we need to add support scale <= 0 in dynamic shape in the future"
        )

        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
