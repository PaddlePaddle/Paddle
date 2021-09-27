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


class TrtConvertPool2dTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        for strides in [[1, 1], [2, 2], [1, 2]]:
            for paddings in [[0, 2], [0, 3], [1, 2, 3, 4]]:
                for pooling_type in ['max', 'avg']:
                    for padding_algotithm in ['EXPLICIT', 'SAME', 'VAILD']:
                        for ksize in [[2, 3], [3, 3]]:
                            for data_format in ['NCHW']:
                                for global_pooling in [True, False]:
                                    for exclusive in [True, False]:
                                        for adaptive in [True, False]:
                                            for ceil_mode in [True, False]:
                                                self.paddings = paddings

                                                dics = [{
                                                    "pooling_type":
                                                    pooling_type,
                                                    "ksize": ksize,
                                                    "data_fromat": data_format,
                                                    "padding_algorithm":
                                                    padding_algotithm,
                                                    "paddings": paddings,
                                                    "strides": strides,
                                                    "data_format": data_format,
                                                    "global_pooling":
                                                    global_pooling,
                                                    "exclusive": exclusive,
                                                    "adaptive": adaptive,
                                                    "ceil_mode": ceil_mode
                                                }]

                                                ops_config = [{
                                                    "op_type": "pool2d",
                                                    "op_inputs": {
                                                        "X": ["input_data"],
                                                    },
                                                    "op_outputs": {
                                                        "Out": ["output_data"]
                                                    },
                                                    "op_attrs": dics[0]
                                                }]
                                                ops = self.generate_op_config(
                                                    ops_config)

                                                program_config = ProgramConfig(
                                                    ops=ops,
                                                    weights={},
                                                    inputs={
                                                        "input_data":
                                                        TensorConfig(
                                                            data_gen=partial(
                                                                generate_input1,
                                                                dics))
                                                    },
                                                    outputs=["output_data"])

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
            if self.paddings == [0, 3] or attrs[0][
                    'global_pooling'] == True or attrs[0]['ceil_mode'] == True:
                return 0, 3
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
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if len(program_config.ops[0].attrs['paddings']) == 4:
                return True
            return False

        self.add_skip_case(teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
                           "4-dims paddings are not support for trt now.")

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
