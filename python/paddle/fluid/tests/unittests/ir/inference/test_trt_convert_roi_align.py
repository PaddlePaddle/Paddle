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


class TrtConvertRoiAlignTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            return np.ones([batch, 256, 32, 32]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            return np.random.random([3, 4]).astype(np.float32)

        def generate_input3(attrs: List[Dict[str, Any]], batch):
            return np.random.random([batch]).astype(np.int32)

        for num_input in [0, 1]:
            for batch in [1, 2, 4]:
                for spatial_scale in [0.5, 0.6]:
                    for pooled_height in [7, 1]:
                        for pooled_width in [7, 1]:
                            for sampling_ratio in [-1, 4, 8]:
                                for aligned in [True, False]:
                                    self.num_input = num_input
                                    if num_input == 1:
                                        batch = 1
                                    dics = [{
                                        "spatial_scale": spatial_scale,
                                        "pooled_height": pooled_height,
                                        "pooled_width": pooled_width,
                                        "sampling_ratio": sampling_ratio,
                                        "aligned": aligned
                                    }, {}]
                                    dics_input = [{
                                        "X": ["roi_align_input"],
                                        "ROIs": ["ROIs"],
                                        "RoisNum": ["RoisNum"]
                                    }, {
                                        "X": ["roi_align_input"],
                                        "ROIs": ["ROIs"]
                                    }]
                                    program_input = [{
                                        "roi_align_input": TensorConfig(
                                            data_gen=partial(generate_input1,
                                                             dics, batch)),
                                        "ROIs": TensorConfig(data_gen=partial(
                                            generate_input2, dics, batch)),
                                        "RoisNum": TensorConfig(
                                            data_gen=partial(generate_input3,
                                                             dics, batch))
                                    }, {
                                        "roi_align_input": TensorConfig(
                                            data_gen=partial(generate_input1,
                                                             dics, batch)),
                                        "ROIs": TensorConfig(
                                            data_gen=partial(generate_input2,
                                                             dics, batch),
                                            lod=[[32, 3]])
                                    }]
                                    ops_config = [{
                                        "op_type": "roi_align",
                                        "op_inputs": dics_input[num_input],
                                        "op_outputs": {
                                            "Out": ["roi_align_out"]
                                        },
                                        "op_attrs": dics[0]
                                    }]
                                    ops = self.generate_op_config(ops_config)
                                    program_config = ProgramConfig(
                                        ops=ops,
                                        weights={},
                                        inputs=program_input[num_input],
                                        outputs=["roi_align_out"])

                                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.num_input == 0:
                self.dynamic_shape.min_input_shape = {
                    "roi_align_input": [1, 256, 32, 32],
                    "ROIs": [3, 4],
                    "RoisNum": [1]
                }
                self.dynamic_shape.max_input_shape = {
                    "roi_align_input": [1, 256, 64, 64],
                    "ROIs": [3, 4],
                    "RoisNum": [1]
                }
                self.dynamic_shape.opt_input_shape = {
                    "roi_align_input": [1, 256, 64, 64],
                    "ROIs": [3, 4],
                    "RoisNum": [1]
                }
            elif self.num_input == 1:
                self.dynamic_shape.min_input_shape = {
                    "roi_align_input": [1, 256, 32, 32],
                    "ROIs": [3, 4]
                }
                self.dynamic_shape.max_input_shape = {
                    "roi_align_input": [1, 256, 64, 64],
                    "ROIs": [3, 4]
                }
                self.dynamic_shape.opt_input_shape = {
                    "roi_align_input": [1, 256, 64, 64],
                    "ROIs": [3, 4]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if self.num_input == 0:
                if dynamic_shape == True:
                    return 0, 5
            elif self.num_input == 1:
                if dynamic_shape == True:
                    return 1, 3
                else:
                    return 0, 4
            return 0, 4

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
            if len(program_config.inputs) == 3:
                return True
            return False

        self.add_skip_case(teller1, SkipReasons.TRT_NOT_SUPPORT,
                           "INPUT RoisNum NOT SUPPORT")

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
