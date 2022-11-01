# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Any, Dict, List
import unittest


class TrtConvertBilinearInterpV2Test(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.ones([1, 3, 64, 64]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]]):
            return np.random.uniform(low=0.5, high=6.0, size=(2)).astype(
                "float32"
            )

        for data_layout in ["NCHW", "NHWC"]:
            for scale_y in [2.0, -1.0, 0.0]:
                for scale_x in [2.0, -1.0, 0.0]:
                    scale = [scale_y, scale_x]
                    for out_h in [32, 64, 128, 192]:
                        for out_w in [32, 64]:
                            dics = [
                                {
                                    "data_layout": data_layout,
                                    "interp_method": "bilinear",
                                    "align_corners": False,
                                    "align_mode": 0,
                                    "scale": scale,
                                    "out_h": out_h,
                                    "out_w": out_w,
                                }
                            ]

                            ops_config = [
                                {
                                    "op_type": "bilinear_interp_v2",
                                    "op_inputs": {
                                        "X": ["input_data"],
                                        "Scale": ["input_scale"],
                                    },
                                    "op_outputs": {
                                        "Out": [
                                            "bilinear_interp_v2_output_data"
                                        ]
                                    },
                                    "op_attrs": dics[0],
                                }
                            ]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={
                                    "input_scale": TensorConfig(
                                        data_gen=partial(generate_input2, dics)
                                    )
                                },
                                inputs={
                                    "input_data": TensorConfig(
                                        data_gen=partial(generate_input1, dics)
                                    )
                                },
                                outputs=["bilinear_interp_v2_output_data"],
                            )

                            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
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
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
