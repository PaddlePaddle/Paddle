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

import unittest
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertTemporalShiftTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs):
            T = attrs[0]["seg_num"]
            return np.ones([3 * T, 10, 64, 64]).astype(np.float32)

        for shift_value in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.49]:
            for T in range(2, 5):
                for data_format in ["NCHW", "NHWC"]:
                    dics = [{"shift_ratio": shift_value, "seg_num": T, "data_format": data_format}, {}]
                    ops_config = [
                        {
                            "op_type": "temporal_shift",
                            "op_inputs": {"X": ["input_data"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[0],
                        }
                    ]

                    ops = self.generate_op_config(ops_config)
                    for i in range(10):
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input1, dics)
                                ),
                            },
                            outputs=["output_data"],
                        )

                    yield program_config

    def sample_predictor_configs(
            self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [6, 10, 64, 64]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [20, 10, 64, 64]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [6, 10, 64, 64]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (0, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 3), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (0, 3), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (0, 3), 1e-3

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
