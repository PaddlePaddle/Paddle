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

import unittest
from typing import List, Tuple

import numpy as np
from program_config import ProgramConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertLeakyEyeTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input_attr():
            if np.random.random() > 0.5:
                return np.random.randint(1, 320, size=2)
            return np.random.randint(1, 320), -1

        for dtype in [2, 4, 5]:
            for _ in range(6):

                num_rows, num_columns = generate_input_attr()
                attr_dic = {
                    "num_rows": num_rows,
                    "num_columns": num_columns,
                    "dtype": dtype,
                }

                ops_config = [
                    {
                        "op_type": "eye",
                        "op_outputs": {
                            "Out": ["out_data"],
                        },
                        "op_attrs": attr_dic,
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={},
                    outputs=["out_data"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> Tuple[paddle_infer.Config, List[int], float]:
        def generate_dynamic_shape(attrs):
            pass

        def clear_dynamic_shape():
            pass

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
        ), 1e-3

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

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
