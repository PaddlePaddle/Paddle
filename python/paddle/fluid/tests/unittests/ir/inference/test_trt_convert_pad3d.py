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
import os

# set cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

class TrtConvertPad3d(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            return np.ones([2, 3, 3, 8, 8]).astype(np.float32)

        def generate_paddings(p):
            return np.array(p).astype(np.int32)

        for value in [True, False]:
            for paddings in [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 1, 2, 1, 2],
                [1, 1, 1, 1, 1, 1],
                [0, 0, -1, -1, 1, 1],
            ]:
                for mode in ['tensor', 'list']:
                    for pad_mode in ['constant', 'reflect', 'replicate']:
                        if mode == 'list':
                            dics = [
                                {
                                    "value": value,
                                    "data_format": "NCDHW",
                                    "mode": pad_mode,
                                    "paddings": paddings,
                                },
                                {},
                            ]
                            ops_config = [
                                {
                                    "op_type": "pad3d",
                                    "op_inputs": {"X": ["input_data"]},
                                    "op_outputs": {"Out": ["output_data"]},
                                    "op_attrs": dics[0],
                                }
                            ]
                            ops = self.generate_op_config(ops_config)
                            inputs = {
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input1)
                                )
                            }
                        else:
                            dics = [
                                {
                                    "value": value,
                                    "data_format": "NCDHW",
                                    "mode": pad_mode,
                                    "paddings": [0],
                                },
                                {},
                            ]
                            ops_config = [
                                {
                                    "op_type": "pad3d",
                                    "op_inputs": {
                                        "X": ["input_data"],
                                        "Paddings": ["input_padding"],
                                    },
                                    "op_outputs": {"Out": ["output_data"]},
                                    "op_attrs": dics[0],
                                }
                            ]
                            ops = self.generate_op_config(ops_config)
                            inputs = {
                                "input_data": TensorConfig(
                                    data_gen=partial(generate_input1)
                                ),
                                "input_padding": TensorConfig(
                                    data_gen=partial(generate_paddings, paddings)
                                ),
                            }
                        for i in range(10):
                            program_config = ProgramConfig(
                                ops=ops,
                                weights={},
                                inputs=inputs,
                                outputs=["output_data"],
                            )

                            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [3, 2, 3, 8, 8],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [3, 4, 7, 8, 8],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [3, 2, 3, 8, 8],
            }
            if len(attrs[0]['paddings']) != 6:
                self.dynamic_shape.min_input_shape["input_padding"] = [6]
                self.dynamic_shape.max_input_shape["input_padding"] = [6]
                self.dynamic_shape.opt_input_shape["input_padding"] = [6]

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape:
                if len(attrs[0]['paddings']) != 6:
                    return 0, 4
                return 0, 3
            else:
                if len(attrs[0]['paddings']) != 6:
                    return 1, 3
                return 1, 3

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

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
