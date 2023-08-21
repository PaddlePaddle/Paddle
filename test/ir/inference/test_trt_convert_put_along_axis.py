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


class TrtConvertPutAlongAxis(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8200:
            return False
        return True

    def sample_program_configs(self):
        def generate_input(attrs: List[Dict[str, Any]]):
            return np.random.random(attrs["input_shape"]).astype(np.float32)

        def generate_value(attrs: List[Dict[str, Any]]):
            return np.random.random(attrs["val_shape"]).astype(np.float32)

        def generate_index(attrs: List[Dict[str, Any]]):
            input_shape = attrs["input_shape"]
            val_shape = attrs["val_shape"]
            axis = attrs["axis"]
            top = input_shape[axis]
            count = val_shape[axis]
            temp = np.random.choice(top, count, replace=False)
            multivalue_left = 1
            multivalue_right = 1
            for i in range(len(val_shape)):
                if i < axis:
                    multivalue_left *= val_shape[i]
                elif i > axis:
                    multivalue_right *= val_shape[i]
            for i in range(multivalue_right - 1):
                temp = np.vstack(
                    (temp, np.random.choice(top, count, replace=False))
                )
            temp = temp.transpose()
            index = temp.copy()
            for i in range(multivalue_left - 1):
                index = np.vstack((index, temp))
            return index.reshape(attrs['val_shape'])

        for input_shape in [[3, 1, 500, 100]]:
            for val_shape in [[3, 1, 200, 100]]:
                for axis in [2]:
                    dics = [
                        {
                            "input_shape": input_shape,
                            "val_shape": val_shape,
                            "axis": axis,
                        },
                    ]

                    ops_config = [
                        {
                            "op_type": "put_along_axis",
                            "op_inputs": {
                                "Input": ["input_data"],
                                "Value": ["value_data"],
                                "Index": ["index_data"],
                            },
                            "op_outputs": {"Result": ["result_data"]},
                            "op_attrs": {
                                "Axis": axis,
                            },
                        },
                    ]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data": TensorConfig(
                                data_gen=partial(generate_input, dics[0])
                            ),
                            "value_data": TensorConfig(
                                data_gen=partial(generate_value, dics[0])
                            ),
                            "index_data": TensorConfig(
                                data_gen=partial(generate_index, dics[0])
                            ),
                        },
                        outputs=["result_data"],
                    )
                    yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [3, 1, 500, 100],
                "value_data": [3, 1, 200, 100],
                "index_data": [3, 1, 200, 100],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 1, 500, 100],
                "value_data": [3, 1, 200, 100],
                "index_data": [3, 1, 200, 100],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [3, 1, 500, 100],
                "value_data": [3, 1, 200, 100],
                "index_data": [3, 1, 200, 100],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                return 1, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
