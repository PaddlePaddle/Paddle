# Copyright (c) 422 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer

class TrtConvertPutAlongAxis(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1():
            b = np.random.random([1,3,3]).astype(np.float32)
            print(b)
            return b

        def generate_input2():
            a = np.random.random([3]).astype(np.float32)
            return a

        ops_config = [
            {
                "op_type": "put_along_axis",
                "op_inputs": {
                    "Input": ["output_data"],
                    "ValueTensor": ["update_data"],
                },
                "op_outputs": {"Out": ["output_data"]},
                "op_attrs": {
                    "axes": [0],
                    "starts": [0],
                    "ends": [1],
                    "steps": [1],
                    "decrease_axes":[0]
                },
            },
        ]
        ops = self.generate_op_config(ops_config)
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1)),
                "update_data": TensorConfig(data_gen=partial(generate_input2)),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1,3,3],
                "update_data": [3],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [1,3,3],
                "update_data": [3],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1,3,3],
                "update_data": [3],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape:
                ver = paddle_infer.get_trt_compile_version()
                if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 840:
                    return 1, 5
                return 1, 3

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
<<<<<<< HEAD
        # self.trt_param.precision = paddle_infer.PrecisionType.Half
        # yield self.create_inference_config(), generate_trt_nodes_num(
        #     attrs, True
        # ), (1e-3, 1e-3)
=======
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)
>>>>>>> develop

    def test(self):
        self.run_test()

if __name__ == "__main__":
    unittest.main()
