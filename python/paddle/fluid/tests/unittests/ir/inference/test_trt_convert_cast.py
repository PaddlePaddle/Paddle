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
from paddle.framework import convert_np_dtype_to_dtype_


class TrtConvertCastTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['in_dtype'] == 0:
            return False
        if attrs[0]['in_dtype'] in [4, 5] and attrs[0]['out_dtype'] == 4:
            return False

        out_dtype = [2, 4, 5]
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 > 8400:
            out_dtype.insert(3, 0)

        if (
            attrs[0]['in_dtype'] not in [2, 4, 5]
            or attrs[0]['out_dtype'] not in out_dtype
        ):
            return False
        return True

    def sample_program_configs(self):
        def generate_input(type):
            return np.ones([1, 3, 64, 64]).astype(type)

        for in_dtype in [np.bool_, np.int32, np.float32, np.float64]:
            for out_dtype in [np.bool_, np.int32, np.float32, np.float64]:
                self.out_dtype = out_dtype
                dics = [
                    {
                        "in_dtype": convert_np_dtype_to_dtype_(in_dtype),
                        "out_dtype": convert_np_dtype_to_dtype_(out_dtype),
                    },
                    {
                        "in_dtype": convert_np_dtype_to_dtype_(out_dtype),
                        "out_dtype": convert_np_dtype_to_dtype_(in_dtype),
                    },
                ]

                ops_config = [
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["input_data"]},
                        "op_outputs": {"Out": ["cast_output_data0"]},
                        "op_attrs": dics[0],
                        "op_outputs_dtype": {"cast_output_data0": out_dtype},
                    },
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["cast_output_data0"]},
                        "op_outputs": {"Out": ["cast_output_data1"]},
                        "op_attrs": dics[1],
                        "op_outputs_dtype": {"cast_output_data1": in_dtype},
                    },
                ]

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input, in_dtype)
                        )
                    },
                    outputs=["cast_output_data1"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and self.out_dtype == 0:
                return 0, 4
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
