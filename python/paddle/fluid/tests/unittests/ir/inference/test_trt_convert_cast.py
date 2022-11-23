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

<<<<<<< HEAD
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
<<<<<<< HEAD
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertCastTest(TrtLayerAutoScanTest):

=======
from typing import List


class TrtConvertCastTest(TrtLayerAutoScanTest):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['in_dtype'] == 0:
            return False
        if attrs[0]['in_dtype'] in [4, 5] and attrs[0]['out_dtype'] == 4:
            return False
<<<<<<< HEAD
        if attrs[0]['in_dtype'] not in [
                2, 4, 5
        ] or attrs[0]['out_dtype'] not in [2, 4, 5]:
=======

        out_dtype = [2, 4, 5]
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 > 8400:
            out_dtype.insert(3, 0)

        if (
            attrs[0]['in_dtype'] not in [2, 4, 5]
            or attrs[0]['out_dtype'] not in out_dtype
        ):
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return False
        return True

    def sample_program_configs(self):
<<<<<<< HEAD

=======
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
        def generate_input(type):
            if type == 0:
                return np.ones([1, 3, 64, 64]).astype(np.bool)
            elif type == 2:
                return np.ones([1, 3, 64, 64]).astype(np.int32)
            elif type == 4:
                return np.ones([1, 3, 64, 64]).astype(np.float16)
            else:
                return np.ones([1, 3, 64, 64]).astype(np.float32)

<<<<<<< HEAD
        for in_dtype in [0, 2, 4, 5, 6]:
            for out_dtype in [0, 2, 4, 5, 6]:
                dics = [{"in_dtype": in_dtype, "out_dtype": out_dtype}]

                ops_config = [{
                    "op_type": "cast",
                    "op_inputs": {
                        "X": ["input_data"]
                    },
                    "op_outputs": {
                        "Out": ["cast_output_data"]
                    },
                    "op_attrs": dics[0]
                }]
=======
        for in_dtype in [0, 2, 5, 6]:
            for out_dtype in [0, 2, 5, 6]:
                self.out_dtype = out_dtype
                dics = [
                    {"in_dtype": in_dtype, "out_dtype": out_dtype},
                    {"in_dtype": out_dtype, "out_dtype": in_dtype},
                ]

                ops_config = [
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["input_data"]},
                        "op_outputs": {"Out": ["cast_output_data0"]},
                        "op_attrs": dics[0],
                    },
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["cast_output_data0"]},
                        "op_outputs": {"Out": ["cast_output_data1"]},
                        "op_attrs": dics[1],
                    },
                ]

>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
<<<<<<< HEAD
                        "input_data":
                        TensorConfig(data_gen=partial(generate_input, in_dtype))
                    },
                    outputs=["cast_output_data"])
=======
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input, in_dtype)
                        )
                    },
                    outputs=["cast_output_data1"],
                )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
=======
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [1, 3, 64, 64]}
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
<<<<<<< HEAD
=======
            if not dynamic_shape and self.out_dtype == 0:
                return 0, 4
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-2
=======
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-2
=======
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
