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
import unittest
from functools import partial
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer
from paddle.framework import convert_np_dtype_to_dtype_


class TrtConvertCastTest(TrtLayerAutoScanTest):
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertCastTest(TrtLayerAutoScanTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
<<<<<<< HEAD
        if attrs[0]['in_dtype'] not in [0, 1, 2, 3, 4, 5] or attrs[0][
            'out_dtype'
        ] not in [0, 1, 2, 3, 4, 5]:
            return False
        compile_version = paddle_infer.get_trt_compile_version()
        runtime_version = paddle_infer.get_trt_runtime_version()
        if (
            compile_version[0] * 1000
            + compile_version[1] * 100
            + compile_version[2] * 10
            < 8400
        ):
            return False
        if (
            runtime_version[0] * 1000
            + runtime_version[1] * 100
            + runtime_version[2] * 10
            < 8400
        ):
=======
        if attrs[0]['in_dtype'] == 0:
            return False
        if attrs[0]['in_dtype'] in [4, 5] and attrs[0]['out_dtype'] == 4:
            return False
        if attrs[0]['in_dtype'] not in [
                2, 4, 5
        ] or attrs[0]['out_dtype'] not in [2, 4, 5]:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False
        return True

    def sample_program_configs(self):
<<<<<<< HEAD
        def generate_input(type):
            return np.ones([1, 3, 64, 64]).astype(type)

        for in_dtype in [np.bool_, np.int32, np.float32, np.float64, np.int64]:
            for out_dtype in [
                np.bool_,
                np.int32,
                np.float32,
                np.float64,
                np.int64,
            ]:
                self.has_bool_dtype = (in_dtype == np.bool_) or (
                    out_dtype == np.bool_
                )
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
                        "outputs_dtype": {"cast_output_data0": out_dtype},
                    },
                    {
                        "op_type": "cast",
                        "op_inputs": {"X": ["cast_output_data0"]},
                        "op_outputs": {"Out": ["cast_output_data1"]},
                        "op_attrs": dics[1],
                        "outputs_dtype": {"cast_output_data1": in_dtype},
                    },
                ]
=======

        def generate_input(type):
            if type == 0:
                return np.ones([1, 3, 64, 64]).astype(np.bool)
            elif type == 2:
                return np.ones([1, 3, 64, 64]).astype(np.int32)
            elif type == 4:
                return np.ones([1, 3, 64, 64]).astype(np.float16)
            else:
                return np.ones([1, 3, 64, 64]).astype(np.float32)

        for in_dtype in [0, 2, 5, 6]:
            for out_dtype in [0, 2, 5, 6]:
                dics = [{
                    "in_dtype": in_dtype,
                    "out_dtype": out_dtype
                }, {
                    "in_dtype": out_dtype,
                    "out_dtype": in_dtype
                }]

                ops_config = [{
                    "op_type": "cast",
                    "op_inputs": {
                        "X": ["input_data"]
                    },
                    "op_outputs": {
                        "Out": ["cast_output_data0"]
                    },
                    "op_attrs": dics[0]
                }, {
                    "op_type": "cast",
                    "op_inputs": {
                        "X": ["cast_output_data0"]
                    },
                    "op_outputs": {
                        "Out": ["cast_output_data1"]
                    },
                    "op_attrs": dics[1]
                }]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
<<<<<<< HEAD
                        "input_data": TensorConfig(
                            data_gen=partial(generate_input, in_dtype)
                        )
                    },
                    outputs=["cast_output_data1"],
                )
=======
                        "input_data":
                        TensorConfig(data_gen=partial(generate_input, in_dtype))
                    },
                    outputs=["cast_output_data1"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [1, 3, 64, 64]}
=======
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 64, 64]}
            self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
<<<<<<< HEAD
            if not dynamic_shape and self.has_bool_dtype:
                return 0, 4
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2
=======
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2
=======
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
