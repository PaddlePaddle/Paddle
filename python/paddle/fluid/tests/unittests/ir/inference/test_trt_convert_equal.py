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
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TrtConvertElementwiseTest_one_input_corner_case(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['axis'] == 0:
<<<<<<< HEAD
            return False
=======
            return false
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8415:
            return False
        return True

    def sample_program_configs(self):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

<<<<<<< HEAD
        for op_type in ["equal", "not_equal"]:
            for batch in [1, 2, 4]:
                for shape in [[batch, 1], [batch, 1, 32], [batch, 1, 16, 32]]:
                    for axis in [-1 if len(shape) == 1 else 1]:
                        self.dims = len(shape)
                        dics = [{"axis": axis}, {"in_dtype": 0, "out_dtype": 5}]
                        ops_config = [
                            {
                                "op_type": op_type,
                                "op_inputs": {
                                    "X": ["input_data1"],
                                    "Y": ["input_data2"],
                                },
                                "op_outputs": {"Out": ["compare_output_data"]},
                                "op_attrs": dics[0],
                                "outputs_dtype": {
                                    "compare_output_data": np.bool_
                                },
                            },
                            {
                                "op_type": "cast",
                                "op_inputs": {"X": ["compare_output_data"]},
                                "op_outputs": {"Out": ["output_data"]},
                                "op_attrs": dics[1],
                                "outputs_dtype": {"output_data": np.float32},
                            },
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "input_data1": TensorConfig(
                                    data_gen=partial(generate_input, shape)
                                ),
                                "input_data2": TensorConfig(
                                    data_gen=partial(generate_input, shape)
                                ),
                            },
                            outputs=["output_data"],
                        )
                        yield program_config
=======
        for batch in [1, 2, 4]:
            for shape in [[batch, 1], [batch, 1, 32], [batch, 1, 16, 32]]:
                for axis in [-1 if len(shape) == 1 else 1]:
                    self.dims = len(shape)
                    dics = [{"axis": axis}, {"in_dtype": 0, "out_dtype": 5}]
                    ops_config = [
                        {
                            "op_type": "equal",
                            "op_inputs": {
                                "X": ["input_data1"],
                                "Y": ["input_data2"],
                            },
                            "op_outputs": {"Out": ["compare_output_data"]},
                            "op_attrs": dics[0],
                        },
                        {
                            "op_type": "cast",
                            "op_inputs": {"X": ["compare_output_data"]},
                            "op_outputs": {"Out": ["output_data"]},
                            "op_attrs": dics[1],
                        },
                    ]
                    ops = self.generate_op_config(ops_config)

                    program_config = ProgramConfig(
                        ops=ops,
                        weights={},
                        inputs={
                            "input_data1": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                            "input_data2": TensorConfig(
                                data_gen=partial(generate_input, shape)
                            ),
                        },
                        outputs=["output_data"],
                    )

                    yield program_config
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The input.dims[1] must be equal to the weight's length.
            if self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 1],
                    "input_data2": [1, 1],
                }
                self.dynamic_shape.max_input_shape = {
                    "input_data1": [4, 1],
                    "input_data2": [4, 1],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 1],
                    "input_data2": [2, 1],
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 1, 4],
                    "input_data2": [1, 1, 4],
                }
                self.dynamic_shape.max_input_shape = {
<<<<<<< HEAD
                    "input_data1": [4, 1, 32],
                    "input_data2": [4, 1, 32],
=======
                    "input_data1": [4, 1, 256],
                    "input_data2": [1, 1, 256],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 1, 16],
                    "input_data2": [2, 1, 16],
                }
            elif self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input_data1": [1, 1, 4, 4],
                    "input_data2": [1, 1, 4, 4],
                }
                self.dynamic_shape.max_input_shape = {
<<<<<<< HEAD
                    "input_data1": [4, 1, 64, 32],
                    "input_data2": [4, 1, 64, 32],
=======
                    "input_data1": [4, 1, 128, 256],
                    "input_data2": [4, 1, 128, 256],
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                }
                self.dynamic_shape.opt_input_shape = {
                    "input_data1": [2, 1, 32, 16],
                    "input_data2": [2, 1, 32, 16],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
<<<<<<< HEAD
            if not dynamic_shape:
                return 0, 5
            if self.dims == 1:
                return 0, 3
            return 1, 3
=======
            if self.dims == 1:
                return 0, 3
            return 1, 2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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
<<<<<<< HEAD
        self.trt_param.workspace_size = 1 << 20
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.run_test()


if __name__ == "__main__":
    unittest.main()
