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

<<<<<<< HEAD
import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertStridedSliceTest(TrtLayerAutoScanTest):
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertStridedSliceTest(TrtLayerAutoScanTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        return True

    def sample_program_configs(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.random.random([1, 56, 56, 192]).astype(np.float32)

        for axes in [[1, 2]]:
            for starts in [[1, 1]]:
                for ends in [[10000000, 10000000]]:
                    for decrease_axis in [[]]:
                        for infer_flags in [[1, 1]]:
                            for strides in [[2, 2]]:
<<<<<<< HEAD
                                dics = [
                                    {
                                        "axes": axes,
                                        "starts": starts,
                                        "ends": ends,
                                        "decrease_axis": decrease_axis,
                                        "infer_flags": infer_flags,
                                        "strides": strides,
                                    }
                                ]

                                ops_config = [
                                    {
                                        "op_type": "strided_slice",
                                        "op_inputs": {"Input": ["input_data"]},
                                        "op_outputs": {
                                            "Out": ["slice_output_data"]
                                        },
                                        "op_attrs": dics[0],
                                    }
                                ]
=======
                                dics = [{
                                    "axes": axes,
                                    "starts": starts,
                                    "ends": ends,
                                    "decrease_axis": decrease_axis,
                                    "infer_flags": infer_flags,
                                    "strides": strides
                                }]

                                ops_config = [{
                                    "op_type": "strided_slice",
                                    "op_inputs": {
                                        "Input": ["input_data"]
                                    },
                                    "op_outputs": {
                                        "Out": ["slice_output_data"]
                                    },
                                    "op_attrs": dics[0]
                                }]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs={
<<<<<<< HEAD
                                        "input_data": TensorConfig(
                                            data_gen=partial(
                                                generate_input1, dics
                                            )
                                        )
                                    },
                                    outputs=["slice_output_data"],
                                )
=======
                                        "input_data":
                                        TensorConfig(data_gen=partial(
                                            generate_input1, dics))
                                    },
                                    outputs=["slice_output_data"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
=======
            self, program_config) -> (paddle_infer.Config, List[int], float):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 56, 56, 192]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [8, 56, 56, 192]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [4, 56, 56, 192]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            inputs = program_config.inputs

            if dynamic_shape:
                for i in range(len(attrs[0]["starts"])):
                    if attrs[0]["starts"][i] < 0 or attrs[0]["ends"][i] < 0:
                        return 0, 3
            if not dynamic_shape:
                for x in attrs[0]["axes"]:
                    if x == 0:
                        return 0, 3
            ver = paddle_infer.get_trt_compile_version()
            if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 7000:
                return 0, 3
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
=======
            attrs, False), 1e-5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, True
        ), 1e-5
=======
            attrs, True), 1e-5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test(self):
        self.run_test()


class TrtConvertStridedSliceTest2(TrtLayerAutoScanTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def generate_input1(attrs: List[Dict[str, Any]]):
            return np.random.random([1, 56, 56, 192]).astype(np.float32)

        for axes in [[1, 2], [2, 3], [1, 3]]:
<<<<<<< HEAD
            for starts in [
                [-10, 1],
                [-10, 20],
                [-10, 15],
                [-10, 16],
                [-10, 20],
            ]:
=======
            for starts in [[-10, 1], [-10, 20], [-10, 15], [-10, 16], [-10,
                                                                       20]]:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                for ends in [[-9, 10000], [-9, -1], [-9, 40]]:
                    for decrease_axis in [[]]:
                        for infer_flags in [[1, 1]]:
                            for strides in [[2, 2]]:
<<<<<<< HEAD
                                dics = [
                                    {
                                        "axes": axes,
                                        "starts": starts,
                                        "ends": ends,
                                        "decrease_axis": [axes[0]],
                                        "infer_flags": infer_flags,
                                        "strides": strides,
                                    }
                                ]

                                ops_config = [
                                    {
                                        "op_type": "strided_slice",
                                        "op_inputs": {"Input": ["input_data"]},
                                        "op_outputs": {
                                            "Out": ["slice_output_data"]
                                        },
                                        "op_attrs": dics[0],
                                    }
                                ]
=======
                                dics = [{
                                    "axes": axes,
                                    "starts": starts,
                                    "ends": ends,
                                    "decrease_axis": [axes[0]],
                                    "infer_flags": infer_flags,
                                    "strides": strides
                                }]

                                ops_config = [{
                                    "op_type": "strided_slice",
                                    "op_inputs": {
                                        "Input": ["input_data"]
                                    },
                                    "op_outputs": {
                                        "Out": ["slice_output_data"]
                                    },
                                    "op_attrs": dics[0]
                                }]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={},
                                    inputs={
<<<<<<< HEAD
                                        "input_data": TensorConfig(
                                            data_gen=partial(
                                                generate_input1, dics
                                            )
                                        )
                                    },
                                    outputs=["slice_output_data"],
                                )
=======
                                        "input_data":
                                        TensorConfig(data_gen=partial(
                                            generate_input1, dics))
                                    },
                                    outputs=["slice_output_data"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
=======
            self, program_config) -> (paddle_infer.Config, List[int], float):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        def generate_dynamic_shape():
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 56, 56, 192]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [8, 100, 100, 200]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [4, 56, 56, 192]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 2), 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
