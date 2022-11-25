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
from typing import List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertSumTest(TrtLayerAutoScanTest):
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertSumTest(TrtLayerAutoScanTest):

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        def generate_input1(batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input2(batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        def generate_input3(batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        for dims in [1, 2, 3, 4]:
            for batch in [1, 4]:
                self.dims = dims
<<<<<<< HEAD
                ops_config = [
                    {
                        "op_type": "sum",
                        "op_inputs": {"X": ["input1", "input2", "input3"]},
                        "op_outputs": {"Out": ["output"]},
                        "op_attrs": {},
                    }
                ]
=======
                ops_config = [{
                    "op_type": "sum",
                    "op_inputs": {
                        "X": ["input1", "input2", "input3"]
                    },
                    "op_outputs": {
                        "Out": ["output"]
                    },
                    "op_attrs": {}
                }]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
<<<<<<< HEAD
                        "input1": TensorConfig(
                            data_gen=partial(generate_input1, batch)
                        ),
                        "input2": TensorConfig(
                            data_gen=partial(generate_input2, batch)
                        ),
                        "input3": TensorConfig(
                            data_gen=partial(generate_input3, batch)
                        ),
                    },
                    outputs=["output"],
                )
=======
                        "input1":
                        TensorConfig(data_gen=partial(generate_input1, batch)),
                        "input2":
                        TensorConfig(data_gen=partial(generate_input2, batch)),
                        "input3":
                        TensorConfig(data_gen=partial(generate_input3, batch))
                    },
                    outputs=["output"])
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
=======
            self, program_config) -> (paddle_infer.Config, List[int], float):

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        def generate_dynamic_shape():
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "input1": [1, 3, 24, 24],
                    "input2": [1, 3, 24, 24],
<<<<<<< HEAD
                    "input3": [1, 3, 24, 24],
=======
                    "input3": [1, 3, 24, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [4, 3, 48, 48],
                    "input2": [4, 3, 48, 48],
<<<<<<< HEAD
                    "input3": [4, 3, 48, 48],
=======
                    "input3": [4, 3, 48, 48]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [1, 3, 24, 24],
                    "input2": [1, 3, 24, 24],
<<<<<<< HEAD
                    "input3": [1, 3, 24, 24],
=======
                    "input3": [1, 3, 24, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "input1": [1, 3, 24],
                    "input2": [1, 3, 24],
<<<<<<< HEAD
                    "input3": [1, 3, 24],
=======
                    "input3": [1, 3, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [4, 3, 48],
                    "input2": [4, 3, 48],
<<<<<<< HEAD
                    "input3": [4, 3, 48],
=======
                    "input3": [4, 3, 48]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [1, 3, 24],
                    "input2": [1, 3, 24],
<<<<<<< HEAD
                    "input3": [1, 3, 24],
=======
                    "input3": [1, 3, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input1": [1, 24],
                    "input2": [1, 24],
<<<<<<< HEAD
                    "input3": [1, 24],
=======
                    "input3": [1, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [4, 48],
                    "input2": [4, 48],
<<<<<<< HEAD
                    "input3": [4, 48],
=======
                    "input3": [4, 48]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [1, 24],
                    "input2": [1, 24],
<<<<<<< HEAD
                    "input3": [1, 24],
=======
                    "input3": [1, 24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "input1": [24],
                    "input2": [24],
<<<<<<< HEAD
                    "input3": [24],
=======
                    "input3": [24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [48],
                    "input2": [48],
<<<<<<< HEAD
                    "input3": [48],
=======
                    "input3": [48]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [24],
                    "input2": [24],
<<<<<<< HEAD
                    "input3": [24],
=======
                    "input3": [24]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(dynamic_shape):
<<<<<<< HEAD
            if self.dims == 1 and not dynamic_shape:
=======
            if (self.dims == 1 and not dynamic_shape):
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                return 0, 5
            return 1, 4

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            False
        ), 1e-3
=======
            False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            False), 1e-5
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
<<<<<<< HEAD
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-3
=======
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

    def test(self):
        self.run_test()


# special case when sum having olny one input
class TrtConvertSumTest1(TrtLayerAutoScanTest):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
<<<<<<< HEAD
=======

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        def generate_input1(batch):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([24]).astype(np.float32)

        for dims in [1, 2, 3, 4]:
            for batch in [1, 4]:
                self.dims = dims
<<<<<<< HEAD
                ops_config = [
                    {
                        "op_type": "sum",
                        "op_inputs": {"X": ["input1"]},
                        "op_outputs": {"Out": ["output"]},
                        "op_attrs": {},
                    }
                ]
=======
                ops_config = [{
                    "op_type": "sum",
                    "op_inputs": {
                        "X": ["input1"]
                    },
                    "op_outputs": {
                        "Out": ["output"]
                    },
                    "op_attrs": {}
                }]
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
<<<<<<< HEAD
                        "input1": TensorConfig(
                            data_gen=partial(generate_input1, batch)
                        ),
                    },
                    outputs=["output"],
                )
=======
                        "input1":
                        TensorConfig(data_gen=partial(generate_input1, batch)),
                    },
                    outputs=["output"])
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

                yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
=======
            self, program_config) -> (paddle_infer.Config, List[int], float):

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        def generate_dynamic_shape():
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {"input1": [1, 3, 24, 24]}
                self.dynamic_shape.max_input_shape = {"input1": [4, 3, 48, 48]}
                self.dynamic_shape.opt_input_shape = {"input1": [1, 3, 24, 24]}
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"input1": [1, 3, 24]}
                self.dynamic_shape.max_input_shape = {"input1": [4, 3, 48]}
                self.dynamic_shape.opt_input_shape = {"input1": [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {
                    "input1": [1, 24],
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [4, 48],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [1, 24],
                }
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {
                    "input1": [24],
                }
                self.dynamic_shape.max_input_shape = {
                    "input1": [48],
                }
                self.dynamic_shape.opt_input_shape = {
                    "input1": [24],
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(dynamic_shape):
<<<<<<< HEAD
            if self.dims == 1 and not dynamic_shape:
=======
            if (self.dims == 1 and not dynamic_shape):
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
                return 0, 3
            return 1, 2

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            False
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            False
        ), 1e-3
=======
            False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            False), 1e-5
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

        # for dynamic_shape
        generate_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
<<<<<<< HEAD
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-3
=======
        yield self.create_inference_config(), generate_trt_nodes_num(True), 1e-5
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
