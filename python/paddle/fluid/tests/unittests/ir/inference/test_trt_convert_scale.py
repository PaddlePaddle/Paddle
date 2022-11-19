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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Any, Dict, List
import unittest


class TrtConvertScaleTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch, is_int):
            if self.dims == 4:
                return np.ones([batch, 3, 24, 24]).astype(
                    np.int32 if is_int else np.float32
                )
            elif self.dims == 3:
                return np.ones([batch, 3, 24]).astype(
                    np.int32 if is_int else np.float32
                )
            elif self.dims == 2:
                return np.ones([batch, 24]).astype(
                    np.int32 if is_int else np.float32
                )
            elif self.dims == 1:
                return np.ones([24]).astype(np.int32 if is_int else np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]], is_int):
            return np.ones([1]).astype(np.int32 if is_int else np.float32)

        for num_input in [0, 1]:
            for dims in [1, 2, 3, 4]:
                for batch in [1, 2]:
                    for scale in [0.1, -1.0]:
                        for bias in [0.0, 1.2]:
                            for bias_after_scale in [False, True]:
                                for is_int in [False, True]:
                                    self.num_input = num_input
                                    self.dims = dims
                                    self.is_int = is_int
                                    dics = [
                                        {
                                            "scale": scale,
                                            "bias": bias,
                                            "bias_after_scale": bias_after_scale,
                                        },
                                        {},
                                    ]

                                    dics_intput = [
                                        {
                                            "X": ["scale_input"],
                                            "ScaleTensor": ["ScaleTensor"],
                                        },
                                        {"X": ["scale_input"]},
                                    ]
                                    dics_intputs = [
                                        {
                                            "ScaleTensor": TensorConfig(
                                                data_gen=partial(
                                                    generate_weight1,
                                                    dics,
                                                    is_int,
                                                )
                                            )
                                        },
                                        {},
                                    ]

                                    ops_config = [
                                        {
                                            "op_type": "scale",
                                            "op_inputs": dics_intput[num_input],
                                            "op_outputs": {
                                                "Out": ["scale_out"]
                                            },
                                            "op_attrs": dics[0],
                                        }
                                    ]
                                    ops = self.generate_op_config(ops_config)
                                    program_config = ProgramConfig(
                                        ops=ops,
                                        weights=dics_intputs[num_input],
                                        inputs={
                                            "scale_input": TensorConfig(
                                                data_gen=partial(
                                                    generate_input1,
                                                    dics,
                                                    batch,
                                                    is_int,
                                                )
                                            )
                                        },
                                        outputs=["scale_out"],
                                    )

                                    yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "scale_input": [1, 3, 24, 24]
                }
                self.dynamic_shape.max_input_shape = {
                    "scale_input": [4, 3, 24, 24]
                }
                self.dynamic_shape.opt_input_shape = {
                    "scale_input": [1, 3, 24, 24]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {"scale_input": [1, 3, 24]}
                self.dynamic_shape.max_input_shape = {"scale_input": [4, 3, 24]}
                self.dynamic_shape.opt_input_shape = {"scale_input": [1, 3, 24]}
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"scale_input": [1, 24]}
                self.dynamic_shape.max_input_shape = {"scale_input": [9, 48]}
                self.dynamic_shape.opt_input_shape = {"scale_input": [1, 24]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"scale_input": [24]}
                self.dynamic_shape.max_input_shape = {"scale_input": [48]}
                self.dynamic_shape.opt_input_shape = {"scale_input": [24]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

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
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.num_input == 0:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_SUPPORT,
            "INPUT ScaleTensor and Shape NOT SUPPORT",
        )

        def teller2(program_config, predictor_config):
            if self.dims == 1 and len(self.dynamic_shape.min_input_shape) == 0:
                return True
            return False

        self.add_skip_case(
            teller2,
            SkipReasons.TRT_NOT_SUPPORT,
            "INPUT DIM EQUAL TO 1 OF STATIC SHAPE NOT SUPPORT",
        )

        def teller3(program_config, predictor_config):
            if self.is_int and len(self.dynamic_shape.min_input_shape) == 0:
                return True
            return False

        self.add_skip_case(
            teller3,
            SkipReasons.TRT_NOT_SUPPORT,
            "INTEGER INPUT OF STATIC SHAPE NOT SUPPORT",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
