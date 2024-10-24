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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertQkAttentionTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8520:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1(batch, length):
            return np.random.rand(batch, length, 256).astype(np.float32) / 10

        def generate_input2(batch, length):
            return np.random.rand(batch, length, 256).astype(np.float32) / 10

        def generate_weight_q():
            return np.random.rand(256, 256).astype(np.float32) / 10

        def generate_weight_k():
            return np.random.rand(256, 256).astype(np.float32) / 10

        def generate_weight_v():
            return np.random.rand(256, 256).astype(np.float32) / 10

        def generate_bias_q():
            return np.random.rand(256).astype(np.float32) / 10

        def generate_bias_k():
            return np.random.rand(256).astype(np.float32) / 10

        def generate_bias_v():
            return np.random.rand(256).astype(np.float32) / 10

        for batch in [1, 2]:
            self.batch = batch
            for length in [300, 400]:
                ops_config = [
                    # q
                    {
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["input_data1"],
                            "Y": ["matmul_q_weight"],
                        },
                        "op_outputs": {"Out": ["matmul_q_output"]},
                        "op_attrs": {"trans_x": False, "trans_y": False},
                    },
                    {
                        "op_type": "elementwise_add",
                        "op_inputs": {
                            "X": ["matmul_q_output"],
                            "Y": ["bias_q"],
                        },
                        "op_outputs": {"Out": ["elementwise_q_output"]},
                        "op_attrs": {
                            "Scale_out": 1.0,
                            "Scale_x": 1.0,
                            "Scale_y": 1.0,
                            "axis": 2,
                        },
                    },
                    {
                        "op_type": "reshape2",
                        "op_inputs": {
                            "X": ["elementwise_q_output"],
                        },
                        "op_outputs": {
                            "Out": ["reshape_q_output"],
                            "XShape": ["reshape_q_output_xshape"],
                        },
                        "op_attrs": {"shape": [0, 0, 8, 32]},
                    },
                    {
                        "op_type": "transpose2",
                        "op_inputs": {"X": ["reshape_q_output"]},
                        "op_outputs": {
                            "Out": ["transpose_q_output"],
                            "XShape": ["transpose_q_output_xshape"],
                        },
                        "op_attrs": {
                            "axis": [0, 2, 1, 3],
                            "data_format": "AnyLayout",
                        },
                    },
                    # k
                    {
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["input_data1"],
                            "Y": ["matmul_k_weight"],
                        },
                        "op_outputs": {"Out": ["matmul_k_output"]},
                        "op_attrs": {"trans_x": False, "trans_y": False},
                    },
                    {
                        "op_type": "elementwise_add",
                        "op_inputs": {
                            "X": ["matmul_k_output"],
                            "Y": ["bias_k"],
                        },
                        "op_outputs": {"Out": ["elementwise_k_output"]},
                        "op_attrs": {
                            "Scale_out": 1.0,
                            "Scale_x": 1.0,
                            "Scale_y": 1.0,
                            "axis": 2,
                        },
                    },
                    {
                        "op_type": "reshape2",
                        "op_inputs": {
                            "X": ["elementwise_k_output"],
                        },
                        "op_outputs": {
                            "Out": ["reshape_k_output"],
                            "XShape": ["reshape_k_output_xshape"],
                        },
                        "op_attrs": {"shape": [0, 0, 8, 32]},
                    },
                    {
                        "op_type": "transpose2",
                        "op_inputs": {"X": ["reshape_k_output"]},
                        "op_outputs": {
                            "Out": ["transpose_k_output"],
                            "XShape": ["transpose_k_output_xshape"],
                        },
                        "op_attrs": {
                            "axis": [0, 2, 1, 3],
                            "data_format": "AnyLayout",
                        },
                    },
                    # V
                    {
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["input_data2"],
                            "Y": ["matmul_v_weight"],
                        },
                        "op_outputs": {"Out": ["matmul_v_output"]},
                        "op_attrs": {"trans_x": False, "trans_y": False},
                    },
                    {
                        "op_type": "elementwise_add",
                        "op_inputs": {
                            "X": ["matmul_v_output"],
                            "Y": ["bias_v"],
                        },
                        "op_outputs": {"Out": ["elementwise_v_output"]},
                        "op_attrs": {
                            "Scale_out": 1.0,
                            "Scale_x": 1.0,
                            "Scale_y": 1.0,
                            "axis": 2,
                        },
                    },
                    {
                        "op_type": "reshape2",
                        "op_inputs": {
                            "X": ["elementwise_v_output"],
                        },
                        "op_outputs": {
                            "Out": ["reshape_v_output"],
                            "XShape": ["reshape_v_output_xshape"],
                        },
                        "op_attrs": {"shape": [0, 0, 8, 32]},
                    },
                    {
                        "op_type": "transpose2",
                        "op_inputs": {"X": ["reshape_v_output"]},
                        "op_outputs": {
                            "Out": ["transpose_v_output"],
                            "XShape": ["transpose_v_output_xshape"],
                        },
                        "op_attrs": {
                            "axis": [0, 2, 1, 3],
                            "data_format": "AnyLayout",
                        },
                    },
                    # matmul1+matmul2
                    {
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["transpose_q_output"],
                            "Y": ["transpose_k_output"],
                        },
                        "op_outputs": {"Out": ["matmul1_output"]},
                        "op_attrs": {"trans_x": False, "trans_y": True},
                    },
                    {
                        "op_type": "scale",
                        "op_inputs": {
                            "X": ["matmul1_output"],
                        },
                        "op_outputs": {"Out": ["scale_output"]},
                        "op_attrs": {
                            "scale": 0.17677,
                            "bias": 0.0,
                            "bias_after_scale": True,
                        },
                    },
                    {
                        "op_type": "softmax",
                        "op_inputs": {"X": ["scale_output"]},
                        "op_outputs": {"Out": ["softmax_output"]},
                        "op_attrs": {
                            "axis": -1,
                            "data_format": "AnyLayout",
                        },
                    },
                    {
                        "op_type": "matmul_v2",
                        "op_inputs": {
                            "X": ["softmax_output"],
                            "Y": ["transpose_v_output"],
                        },
                        "op_outputs": {"Out": ["matmul2_output"]},
                        "op_attrs": {"trans_x": False, "trans_y": False},
                    },
                    {
                        "op_type": "transpose2",
                        "op_inputs": {"X": ["matmul2_output"]},
                        "op_outputs": {
                            "Out": ["transpose_output"],
                            "XShape": ["transpose_output_xshape"],
                        },
                        "op_attrs": {
                            "axis": [0, 2, 1, 3],
                            "data_format": "AnyLayout",
                        },
                    },
                    {
                        "op_type": "reshape2",
                        "op_inputs": {"X": ["transpose_output"]},
                        "op_outputs": {
                            "Out": ["reshape_output"],
                            "XShape": ["reshape_output_xshape"],
                        },
                        "op_attrs": {"shape": [0, 0, 256]},
                    },
                ]
                ops = self.generate_op_config(ops_config)

                program_config = ProgramConfig(
                    ops=ops,
                    weights={
                        "matmul_q_weight": TensorConfig(
                            data_gen=partial(generate_weight_q)
                        ),
                        "matmul_k_weight": TensorConfig(
                            data_gen=partial(generate_weight_k)
                        ),
                        "matmul_v_weight": TensorConfig(
                            data_gen=partial(generate_weight_v)
                        ),
                        "bias_q": TensorConfig(
                            data_gen=partial(generate_bias_q)
                        ),
                        "bias_k": TensorConfig(
                            data_gen=partial(generate_bias_k)
                        ),
                        "bias_v": TensorConfig(
                            data_gen=partial(generate_bias_v)
                        ),
                    },
                    inputs={
                        "input_data1": TensorConfig(
                            data_gen=partial(generate_input1, batch, length)
                        ),
                        "input_data2": TensorConfig(
                            data_gen=partial(generate_input2, batch, length)
                        ),
                    },
                    outputs=["reshape_output"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            # The last dim of input1 and input2 should be static.
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 300, 256],
                "input_data2": [1, 300, 256],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [4, 1200, 256],
                "input_data2": [4, 1200, 256],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [1, 300, 256],
                "input_data2": [1, 300, 256],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-2, 1e-3)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.dynamic_shape.min_input_shape == {}:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "The qk attention trt oss plugin do not support static shape yet",
        )

        def teller2(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Float32:
                return True
            return False

        self.add_skip_case(
            teller2,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "The qk attention trt oss plugin do not support fp32 yet",
        )

        def teller3(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False

        self.add_skip_case(
            teller3,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "The qk attention trt oss plugin do not support int8 yet.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
