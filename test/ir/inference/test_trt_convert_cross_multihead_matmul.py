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

from __future__ import annotations

import unittest
from functools import partial

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import SkipReasons, TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertCrossMultiHeadMatmulTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[2] * 10 < 8520:
            return False
        return True

    def sample_program_configs(self):
        def generate_input1(batch, dim1):
            return np.random.random((batch, dim1, 320)).astype(np.float32) / 10

        def generate_input2(batch, dim2):
            return np.random.random((batch, dim2, 768)).astype(np.float32) / 10

        def generate_weight1():
            return np.random.random((320, 320)).astype(np.float32) / 10

        def generate_weight2():
            return np.random.random((768, 320)).astype(np.float32) / 10

        for batch in [1, 2]:
            self.batch = batch
            for reshape_shape in [[0, 0, 8, 40]]:
                for dim1 in [4096]:
                    for dim2 in [768]:
                        dics = [
                            {"trans_x": False, "trans_y": False},
                            {"shape": reshape_shape},
                            {"axis": [0, 2, 1, 3]},
                            {"trans_x": False, "trans_y": False},
                            {"shape": reshape_shape},
                            {"axis": [0, 2, 1, 3]},
                            {"trans_x": False, "trans_y": False},
                            {"shape": reshape_shape},
                            {"axis": [0, 2, 1, 3]},
                            {
                                "trans_x": False,
                                "trans_y": True,
                            },
                            {
                                "scale": 0.15811388194561005,
                                "bias": 0.0,
                                "bias_after_scale": True,
                            },
                            {"axis": -1, "is_test": True},
                            {"trans_x": False, "trans_y": False},
                            {"axis": [0, 2, 1, 3]},
                            {"shape": [0, 0, 320]},
                        ]

                        ops_config = [
                            {
                                "op_type": "matmul_v2",
                                "op_inputs": {
                                    "X": ["input_data1"],
                                    "Y": ["mul1_weight"],
                                },
                                "op_outputs": {"Out": ["mul1_output"]},
                                "op_attrs": dics[0],
                            },
                            {
                                "op_type": "reshape2",
                                "op_inputs": {
                                    "X": ["mul1_output"],
                                },
                                "op_outputs": {
                                    "Out": ["reshape21_output"],
                                    "XShape": ["reshape21_output_xshape"],
                                },
                                "op_attrs": dics[1],
                            },
                            {
                                "op_type": "transpose2",
                                "op_inputs": {"X": ["reshape21_output"]},
                                "op_outputs": {
                                    "Out": ["transpose21_output"],
                                    "XShape": ["transpose21_output_xshape"],
                                },
                                "op_attrs": dics[2],
                            },
                            {
                                "op_type": "matmul_v2",
                                "op_inputs": {
                                    "X": ["input_data2"],
                                    "Y": ["mul2_weight"],
                                },
                                "op_outputs": {"Out": ["mul2_output"]},
                                "op_attrs": dics[3],
                            },
                            {
                                "op_type": "reshape2",
                                "op_inputs": {"X": ["mul2_output"]},
                                "op_outputs": {
                                    "Out": ["reshape22_output"],
                                    "XShape": ["reshape22_output_xshape"],
                                },
                                "op_attrs": dics[4],
                            },
                            {
                                "op_type": "transpose2",
                                "op_inputs": {"X": ["reshape22_output"]},
                                "op_outputs": {
                                    "Out": ["transpose22_output"],
                                    "XShape": ["transpose22_output_xshape"],
                                },
                                "op_attrs": dics[5],
                            },
                            {
                                "op_type": "matmul_v2",
                                "op_inputs": {
                                    "X": ["input_data2"],
                                    "Y": ["mul3_weight"],
                                },
                                "op_outputs": {"Out": ["mul3_output"]},
                                "op_attrs": dics[6],
                            },
                            {
                                "op_type": "reshape2",
                                "op_inputs": {"X": ["mul3_output"]},
                                "op_outputs": {
                                    "Out": ["reshape23_output"],
                                    "XShape": ["reshape23_output_xshape"],
                                },
                                "op_attrs": dics[7],
                            },
                            {
                                "op_type": "transpose2",
                                "op_inputs": {"X": ["reshape23_output"]},
                                "op_outputs": {
                                    "Out": ["transpose23_output"],
                                    "XShape": ["transpose23_output_xshape"],
                                },
                                "op_attrs": dics[8],
                            },
                            {
                                "op_type": "matmul_v2",
                                "op_inputs": {
                                    "X": ["transpose21_output"],
                                    "Y": ["transpose22_output"],
                                },
                                "op_outputs": {"Out": ["matmul1_output"]},
                                "op_attrs": dics[9],
                            },
                            {
                                "op_type": "scale",
                                "op_inputs": {
                                    "X": ["matmul1_output"],
                                },
                                "op_outputs": {"Out": ["scale_output"]},
                                "op_attrs": dics[10],
                            },
                            {
                                "op_type": "softmax",
                                "op_inputs": {"X": ["scale_output"]},
                                "op_outputs": {"Out": ["softmax_output"]},
                                "op_attrs": dics[11],
                            },
                            {
                                "op_type": "matmul_v2",
                                "op_inputs": {
                                    "X": ["softmax_output"],
                                    "Y": ["transpose23_output"],
                                },
                                "op_outputs": {"Out": ["matmul2_output"]},
                                "op_attrs": dics[12],
                            },
                            {
                                "op_type": "transpose2",
                                "op_inputs": {"X": ["matmul2_output"]},
                                "op_outputs": {
                                    "Out": ["transpose24_output"],
                                    "XShape": ["transpose24_output_xshape"],
                                },
                                "op_attrs": dics[13],
                            },
                            {
                                "op_type": "reshape2",
                                "op_inputs": {"X": ["transpose24_output"]},
                                "op_outputs": {
                                    "Out": ["reshape24_output"],
                                    "XShape": ["reshape24_output_xshape"],
                                },
                                "op_attrs": dics[14],
                            },
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "mul1_weight": TensorConfig(
                                    data_gen=partial(generate_weight1)
                                ),
                                "mul2_weight": TensorConfig(
                                    data_gen=partial(generate_weight2)
                                ),
                                "mul3_weight": TensorConfig(
                                    data_gen=partial(generate_weight2)
                                ),
                            },
                            inputs={
                                "input_data1": TensorConfig(
                                    data_gen=partial(
                                        generate_input1, batch, dim1
                                    )
                                ),
                                "input_data2": TensorConfig(
                                    data_gen=partial(
                                        generate_input2, batch, dim2
                                    )
                                ),
                            },
                            outputs=["reshape24_output"],
                        )

                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        def generate_dynamic_shape(attrs):
            # The last dim of input1 and input2 should be static.
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 4096, 320],
                "input_data2": [1, 768, 768],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [8, 4096, 320],
                "input_data2": [8, 768, 768],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [2, 4096, 320],
                "input_data2": [2, 768, 768],
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
        yield self.create_inference_config(), (1, 4), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 4), (1e-2, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-2, 1e-2)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.dynamic_shape.min_input_shape == {}:
                return True
            return False

        self.add_skip_case(
            teller1,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "TThe cross attention trt oss plugin do not support static shape yet",
        )

        def teller2(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Float32:
                return True
            return False

        self.add_skip_case(
            teller2,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "The cross attention trt oss plugin do not support fp32 yet",
        )

        def teller3(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False

        self.add_skip_case(
            teller3,
            SkipReasons.TRT_NOT_IMPLEMENTED,
            "The cross attention trt oss plugin do not support int8 yet.",
        )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
