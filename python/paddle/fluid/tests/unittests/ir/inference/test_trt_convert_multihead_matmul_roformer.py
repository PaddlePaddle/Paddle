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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import List


class TrtConvertMultiHeadMatmulRoformerTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(batch, dim1):
            return (
                np.random.random((batch, dim1, 768)).astype(np.float32) - 0.5
            ) / 100.0

        def generate_input2(shape):
            return (np.random.random(shape).astype(np.float32) - 0.5) / 100.0

        def generate_cos_input(batch, dim1):
            return (
                np.random.random((batch, 12, dim1, 64)).astype(np.float32) - 0.5
            )

        def generate_sin_input(batch, dim1):
            return (
                np.random.random((batch, 12, dim1, 64)).astype(np.float32) - 0.5
            )

        def generate_weight1():
            return (
                np.random.random((768, 768)).astype(np.float32) - 0.5
            ) / 100.0

        def generate_weight2():
            return (np.random.random(768).astype(np.float32) - 0.5) / 100.0

        for batch in [1, 2, 4]:
            self.batch = batch
            for reshape_shape in [[0, 0, 12, 64]]:
                for dim1 in [128]:
                    input2_shapes = [
                        (batch, reshape_shape[2], dim1, dim1)
                    ]  # 10,12,128,128
                    # [batch, 1, 1, dim1]]
                    for input2_shape in input2_shapes:
                        for axis in [0]:
                            dics = [
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": False,
                                },
                                {"axis": 2},
                                {"shape": reshape_shape},
                                {"axis": [0, 2, 1, 3]},
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": False,
                                },
                                {"axis": 2},
                                {"shape": reshape_shape},
                                {"axis": [0, 2, 1, 3]},
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": False,
                                },
                                {"axis": 2},
                                {"shape": reshape_shape},
                                {"axis": [0, 2, 1, 3]},
                                {
                                    "scale": 0.125,
                                    "bias": 0.0,
                                    "bias_after_scale": True,
                                },
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": True,
                                    "fused_reshape_X": [],
                                    "fused_reshape_Y": [],
                                    "fused_transpose_X": [],
                                    "fused_transpose_Y": [],
                                    "fused_reshape_Out": [],
                                    "fused_transpose_Out": [],
                                },
                                {"axis": axis},
                                {"axis": -1, "is_test": True},
                                {
                                    "seed": 0,
                                    "dropout_prob": 0.10000000149011612,
                                    "dropout_implementation": "upscale_in_train",
                                    "fix_seed": False,
                                    "is_test": True,
                                },
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": False,
                                    "fused_reshape_X": [],
                                    "fused_reshape_Y": [],
                                    "fused_transpose_X": [],
                                    "fused_transpose_Y": [],
                                    "fused_reshape_Out": [],
                                    "fused_transpose_Out": [],
                                },
                                {"axis": [0, 2, 1, 3]},
                                {"shape": [0, 0, 768]},
                                {
                                    "alpha": 1.0,
                                    "transpose_X": False,
                                    "transpose_Y": False,
                                },
                            ]

                            ops_config = [
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul1_weight"],
                                    },
                                    "op_outputs": {"Out": ["mul1_output"]},
                                    "op_attrs": dics[0],
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul1_output"],
                                        "Y": ["elementwise_add1_weight"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add1_output"]
                                    },
                                    "op_attrs": dics[1],
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add1_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape21_output"],
                                        "XShape": ["reshape21_output_xshape"],
                                    },
                                    "op_attrs": dics[2],
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {"X": ["reshape21_output"]},
                                    "op_outputs": {
                                        "Out": ["transpose21_output"],
                                        "XShape": ["transpose21_output_xshape"],
                                    },
                                    "op_attrs": dics[3],
                                },
                                # roformer part
                                # q with scale branch
                                {
                                    "op_type": "elementwise_mul",
                                    "op_inputs": {
                                        "X": ["transpose21_output"],
                                        "Y": ["cos_input"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_mul_q_0_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "split",
                                    "op_inputs": {
                                        "X": ["transpose21_output"],
                                    },
                                    "op_outputs": {
                                        "Out": [
                                            "split_q_0_output_0",
                                            "split_q_0_output_1",
                                        ],
                                    },
                                    "op_attrs": {
                                        "axis": 3,
                                        "num": 2,
                                    },
                                },
                                {
                                    "op_type": "concat",
                                    "op_inputs": {
                                        "X": [
                                            "split_q_0_output_1",
                                            "split_q_0_output_0",
                                        ],
                                    },
                                    "op_outputs": {
                                        "Out": ["concat_q_0_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "elementwise_mul",
                                    "op_inputs": {
                                        "X": ["concat_q_0_output"],
                                        "Y": ["sin_input"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_mul_q_1_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["elementwise_mul_q_0_output"],
                                        "Y": ["elementwise_mul_q_1_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add_q_0_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "scale",
                                    "op_inputs": {
                                        "X": ["elementwise_add_q_0_output"],
                                    },
                                    "op_outputs": {"Out": ["scale_output"]},
                                    "op_attrs": dics[12],
                                },
                                # k branch
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul2_weight"],
                                    },
                                    "op_outputs": {"Out": ["mul2_output"]},
                                    "op_attrs": dics[4],
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul2_output"],
                                        "Y": ["elementwise_add2_weight"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add2_output"]
                                    },
                                    "op_attrs": dics[5],
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add2_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape22_output"],
                                        "XShape": ["reshape22_output_xshape"],
                                    },
                                    "op_attrs": dics[6],
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {"X": ["reshape22_output"]},
                                    "op_outputs": {
                                        "Out": ["transpose22_output"],
                                        "XShape": ["transpose22_output_xshape"],
                                    },
                                    "op_attrs": dics[7],
                                },
                                # roformer part
                                # k without scale branch
                                {
                                    "op_type": "elementwise_mul",
                                    "op_inputs": {
                                        "X": ["transpose22_output"],
                                        "Y": ["cos_input"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_mul_k_0_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "split",
                                    "op_inputs": {
                                        "X": ["transpose22_output"],
                                    },
                                    "op_outputs": {
                                        "Out": [
                                            "split_k_0_output_0",
                                            "split_k_0_output_1",
                                        ]
                                    },
                                    "op_attrs": {"axis": 3, "num": 2},
                                },
                                {
                                    "op_type": "concat",
                                    "op_inputs": {
                                        "X": [
                                            "split_k_0_output_1",
                                            "split_k_0_output_0",
                                        ]
                                    },
                                    "op_outputs": {
                                        "Out": ["concat_k_0_output"]
                                    },
                                    "op_attrs": {
                                        "axis": -1,
                                    },
                                },
                                {
                                    "op_type": "elementwise_mul",
                                    "op_inputs": {
                                        "X": ["concat_k_0_output"],
                                        "Y": ["sin_input"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_mul_k_1_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["elementwise_mul_k_0_output"],
                                        "Y": ["elementwise_mul_k_1_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add_k_0_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                # v branch
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul3_weight"],
                                    },
                                    "op_outputs": {"Out": ["mul3_output"]},
                                    "op_attrs": dics[8],
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul3_output"],
                                        "Y": ["elementwise_add3_weight"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add3_output"]
                                    },
                                    "op_attrs": dics[9],
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add3_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape23_output"],
                                        "XShape": ["reshape23_output_xshape"],
                                    },
                                    "op_attrs": dics[10],
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {"X": ["reshape23_output"]},
                                    "op_outputs": {
                                        "Out": ["transpose23_output"],
                                        "XShape": ["transpose23_output_xshape"],
                                    },
                                    "op_attrs": dics[11],
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["scale_output"],
                                        "Y": ["elementwise_add_k_0_output"],
                                    },
                                    "op_outputs": {"Out": ["matmul1_output"]},
                                    "op_attrs": dics[13],
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["matmul1_output"],
                                        "Y": ["input_data2"],
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add4_output"]
                                    },
                                    "op_attrs": {"axis": -1},
                                },
                                {
                                    "op_type": "softmax",
                                    "op_inputs": {
                                        "X": ["elementwise_add4_output"]
                                    },
                                    "op_outputs": {"Out": ["softmax_output"]},
                                    "op_attrs": dics[15],
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["softmax_output"],
                                        "Y": ["transpose23_output"],
                                    },
                                    "op_outputs": {"Out": ["matmul2_output"]},
                                    "op_attrs": dics[17],
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {"X": ["matmul2_output"]},
                                    "op_outputs": {
                                        "Out": ["transpose24_output"],
                                        "XShape": ["transpose24_output_xshape"],
                                    },
                                    "op_attrs": dics[18],
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {"X": ["transpose24_output"]},
                                    "op_outputs": {
                                        "Out": ["reshape24_output"],
                                        "XShape": ["reshape24_output_xshape"],
                                    },
                                    "op_attrs": dics[19],
                                },
                                # In order to fuse ops with
                                # multihead_matmul_fuse_pass_v2, the last op
                                # must be mul.
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["reshape24_output"],
                                        "Y": ["mul4_weight"],
                                    },
                                    "op_outputs": {"Out": ["mul4_output"]},
                                    "op_attrs": dics[20],
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
                                        data_gen=partial(generate_weight1)
                                    ),
                                    "mul3_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)
                                    ),
                                    "mul4_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)
                                    ),
                                    "elementwise_add1_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)
                                    ),
                                    "elementwise_add2_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)
                                    ),
                                    "elementwise_add3_weight": TensorConfig(
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
                                            generate_input2, input2_shape
                                        )
                                    ),
                                    "cos_input": TensorConfig(
                                        data_gen=partial(
                                            generate_cos_input, batch, dim1
                                        )
                                    ),
                                    "sin_input": TensorConfig(
                                        data_gen=partial(
                                            generate_sin_input, batch, dim1
                                        )
                                    ),
                                },
                                outputs=["mul4_output"],
                            )

                            yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The last dim of input1 and input2 should be static.
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 1, 768],
                "input_data2": [1, 12, 1, 1],
                "cos_input": [1, 12, 1, 64],
                "sin_input": [1, 12, 1, 64],
                "reshape24_output": [1, 1, 768],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [10, 128, 768],
                "input_data2": [10, 12, 128, 128],
                "cos_input": [10, 12, 128, 64],
                "sin_input": [10, 12, 128, 64],
                "reshape24_output": [10, 128, 768],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [8, 128, 768],
                "input_data2": [8, 12, 128, 128],
                "cos_input": [8, 12, 128, 64],
                "sin_input": [8, 12, 128, 64],
                "reshape24_output": [8, 128, 768],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 2013265920
        yield self.create_inference_config(), (1, 5), (1e-3, 1e-3)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 5), (1e-3, 1e-3)

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
