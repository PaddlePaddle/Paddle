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

from auto_scan_test import PassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestMultiheadMatmulFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        # gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["multihead_matmul", "mul"], (1e-2, 1e-3)

    def sample_program_config(self, draw):

        def generate_mul_input():
            return np.random.random([1, 128, 768]).astype(np.float32) - 0.5

        def generate_elewise_input():
            return np.random.random([1, 12, 128, 128]).astype(np.float32)

        mul_0 = OpConfig("mul",
                         inputs={
                             "X": ["mul_x"],
                             "Y": ["mul_0_w"]
                         },
                         outputs={"Out": ["mul_0_out"]},
                         x_num_col_dims=2,
                         y_num_col_dims=1)
        mul_1 = OpConfig("mul",
                         inputs={
                             "X": ["mul_x"],
                             "Y": ["mul_1_w"]
                         },
                         outputs={"Out": ["mul_1_out"]},
                         x_num_col_dims=2,
                         y_num_col_dims=1)
        mul_2 = OpConfig("mul",
                         inputs={
                             "X": ["mul_x"],
                             "Y": ["mul_2_w"]
                         },
                         outputs={"Out": ["mul_2_out"]},
                         x_num_col_dims=2,
                         y_num_col_dims=1)
        ele_0 = OpConfig("elementwise_add",
                         inputs={
                             "X": [mul_0.outputs["Out"][0]],
                             "Y": ["ele_0_w"]
                         },
                         outputs={"Out": ["ele_0_out"]},
                         axis=-1)
        ele_1 = OpConfig("elementwise_add",
                         inputs={
                             "X": [mul_1.outputs["Out"][0]],
                             "Y": ["ele_1_w"]
                         },
                         outputs={"Out": ["ele_1_out"]},
                         axis=-1)
        ele_2 = OpConfig("elementwise_add",
                         inputs={
                             "X": [mul_2.outputs["Out"][0]],
                             "Y": ["ele_2_w"]
                         },
                         outputs={"Out": ["ele_2_out"]},
                         axis=-1)
        reshape_0 = OpConfig("reshape2",
                             inputs={"X": [ele_0.outputs["Out"][0]]},
                             outputs={
                                 "Out": ["reshape_0_out"],
                                 "XShape": ["reshape_0_Xout"]
                             },
                             shape=(1, 128, 12, 64))
        reshape_1 = OpConfig("reshape2",
                             inputs={"X": [ele_1.outputs["Out"][0]]},
                             outputs={
                                 "Out": ["reshape_1_out"],
                                 "XShape": ["reshape_1_Xout"]
                             },
                             shape=(1, 128, 12, 64))
        reshape_2 = OpConfig("reshape2",
                             inputs={"X": [ele_2.outputs["Out"][0]]},
                             outputs={
                                 "Out": ["reshape_2_out"],
                                 "XShape": ["reshape_2_Xout"]
                             },
                             shape=(1, 128, 12, 64))
        transpose_0 = OpConfig("transpose2",
                               inputs={"X": [reshape_0.outputs["Out"][0]]},
                               outputs={"Out": ["transpose_0_out"]},
                               axis=(0, 2, 1, 3))
        transpose_1 = OpConfig("transpose2",
                               inputs={"X": [reshape_1.outputs["Out"][0]]},
                               outputs={"Out": ["transpose_1_out"]},
                               axis=(0, 2, 3, 1))
        transpose_2 = OpConfig("transpose2",
                               inputs={"X": [reshape_2.outputs["Out"][0]]},
                               outputs={"Out": ["transpose_2_out"]},
                               axis=(0, 2, 1, 3))
        matmul_0 = OpConfig("matmul",
                            inputs={
                                "X": [transpose_0.outputs["Out"][0]],
                                "Y": [transpose_1.outputs["Out"][0]]
                            },
                            outputs={"Out": ["matmul_0_out"]},
                            alpha=0.125,
                            transpose_X=False,
                            transpose_Y=False,
                            fused_reshape_Out=[],
                            fused_reshape_X=[],
                            fused_reshape_Y=[],
                            fused_transpose_Out=[],
                            fused_transpose_X=[],
                            fused_transpose_Y=[])
        ele_3 = OpConfig("elementwise_add",
                         inputs={
                             "X": [matmul_0.outputs["Out"][0]],
                             "Y": ["eltadd_qk_b_var"]
                         },
                         outputs={"Out": ["ele_3_out"]},
                         axis=-1)
        softmax_op = OpConfig("softmax",
                              inputs={"X": [ele_3.outputs["Out"][0]]},
                              outputs={"Out": ["softmax_out"]},
                              axis=3,
                              is_test=True)
        matmul_1 = OpConfig("matmul",
                            inputs={
                                "X": [softmax_op.outputs["Out"][0]],
                                "Y": [transpose_2.outputs["Out"][0]]
                            },
                            outputs={"Out": ["matmul_1_out"]},
                            alpha=1.0,
                            transpose_X=False,
                            transpose_Y=False,
                            fused_reshape_Out=[],
                            fused_reshape_X=[],
                            fused_reshape_Y=[],
                            fused_transpose_Out=[],
                            fused_transpose_X=[],
                            fused_transpose_Y=[])
        transpose_3 = OpConfig("transpose2",
                               inputs={"X": [matmul_1.outputs["Out"][0]]},
                               outputs={"Out": ["transpose_3_out"]},
                               axis=(0, 2, 1, 3))
        reshape_3 = OpConfig("reshape2",
                             inputs={"X": [transpose_3.outputs["Out"][0]]},
                             outputs={
                                 "Out": ["reshape_3_out"],
                                 "XShape": ["reshape_3_Xout"]
                             },
                             shape=(1, 128, 768))
        mul_3 = OpConfig("mul",
                         inputs={
                             "X": [reshape_3.outputs["Out"][0]],
                             "Y": ["mul_3_w"]
                         },
                         outputs={"Out": ["mul_3_out"]},
                         x_num_col_dims=2,
                         y_num_col_dims=1)
        ops = [
            mul_0, mul_1, mul_2, ele_0, ele_1, ele_2, reshape_0, reshape_1,
            reshape_2, transpose_0, transpose_1, transpose_2, matmul_0, ele_3,
            softmax_op, matmul_1, transpose_3, reshape_3, mul_3
        ]
        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "mul_x":
                TensorConfig(data_gen=partial(generate_mul_input)),
                "eltadd_qk_b_var":
                TensorConfig(data_gen=partial(generate_elewise_input))
            },
            weights={
                "mul_0_w": TensorConfig(shape=[768, 768]),
                "mul_1_w": TensorConfig(shape=[768, 768]),
                "mul_2_w": TensorConfig(shape=[768, 768]),
                "mul_3_w": TensorConfig(shape=[768, 768]),
                "ele_0_w": TensorConfig(shape=[768]),
                "ele_1_w": TensorConfig(shape=[768]),
                "ele_2_w": TensorConfig(shape=[768])
            },
            outputs=[ops[-1].outputs["Out"][0]])
        return program_config

    def test(self):
        self.run_and_statis(quant=False,
                            max_examples=100,
                            min_success_num=1,
                            passes=["multihead_matmul_fuse_pass_v3"])


if __name__ == "__main__":
    unittest.main()
