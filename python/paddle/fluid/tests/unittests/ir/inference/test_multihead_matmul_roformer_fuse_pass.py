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


class TestMultiheadMatmulRoformerFusePass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        # gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["multihead_matmul_roformer", "mul"], (1e-2, 1e-3)

    def sample_program_config(self, draw):

        def generate_mul_input():
            return np.random.random([1, 128, 768]).astype(np.float32) - 0.5

        def generate_elewise_input():
            return np.random.random([1, 12, 128, 128]).astype(np.float32)

        def generate_cos_input():
            return np.random.random([1, 12, 128, 64]).astype(np.float32) - 0.5

        def generate_sin_input():
            return np.random.random([1, 12, 128, 64]).astype(np.float32) - 0.5

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
                               axis=(0, 2, 1, 3))
        transpose_2 = OpConfig("transpose2",
                               inputs={"X": [reshape_2.outputs["Out"][0]]},
                               outputs={"Out": ["transpose_2_out"]},
                               axis=(0, 2, 1, 3))

        # roformer part
        # q without scale branch
        ele_mul_q_0 = OpConfig(
            "elementwise_mul",  # without split && concat
            inputs={
                "X": [transpose_0.outputs["Out"][0]],
                "Y": ["sin_input"]
            },
            outputs={"Out": ["ele_mul_q_0_out"]},
            axis=-1)

        split_q_0 = OpConfig(
            "split",
            inputs={"X": [ele_mul_q_0.outputs["Out"][0]]},
            outputs={"Out": ["split_q_0_out_0", "split_q_0_out_1"]},
            axis=3,
            num=2)

        concat_q_0 = OpConfig(
            "concat",
            inputs={
                "X": [split_q_0.outputs["Out"][1], split_q_0.outputs["Out"][0]]
            },
            outputs={"Out": ["concat_q_0_out"]},
            axis=-1)

        ele_mul_q_1 = OpConfig(
            "elementwise_mul",  # with split && concat
            inputs={
                "X": [concat_q_0.outputs["Out"][0]],
                "Y": ["cos_input"]
            },
            outputs={"Out": ["ele_mul_q_1_out"]},
            axis=-1)

        ele_add_q_0 = OpConfig("elementwise_add",
                               inputs={
                                   "X": [ele_mul_q_0.outputs["Out"][0]],
                                   "Y": [ele_mul_q_1.outputs["Out"][0]],
                               },
                               outputs={"Out": ["ele_add_q_0_out"]},
                               axis=-1)

        #k branch which with scale op
        ele_mul_k_0 = OpConfig(
            "elementwise_mul",  # without split && concat
            inputs={
                "X": [transpose_1.outputs["Out"][0]],
                "Y": ["sin_input"]
            },
            outputs={"Out": ["ele_mul_k_0_out"]},
            axis=-1)

        split_k_0 = OpConfig(
            "split",
            inputs={"X": [ele_mul_k_0.outputs["Out"][0]]},
            outputs={"Out": ["split_k_0_out_0", "split_k_0_out_1"]},
            axis=3,
            num=2)

        concat_k_0 = OpConfig(
            "concat",
            inputs={
                "X": [split_k_0.outputs["Out"][1], split_k_0.outputs["Out"][0]]
            },
            outputs={"Out": ["concat_k_0_out"]},
            axis=-1)

        ele_mul_k_1 = OpConfig(
            "elementwise_mul",  # with split && concat
            inputs={
                "X": [concat_k_0.outputs["Out"][0]],
                "Y": ["cos_input"]
            },
            outputs={"Out": ["ele_mul_k_1_out"]},
            axis=-1)

        ele_add_k_0 = OpConfig("elementwise_add",
                               inputs={
                                   "X": [ele_mul_k_0.outputs["Out"][0]],
                                   "Y": [ele_mul_k_1.outputs["Out"][0]],
                               },
                               outputs={"Out": ["ele_add_k_0_out"]},
                               axis=-1)
        scale_0 = OpConfig("scale",
                           inputs={"X": [ele_add_k_0.outputs["Out"][0]]},
                           outputs={"Out": ["scale_0_out"]},
                           scale=0.1961161345243454,
                           bias=0)

        matmul_0 = OpConfig("matmul",
                            inputs={
                                "X": [scale_0.outputs["Out"][0]],
                                "Y": [ele_add_q_0.outputs["Out"][0]]
                            },
                            outputs={"Out": ["matmul_0_out"]},
                            alpha=1.0,
                            transpose_X=False,
                            transpose_Y=True,
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
            mul_0,
            mul_1,
            mul_2,
            ele_0,
            ele_1,
            ele_2,
            reshape_0,
            reshape_1,
            reshape_2,
            transpose_0,
            transpose_1,
            transpose_2,
            ele_mul_q_0,
            split_q_0,
            concat_q_0,
            ele_mul_q_1,
            ele_add_q_0,
            ele_mul_k_0,
            split_k_0,
            concat_k_0,
            ele_mul_k_1,
            ele_add_k_0,
            scale_0,
            matmul_0,
            ele_3,
            softmax_op,
            matmul_1,
            transpose_3,
            reshape_3,
            mul_3,
        ]
        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "mul_x":
                TensorConfig(data_gen=partial(generate_mul_input)),
                "eltadd_qk_b_var":
                TensorConfig(data_gen=partial(generate_elewise_input)),
                "cos_input":
                TensorConfig(data_gen=partial(generate_cos_input)),
                "sin_input":
                TensorConfig(data_gen=partial(generate_sin_input))
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
                            passes=["multihead_matmul_roformer_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
