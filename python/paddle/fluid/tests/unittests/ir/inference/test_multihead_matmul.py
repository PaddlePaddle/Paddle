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
        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=16,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        config.set_trt_dynamic_shape_info({
            "mul_x": [1, 1, 768],
            "eltadd_qk_b_var": [1, 12, 1, 1]
        }, {"mul_x": [16, 128, 768],
            "eltadd_qk_b_var": [16, 12, 128, 128]}, {
                "mul_x": [8, 128, 768],
                "eltadd_qk_b_var": [8, 12, 128, 128]
            })
        yield config, ['multihead_matmul'], (1e-1, 1e-5)

    def add_ignore_pass_case(self):
        # Here we put some skip rules to avoid known bugs
        def teller1(program_config, predictor_config):
            return False

        self.add_ignore_check_case(
            teller1,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "...", )

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        def generate_mul_input():
            return np.random.random([16, 128, 768]).astype(np.float32)

        def generate_elewise_input():
            return np.random.random([16, 12, 128, 128]).astype(np.float32)

        mul_0 = OpConfig(
            "mul",
            inputs={"X": ["mul_x"],
                    "Y": ["mul_0_w"]},
            outputs={"Out": ["mul_0_out"]},
            x_num_col_dims=2,
            y_num_col_dims=1)
        mul_1 = OpConfig(
            "mul",
            inputs={"X": ["mul_x"],
                    "Y": ["mul_1_w"]},
            outputs={"Out": ["mul_1_out"]},
            x_num_col_dims=2,
            y_num_col_dims=1)
        mul_2 = OpConfig(
            "mul",
            inputs={"X": ["mul_x"],
                    "Y": ["mul_2_w"]},
            outputs={"Out": ["mul_2_out"]},
            x_num_col_dims=2,
            y_num_col_dims=1)
        ele_0 = OpConfig(
            "elementwise_add",
            inputs={"X": [mul_0.outputs["Out"][0]],
                    "Y": ["ele_0_w"]},
            outputs={"Out": ["ele_0_out"]},
            axis=2)
        ele_1 = OpConfig(
            "elementwise_add",
            inputs={"X": [mul_1.outputs["Out"][0]],
                    "Y": ["ele_1_w"]},
            outputs={"Out": ["ele_1_out"]},
            axis=2)
        ele_2 = OpConfig(
            "elementwise_add",
            inputs={"X": [mul_2.outputs["Out"][0]],
                    "Y": ["ele_2_w"]},
            outputs={"Out": ["ele_2_out"]},
            axis=2)
        reshape_0 = OpConfig(
            "reshape2",
            inputs={"X": [ele_0.outputs["Out"][0]]},
            outputs={"Out": ["reshape_0_out"],
                     "XShape": ["reshape_0_Xout"]},
            shape=(0, 0, 12, 64))
        reshape_1 = OpConfig(
            "reshape2",
            inputs={"X": [ele_1.outputs["Out"][0]]},
            outputs={"Out": ["reshape_1_out"],
                     "XShape": ["reshape_1_Xout"]},
            shape=(0, 0, 12, 64))
        reshape_2 = OpConfig(
            "reshape2",
            inputs={"X": [ele_2.outputs["Out"][0]]},
            outputs={"Out": ["reshape_2_out"],
                     "XShape": ["reshape_2_Xout"]},
            shape=(0, 0, 12, 64))
        transpose_0 = OpConfig(
            "transpose2",
            inputs={"X": [reshape_0.outputs["Out"][0]]},
            outputs={"Out": ["transpose_0_out"]},
            axis=(0, 2, 1, 3))
        transpose_1 = OpConfig(
            "transpose2",
            inputs={"X": [reshape_1.outputs["Out"][0]]},
            outputs={"Out": ["transpose_1_out"]},
            axis=(0, 2, 1, 3))
        transpose_2 = OpConfig(
            "transpose2",
            inputs={"X": [reshape_2.outputs["Out"][0]]},
            outputs={"Out": ["transpose_2_out"]},
            axis=(0, 2, 1, 3))
        scale_op = OpConfig(
            "scale",
            inputs={"X": [transpose_0.outputs["Out"][0]]},
            outputs={"Out": ["scale_out"]},
            scale=0.125,
            bias=0.0,
            bias_after_scale=True)
        matmul_0 = OpConfig(
            "matmul",
            inputs={
                "X": [scale_op.outputs["Out"][0]],
                "Y": [transpose_1.outputs["Out"][0]]
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
        ele_3 = OpConfig(
            "elementwise_add",
            inputs={
                "X": [matmul_0.outputs["Out"][0]],
                "Y": ["eltadd_qk_b_var"]
            },
            outputs={"Out": ["ele_3_out"]},
            axis=-1)
        softmax_op = OpConfig(
            "softmax",
            inputs={"X": [ele_3.outputs["Out"][0]]},
            outputs={"Out": ["softmax_out"]},
            axis=-1,
            is_test=True)
        matmul_1 = OpConfig(
            "matmul",
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
        transpose_3 = OpConfig(
            "transpose2",
            inputs={"X": [matmul_1.outputs["Out"][0]]},
            outputs={"Out": ["transpose_3_out"]},
            axis=(0, 2, 1, 3))
        reshape_3 = OpConfig(
            "reshape2",
            inputs={"X": [transpose_3.outputs["Out"][0]]},
            outputs={"Out": ["reshape_3_out"],
                     "XShape": ["reshape_3_Xout"]},
            shape=(0, 0, 768))
        ops = [
            mul_0, mul_1, mul_2, ele_0, ele_1, ele_2, reshape_0, reshape_1,
            reshape_2, transpose_0, transpose_1, transpose_2, scale_op,
            matmul_0, ele_3, softmax_op, matmul_1, transpose_3, reshape_3
        ]
        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "mul_x": TensorConfig(data_gen=partial(generate_mul_input)),
                "eltadd_qk_b_var":
                TensorConfig(data_gen=partial(generate_elewise_input))
            },
            weights={
                "mul_0_w": TensorConfig(shape=[768, 768]),
                "mul_1_w": TensorConfig(shape=[768, 768]),
                "mul_2_w": TensorConfig(shape=[768, 768]),
                "ele_0_w": TensorConfig(shape=[768]),
                "ele_1_w": TensorConfig(shape=[768]),
                "ele_2_w": TensorConfig(shape=[768])
            },
            outputs=[ops[-1].outputs["Out"][0]])
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            min_success_num=1,
            passes=["multihead_matmul_fuse_pass_v2"])


if __name__ == "__main__":
    unittest.main()
