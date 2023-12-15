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

import unittest

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMultiEncoderXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["multi_encoder_xpu"], (1e-1, 1e-1)

    def multi_encoder_xpu_program_config(self, draw):
        # q: matmul+add+reshape+transpose
        q_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["q_matmul_x"], "Y": ["q_matmul_w"]},
            outputs={"Out": ["q_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        q_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["q_matmul_out"], "Y": ["q_add_bias"]},
            outputs={"Out": ["q_add_out"]},
            axis=2,
        )
        q_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["q_add_out"]},
            outputs={"Out": ["q_reshape_out"], "XShape": ["q_reshape_xshape"]},
            shape=[0, 0, 12, 64],
        )
        q_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["q_reshape_out"]},
            outputs={
                "Out": ["q_transpose_out"],
                "XShape": ["q_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        # k: matmul+add+reshape+transpose
        k_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["q_matmul_x"], "Y": ["k_matmul_w"]},
            outputs={"Out": ["k_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        k_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["k_matmul_out"], "Y": ["k_add_bias"]},
            outputs={"Out": ["k_add_out"]},
            axis=2,
        )
        k_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["k_add_out"]},
            outputs={"Out": ["k_reshape_out"], "XShape": ["k_reshape_xshape"]},
            shape=[0, 0, 12, 64],
        )
        k_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["k_reshape_out"]},
            outputs={
                "Out": ["k_transpose_out"],
                "XShape": ["k_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        # v: matmul+add+reshape+transpose
        v_matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["q_matmul_x"], "Y": ["v_matmul_w"]},
            outputs={"Out": ["v_matmul_out"]},
            trans_x=False,
            trans_y=False,
        )
        v_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["v_matmul_out"], "Y": ["v_add_bias"]},
            outputs={"Out": ["v_add_out"]},
            axis=2,
        )
        v_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["v_add_out"]},
            outputs={"Out": ["v_reshape_out"], "XShape": ["v_reshape_xshape"]},
            shape=[0, 0, 12, 64],
        )
        v_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["v_reshape_out"]},
            outputs={
                "Out": ["v_transpose_out"],
                "XShape": ["v_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        # qk: matmul+add+softmax
        qk_matmul_op = OpConfig(
            "matmul",
            inputs={"X": ["q_transpose_out"], "Y": ["k_transpose_out"]},
            outputs={"Out": ["qk_matmul_out"]},
            alpha=0.125,
            transpose_X=False,
            transpose_Y=True,
        )
        qk_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qk_matmul_out"], "Y": ["qk_add_mask"]},
            outputs={"Out": ["qk_add_out"]},
            axis=-1,
        )
        qk_softmax_op = OpConfig(
            "softmax",
            inputs={"X": ["qk_add_out"]},
            outputs={"Out": ["qk_softmax_out"]},
            axis=-1,
        )
        # qkv
        qkv_matmul_0_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qk_softmax_out"], "Y": ["v_transpose_out"]},
            outputs={"Out": ["qkv_matmul_0_out"]},
            trans_x=False,
            trans_y=False,
        )
        qkv_transpose_op = OpConfig(
            "transpose2",
            inputs={"X": ["qkv_matmul_0_out"]},
            outputs={
                "Out": ["qkv_transpose_out"],
                "XShape": ["qkv_transpose_xshape"],
            },
            axis=[0, 2, 1, 3],
        )
        qkv_reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["qkv_transpose_out"]},
            outputs={
                "Out": ["qkv_reshape_out"],
                "XShape": ["qkv_reshape_xshape"],
            },
            shape=[0, 0, 768],
        )
        qkv_matmul_1_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qkv_reshape_out"], "Y": ["qkv_matmul_1_w"]},
            outputs={"Out": ["qkv_matmul_1_out"]},
            trans_x=False,
            trans_y=False,
        )
        qkv_add_0_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qkv_matmul_1_out"], "Y": ["qkv_add_0_bias"]},
            outputs={"Out": ["qkv_add_0_out"]},
            axis=2,
        )
        qkv_add_1_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qkv_add_0_out"], "Y": ["q_matmul_x"]},
            outputs={"Out": ["qkv_add_1_out"]},
            axis=-1,
        )
        ln_1_op = OpConfig(
            "layer_norm",
            inputs={
                "X": ["qkv_add_1_out"],
                "Bias": ["ln_1_bias"],
                "Scale": ["ln_1_scale"],
            },
            outputs={
                "Y": ["ln_1_out"],
                "Mean": ["ln_1_mean"],
                "Variance": ["ln_1_variance"],
            },
            begin_norm_axis=2,
            epsilon=1e-14,
        )
        qkv_matmul_2_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["ln_1_out"], "Y": ["qkv_matmul_2_w"]},
            outputs={"Out": ["qkv_matmul_2_out"]},
            trans_x=False,
            trans_y=False,
        )
        qkv_add_2_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qkv_matmul_2_out"], "Y": ["qkv_add_2_bias"]},
            outputs={"Out": ["qkv_add_2_out"]},
            axis=2,
        )
        qkv_act_op = OpConfig(
            "gelu",
            inputs={"X": ["qkv_add_2_out"]},
            outputs={"Out": ["qkv_act_out"]},
            approximate=False,
        )
        qkv_matmul_3_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["qkv_act_out"], "Y": ["qkv_matmul_3_w"]},
            outputs={"Out": ["qkv_matmul_3_out"]},
            trans_x=False,
            trans_y=False,
        )
        qkv_add_3_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["qkv_matmul_3_out"], "Y": ["qkv_add_3_bias"]},
            outputs={"Out": ["qkv_add_3_out"]},
            axis=2,
        )
        qkv_add_4_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["ln_1_out"], "Y": ["qkv_add_3_out"]},
            outputs={"Out": ["qkv_add_4_out"]},
            axis=-1,
        )
        ln_2_op = OpConfig(
            "layer_norm",
            inputs={
                "X": ["qkv_add_4_out"],
                "Bias": ["ln_2_bias"],
                "Scale": ["ln_2_scale"],
            },
            outputs={
                "Y": ["ln_2_out"],
                "Mean": ["ln_2_mean"],
                "Variance": ["ln_2_variance"],
            },
            begin_norm_axis=2,
            epsilon=1e-14,
        )
        ops = [
            q_matmul_op,
            q_add_op,
            q_reshape_op,
            q_transpose_op,
            k_matmul_op,
            k_add_op,
            k_reshape_op,
            k_transpose_op,
            v_matmul_op,
            v_add_op,
            v_reshape_op,
            v_transpose_op,
            qk_matmul_op,
            qk_add_op,
            qk_softmax_op,
            qkv_matmul_0_op,
            qkv_transpose_op,
            qkv_reshape_op,
            qkv_matmul_1_op,
            qkv_add_0_op,
            qkv_add_1_op,
            ln_1_op,
            qkv_matmul_2_op,
            qkv_add_2_op,
            qkv_act_op,
            qkv_matmul_3_op,
            qkv_add_3_op,
            qkv_add_4_op,
            ln_2_op,
        ]

        q_matmul_x_shape = draw(
            st.lists(
                st.integers(min_value=3, max_value=10), min_size=3, max_size=3
            )
        )
        q_matmul_x_shape[2] = 768
        q_matmul_w_shape = [q_matmul_x_shape[2], q_matmul_x_shape[2]]
        q_add_bias_shape = [q_matmul_x_shape[2]]
        qk_add_mask_shape = [q_matmul_x_shape[0], 1, 1, q_matmul_x_shape[1]]
        qkv_matmul_2_w_shape = [q_matmul_x_shape[2], 3072]
        qkv_add_2_bias_shape = [qkv_matmul_2_w_shape[1]]
        qkv_matmul_3_w_shape = [3072, q_matmul_x_shape[2]]
        qkv_add_3_bias_shape = [qkv_matmul_3_w_shape[1]]
        ln_1_bias_shape = [q_matmul_x_shape[2]]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "q_matmul_w": TensorConfig(shape=q_matmul_w_shape),
                "q_add_bias": TensorConfig(shape=q_add_bias_shape),
                "k_matmul_w": TensorConfig(shape=q_matmul_w_shape),
                "k_add_bias": TensorConfig(shape=q_add_bias_shape),
                "v_matmul_w": TensorConfig(shape=q_matmul_w_shape),
                "v_add_bias": TensorConfig(shape=q_add_bias_shape),
                "qkv_matmul_1_w": TensorConfig(shape=q_matmul_w_shape),
                "qkv_add_0_bias": TensorConfig(shape=q_add_bias_shape),
                "qkv_matmul_2_w": TensorConfig(shape=qkv_matmul_2_w_shape),
                "qkv_add_2_bias": TensorConfig(shape=qkv_add_2_bias_shape),
                "qkv_matmul_3_w": TensorConfig(shape=qkv_matmul_3_w_shape),
                "qkv_add_3_bias": TensorConfig(shape=qkv_add_3_bias_shape),
                "ln_1_bias": TensorConfig(shape=ln_1_bias_shape),
                "ln_1_scale": TensorConfig(shape=ln_1_bias_shape),
                "ln_2_bias": TensorConfig(shape=ln_1_bias_shape),
                "ln_2_scale": TensorConfig(shape=ln_1_bias_shape),
            },
            inputs={
                "q_matmul_x": TensorConfig(shape=q_matmul_x_shape),
                "qk_add_mask": TensorConfig(shape=qk_add_mask_shape),
            },
            outputs=["ln_2_out"],
        )
        return program_config

    def sample_program_config(self, draw):
        return self.multi_encoder_xpu_program_config(draw)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=2,
            min_success_num=2,
            passes=["multi_encoder_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
