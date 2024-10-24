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

import unittest

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMatmulScaleFusePass(PassAutoScanTest):
    r"""
    x_var    y_var(persistable)
      \       /
       matmul
         |
       scale
    """

    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_gpu=False)
        yield config, [
            "matmul",
        ], (1e-5, 1e-5)

        # onednn
        config = self.create_inference_config(use_mkldnn=True)
        yield config, [
            "matmul",
        ], (1e-5, 1e-5)

        # gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, [
            "matmul",
        ], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of matmul
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=2, max_size=5
            )
        )
        x_shape_rank = len(x_shape)
        y_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8),
                min_size=x_shape_rank,
                max_size=x_shape_rank,
            )
        )
        y_shape_rank = len(y_shape)
        y_shape[-2] = x_shape[-1]
        for i in range(y_shape_rank - 3, -1, -1):
            j = x_shape_rank - (y_shape_rank - i)
            if j < 0 or j >= x_shape_rank:
                break
            y_shape[i] = x_shape[j]

        transpose_X = False
        transpose_Y = False
        alpha = draw(st.floats(min_value=-2.0, max_value=2.0, width=32))
        # scale tensor
        scale_shape = [1]
        scale_value = draw(st.floats(min_value=-5.0, max_value=5.0, width=32))

        matmul_op = OpConfig(
            "matmul",
            inputs={"X": ["matmul_x"], "Y": ["matmul_y"]},
            outputs={"Out": ["matmul_out"]},
            transpose_X=transpose_X,
            transpose_Y=transpose_Y,
            alpha=alpha,
            head_number=1,
        )
        is_scale_tensor = draw(st.booleans())
        if is_scale_tensor:
            scale_op = OpConfig(
                "scale",
                inputs={"X": ["matmul_out"], "ScaleTensor": ["scale_tensor"]},
                outputs={"Out": ["scale_out"]},
                scale=scale_value,
                bias=0.0,
                bias_after_scale=draw(st.booleans()),
            )
        else:
            scale_op = OpConfig(
                "scale",
                inputs={
                    "X": ["matmul_out"],
                },
                outputs={"Out": ["scale_out"]},
                scale=scale_value,
                bias=0.0,
                bias_after_scale=draw(st.booleans()),
            )

        ops = [matmul_op, scale_op]
        weights = {}
        inputs = {}
        if is_scale_tensor:
            weights = {
                "matmul_y": TensorConfig(shape=y_shape),
                "scale_tensor": TensorConfig(shape=scale_shape),
            }
            inputs = {
                "matmul_x": TensorConfig(shape=x_shape),
            }
        else:
            inputs = {
                "matmul_x": TensorConfig(shape=x_shape),
                "matmul_y": TensorConfig(shape=y_shape),
            }

        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["matmul_scale_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
