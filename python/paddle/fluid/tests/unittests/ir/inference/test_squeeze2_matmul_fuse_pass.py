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


class TestSqueeze2MatmulFusePass(PassAutoScanTest):
    """
        x_var  
          |          
       squeeze2 
          \
    squeeze2_out_var    y_var
             \           /
                 matmul      bias_var
                    \          /
                   elementwise_add  
    """

    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_gpu=False)
        yield config, ["mul", "elementwise_add"], (1e-5, 1e-5)

        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["mul", "elementwise_add"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X of squeeze2
        x_shape = draw(
            st.lists(st.integers(min_value=1, max_value=8),
                     min_size=2,
                     max_size=2))
        # axes of squeeze2 == [2, 3]
        x_shape += [1, 1]
        axes = [2, 3]

        # 2. Generate attr:transpose_X/transpose_Y/alpha of matmul
        alpha = 1.0
        transpose_X = False
        transpose_Y = False

        # 3. Generate legal shape of input:Y of matmul
        y_shape = draw(
            st.lists(st.integers(min_value=1, max_value=8),
                     min_size=2,
                     max_size=2))
        y_shape[0] = x_shape[1]

        # 4. Generate legal attr:axis of elementwise_add
        axis = draw(st.integers(min_value=-1, max_value=1))
        if axis == 0 or axis == -1:
            if draw(st.booleans()):
                if axis == 0:
                    bias_shape = [
                        x_shape[0],
                    ]
                else:
                    bias_shape = [
                        y_shape[1],
                    ]
            else:
                bias_shape = [x_shape[0], y_shape[1]]
        elif axis == 1:
            bias_shape = [
                y_shape[1],
            ]

        if draw(st.integers(min_value=1, max_value=10)) <= 1:
            bias_shape[-1] = 1
            if len(bias_shape) == 2 and draw(st.booleans()):
                bias_shape[0] = 1

        squeeze2_op = OpConfig(
            "squeeze2",
            inputs={
                "X": ["squeeze2_x"],
            },
            axes=axes,
            outputs={
                "Out": ["squeeze2_out"],
                "XShape": ["xshape"]
            },
        )
        matmul_op = OpConfig(
            "matmul",
            inputs={
                "X": ["squeeze2_out"],
                "Y": ["matmul_y"]
            },
            outputs={"Out": ["matmul_out"]},
            alpha=alpha,
            transpose_X=transpose_X,
            transpose_Y=transpose_Y,
            fused_reshape_X=[],
            fused_reshape_Y=[],
            fused_transpose_X=[],
            fused_transpose_Y=[],
            fused_reshape_Out=[],
            fused_transpose_Out=[],
        )

        add_op = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["matmul_out"],
                "Y": ["bias"]
            },
            outputs={"Out": ["add_out"]},
            axis=axis,
        )

        ops = [squeeze2_op, matmul_op, add_op]

        if draw(st.integers(min_value=1, max_value=10)) <= 8:
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "matmul_y": TensorConfig(shape=y_shape),
                    "bias": TensorConfig(shape=bias_shape),
                },
                inputs={
                    "squeeze2_x": TensorConfig(shape=x_shape),
                },
                outputs=ops[-1].outputs["Out"],
            )
        else:
            program_config = ProgramConfig(
                ops=ops,
                weights={},
                inputs={
                    "squeeze2_x": TensorConfig(shape=x_shape),
                    "matmul_y": TensorConfig(shape=y_shape),
                    "bias": TensorConfig(shape=bias_shape),
                },
                outputs=ops[-1].outputs["Out"],
            )
        return program_config

    def test(self):
        self.run_and_statis(quant=False,
                            max_examples=50,
                            max_duration=1000,
                            passes=["gpu_cpu_squeeze2_matmul_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
