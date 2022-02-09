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


class TestMapMatmulToMulPass(PassAutoScanTest):
    """
     x_var    y_var(persistable)
       \       /
        matmul_v2  
    """

    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_gpu=False)
        yield config, ["matmul", ], (1e-5, 1e-5)

        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["matmul", ], (1e-5, 1e-5)

        # TRT
        # config = self.create_trt_inference_config()
        # config.enable_tensorrt_engine(
        #     max_batch_size=10,
        #     workspace_size=10240,
        #     min_subgraph_size=0,
        #     precision_mode=paddle_infer.PrecisionType.Float32,
        #     use_static=False,
        #     use_calib_mode=False)
        # yield config, ["matmul", ], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        # Here we put some skip rules to avoid known bugs
        def teller1(program_config, predictor_config):
            if predictor_config.tensorrt_engine_enabled():
                # On 3080, the results of MatMul and Mul are different
                return True

                x_shape = list(program_config.inputs["matmul_x"].shape)
                if len(x_shape) > 5:
                    return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The pass error on TRT while shape of mul_x > 5.")

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of matmul
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=5))
        y_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=2))
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        if transpose_X:
            if transpose_Y:
                y_shape[1] = x_shape[-2]
            else:
                y_shape[0] = x_shape[-2]
        else:
            if transpose_Y:
                y_shape[1] = x_shape[-1]
            else:
                y_shape[0] = x_shape[-1]

        y_shape = x_shape[0:len(x_shape) - 2] + y_shape
        alpha = 1.0

        matmul_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul_x"],
                    "Y": ["matmul_y"]},
            outputs={"Out": ["matmul_out"]},
            alpha=alpha,
            trans_x=transpose_X,
            trans_y=transpose_Y,
            fused_reshape_Out=[],
            fused_transpose_Out=[],
            fused_reshape_X=[],
            fused_reshape_Y=[],
            fused_transpose_X=[],
            fused_transpose_Y=[], )

        ops = [matmul_op, ]
        weights = {}
        inputs = {
            "matmul_x": TensorConfig(shape=x_shape),
            "matmul_y": TensorConfig(shape=y_shape),
        }

        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=ops[-1].outputs["Out"], )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["gpu_cpu_map_matmul_v2_to_matmul_pass"])


if __name__ == "__main__":
    unittest.main()
