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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestUnsqueezeEltwiseFusePass(PassAutoScanTest):
    """
        y_var  
          |          
       unsqueeze2 
          \
    unsqueeze2_out_var    x_var
             \           /
            elementwise_mul 
    """

    def sample_predictor_configs(self, program_config):
        # TRT
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=10,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['elementwise_mul', ], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of mul
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=4, max_size=4))
        axis = -1

        # 2. Generate legal shape and attr of input:Y of unsqueeze2
        y_shape = x_shape[:2]
        unsqueeze2_axes = [2, 3]

        unsqueeze2_op = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["unsqueeze2_x"],
                "AxesTensor": [],
                "AxesTensorList": []
            },
            axes=unsqueeze2_axes,
            outputs={"Out": ["unsqueeze2_out"],
                     "XShape": ["xshape"]}, )
        mul_op = OpConfig(
            "elementwise_mul",
            inputs={"Y": ["unsqueeze2_out"],
                    "X": ["mul_x"]},
            axis=axis,
            outputs={"Out": ["mul_out"]}, )

        ops = [
            unsqueeze2_op,
            mul_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "mul_x": TensorConfig(shape=x_shape),
                "unsqueeze2_x": TensorConfig(shape=y_shape),
            },
            outputs=ops[-1].outputs["Out"], )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["unsqueeze2_eltwise_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
