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


class TestSkipLayernormFusePass(PassAutoScanTest):
    """
          x_var           y_var
             \            /
             elementwise_add     
                    |
          layer_norm (Scale Bias)
    """

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
            "elementwise_add_x": [1, 1, 128],
            "elementwise_add_y": [1, 1, 128]
        }, {
            "elementwise_add_x": [16, 128, 128],
            "elementwise_add_y": [16, 128, 128]
        }, {
            "elementwise_add_x": [8, 64, 128],
            "elementwise_add_y": [8, 64, 128]
        })
        yield config, ['skip_layernorm'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        # Here we put some skip rules to avoid known bugs
        def teller1(program_config, predictor_config):
            return False

        self.add_ignore_check_case(
            teller1,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "skip_layernorm", )

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        elewise_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["elementwise_add_x"],
                    "Y": ["elementwise_add_y"]},
            outputs={"Out": ["elementwise_add_out"]},
            axis=-1)
        layernorm_op = OpConfig(
            "layer_norm",
            inputs={
                "X": ["elementwise_add_out"],
                "Bias": ["bias"],
                "Scale": ["scale"]
            },
            outputs={"Y": ["y"],
                     "Mean": ["mean"],
                     "Variance": ["variance"]},
            begin_norm_axis=2,
            epsilon=0.000009)
        ops = [elewise_op, layernorm_op]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "elementwise_add_x": TensorConfig(shape=[2, 16, 128]),
                "elementwise_add_y": TensorConfig(shape=[2, 16, 128])
            },
            weights={
                "bias": TensorConfig(shape=[128]),
                "scale": TensorConfig(shape=[128])
            },
            outputs=[ops[-1].outputs["Y"][0]])
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            min_success_num=1,
            passes=["skip_layernorm_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
