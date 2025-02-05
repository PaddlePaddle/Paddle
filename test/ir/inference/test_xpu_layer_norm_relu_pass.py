# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.framework import core


@unittest.skipIf(
    not core.get_xpu_device_version(0) == core.XPUVersion.XPU2,
    "XpuLayerNormReluFuse only support XPU2",
)
class TestXpuLayerNormReluFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["layer_norm_relu_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=50))
        x_shape = [batch_size, 16, 128]
        y_shape = x_shape

        epsilon = draw(st.floats(min_value=0.0000001, max_value=0.001))
        begin_norm_axis = 2
        layer_norm_op = OpConfig(
            "layer_norm",
            inputs={
                "X": ["x"],
                "Scale": ["layer_norm_scale"],
                "Bias": ["layer_norm_bias"],
            },
            outputs={
                "Y": ["layer_norm_out"],
                "Mean": ["layer_norm_mean"],
                "Variance": ["layer_norm_var"],
            },
            begin_norm_axis=begin_norm_axis,
            epsilon=epsilon,
        )

        relu_op = OpConfig(
            "relu",
            inputs={
                "X": ["layer_norm_out"],
            },
            outputs={
                "Out": ["relu_out"],
            },
        )

        sub_graph = [layer_norm_op, relu_op]

        program_config = ProgramConfig(
            ops=sub_graph,
            weights={
                "layer_norm_scale": TensorConfig(shape=[x_shape[2]]),
                "layer_norm_bias": TensorConfig(shape=[x_shape[2]]),
            },
            inputs={
                "x": TensorConfig(shape=x_shape),
            },
            outputs=["relu_out"],
        )

        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["layer_norm_relu_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    if core.get_xpu_device_version(0) == core.XPUVersion.XPU2:
        unittest.main()
