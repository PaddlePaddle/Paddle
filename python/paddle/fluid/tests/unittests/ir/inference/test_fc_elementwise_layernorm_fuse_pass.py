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


class TestFCElementwiseLayerNormFusePass(PassAutoScanTest):
    """
    x_var   w(persistable) bias_var(persistable)
      \     |              /
          fc     
          |
      fc_out_var  bias_var(persistable)
            \        /
          elementwise_add  bias_var(persistable)  scale_var(persistable)
                  \            |                       /                    
                           layer_norm
                         /      |         \
                        Y    mean_var  variance_var
    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["fused_fc_elementwise_layernorm"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X of fc
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=5))
        x_shape = [2, 1]
        x_rank = len(x_shape)
        # 2. Generate attr:in_num_col_dims of fc
        in_num_col_dims = draw(st.integers(min_value=1, max_value=x_rank - 1))
        # 3. Generate legal shape of input:W/bias of fc
        w_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=2))
        w_shape[0] = int(np.prod(x_shape[in_num_col_dims:]))
        w_shape = [1, 2]
        fc_bias_shape = [w_shape[1], ]
        if draw(st.booleans()):
            fc_bias_shape.insert(0, 1)
        fc_bias_shape = [2, ]
        fc_out_shape = x_shape[:in_num_col_dims] + w_shape[1:]
        # 4. Generate legal attr:axis/shape of elementwise_add
        add_bias_shape = fc_out_shape[:]
        axis = draw(st.integers(min_value=-1, max_value=0))
        # 5. Generate legal shape of layer_norm
        begin_norm_axis = draw(
            st.integers(
                min_value=1, max_value=len(fc_out_shape) - 1))
        layer_norm_shape = [int(np.prod(fc_out_shape[begin_norm_axis:]))]
        epsilon = 1e-5

        fc_op = OpConfig(
            "fc",
            inputs={"Input": ["fc_x"],
                    "W": ["fc_w"],
                    "Bias": ["fc_bias"]},
            outputs={"Out": ["fc_out"]},
            in_num_col_dims=in_num_col_dims,
            padding_weights=False,
            activation_type="",
            use_quantizer=False,
            use_mkldnn=False, )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["fc_out"],
                    "Y": ["add_bias"]},
            outputs={"Out": ["add_out"]},
            axis=axis, )
        layer_norm_op = OpConfig(
            "layer_norm",
            inputs={
                "X": ["add_out"],
                "Scale": ["scale"],
                "Bias": ["layer_norm_bias"]
            },
            outputs={
                "Y": ["layer_norm_out"],
                "Mean": ["layer_norm_mean"],
                "Variance": ["layer_norm_var"]
            },
            begin_norm_axis=begin_norm_axis,
            epsilon=epsilon)

        ops = [fc_op, add_op, layer_norm_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "fc_w": TensorConfig(shape=w_shape),
                "fc_bias": TensorConfig(shape=fc_bias_shape),
                "add_bias": TensorConfig(shape=add_bias_shape),
                "scale": TensorConfig(shape=layer_norm_shape),
                "layer_norm_bias": TensorConfig(shape=layer_norm_shape),
            },
            inputs={"fc_x": TensorConfig(shape=x_shape), },
            outputs=ops[-1].outputs["Y"], )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["fc_elementwise_layernorm_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
