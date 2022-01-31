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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import OpConfig, TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestFCMishMkldnnFusePass(PassAutoScanTest):

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
        mish_op = OpConfig(
            "mish",
            inputs={"X": ["fc_out"]},
            outputs={"Out": ["mish_output"]},
            axis=axis,
            scale=draw(st.floats(min_value=0, max_value=10)),
            offset=draw(st.floats(min_value=0, max_value=10)),
            )

        ops = [fc_op, mish_op]
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
            outputs=ops[1].outputs["Out"], )
        return program_config


    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["fc"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False, passes=["fc_act_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
