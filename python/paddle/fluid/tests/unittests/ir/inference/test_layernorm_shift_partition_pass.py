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


class TestLayernormShiftPartitionPass(PassAutoScanTest):
    """
       |
    layer_norm
       |
    reshape2
       |
    reshape2 
       | 
    transpose2
       |
    reshape2
       |
    reshape2
       |
    """

    def sample_predictor_configs(self, program_config):
        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        config.set_trt_dynamic_shape_info({
            "input_data": [1, 9, 96],
        }, {
            "input_data": [4, 3136, 768],
        }, {
            "input_data": [1, 784, 384],
        })
        yield config, ['layernorm_shift_partition'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        axis = [0, 1, 3, 2, 4, 5]
        epsilon = draw(st.floats(min_value=0.0000001, max_value=0.001))
        # begin_norm_axis has to be 2
        begin_norm_axis = 2
        batch_size = draw(st.integers(min_value=1, max_value=4))

        window_size = draw(st.sampled_from([3, 5, 7]))
        move_shape = draw(st.integers(min_value=1, max_value=8))
        dim = draw(st.sampled_from([96, 192, 384, 768]))

        def generate_input(attrs):
            return np.random.random(
                [attrs[1]["batch_size"],
                 *attrs[1]["input_dim"]]).astype(np.float32)

        def generate_weight(attrs):
            return np.random.random(attrs[1]['input_dim'][-1]).astype(
                np.float32)

        attrs = [{
            'begin_norm_axis': begin_norm_axis,
            'epsilon': epsilon,
        }, {
            'batch_size': batch_size,
            'input_dim': [(window_size * move_shape)**2, dim],
        }, {
            'axis': axis,
            'input_resolution': window_size * move_shape,
            'move_shape': move_shape,
            'window_size': window_size,
        }]

        layer_norm_op = OpConfig(type="layer_norm",
                                 inputs={
                                     "X": ["input_data"],
                                     "Bias": ["layer_norm_bias"],
                                     "Scale": ["layer_norm_scale"]
                                 },
                                 outputs={
                                     "Y": ["layer_norm_output1"],
                                     "Mean": ["layer_norm_output2"],
                                     "Variance": ["layer_norm_output3"]
                                 },
                                 attrs={
                                     "begin_norm_axis":
                                     attrs[0]["begin_norm_axis"],
                                     "epsilon": attrs[0]["epsilon"],
                                 })
        reshape_op2 = OpConfig(type="reshape2",
                               inputs={
                                   "X": ["layer_norm_output1"],
                               },
                               outputs={
                                   "Out": ["reshape_output2"],
                                   "XShape": ["reshape_output2_xshape"],
                               },
                               attrs={
                                   'shape': [
                                       -1, attrs[2]["input_resolution"],
                                       attrs[2]["input_resolution"],
                                       attrs[1]["input_dim"][-1]
                                   ]
                               })
        reshape_op3 = OpConfig(type="reshape2",
                               inputs={
                                   "X": ["reshape_output2"],
                               },
                               outputs={
                                   "Out": ["reshape_output3"],
                                   "XShape": ["reshape_output3_xshape"],
                               },
                               attrs={
                                   'shape': [
                                       -1, attrs[2]["move_shape"],
                                       attrs[2]["window_size"],
                                       attrs[2]["move_shape"],
                                       attrs[2]["window_size"],
                                       attrs[1]["input_dim"][-1]
                                   ]
                               })
        transpose_op4 = OpConfig(type='transpose2',
                                 inputs={
                                     "X": ["reshape_output3"],
                                 },
                                 outputs={"Out": ["transpose_output4"]},
                                 attrs={"axis": attrs[2]['axis']})
        reshape_op5 = OpConfig(type="reshape2",
                               inputs={
                                   "X": ["transpose_output4"],
                               },
                               outputs={
                                   "Out": ["reshape_output5"],
                                   "XShape": ["reshape_output5_xshape"],
                               },
                               attrs={
                                   'shape': [
                                       -1, attrs[2]["window_size"],
                                       attrs[2]["window_size"],
                                       attrs[1]["input_dim"][-1]
                                   ]
                               })
        reshape_op6 = OpConfig(
            type="reshape2",
            inputs={
                "X": ["reshape_output5"],
            },
            outputs={
                "Out": ["reshape_output6"],
                "XShape": ["reshape_output6_xshape"],
            },
            attrs={
                'shape':
                [-1, attrs[2]["window_size"]**2, attrs[1]["input_dim"][-1]]
            })

        program_config = ProgramConfig(
            ops=[
                layer_norm_op, reshape_op2, reshape_op3, transpose_op4,
                reshape_op5, reshape_op6
            ],
            weights={
                "layer_norm_bias":
                TensorConfig(data_gen=partial(generate_weight, attrs)),
                "layer_norm_scale":
                TensorConfig(data_gen=partial(generate_weight, attrs))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            outputs=["reshape_output6"])

        return program_config

    def test(self):
        self.run_and_statis(quant=False,
                            max_examples=20,
                            passes=["layernorm_shift_partition_fuse_pass"],
                            max_duration=250,
                            min_success_num=20)


if __name__ == "__main__":
    unittest.main()
