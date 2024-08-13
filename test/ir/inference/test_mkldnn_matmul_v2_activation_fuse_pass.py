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
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMatmulv2ActivationMkldnnFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        batch_size = draw(st.integers(min_value=2, max_value=4))
        channel = draw(st.sampled_from([16, 32, 64]))
        input_dim = draw(st.sampled_from([16, 32, 64]))
        activation_type = draw(
            st.sampled_from(
                [
                    'relu',
                    'gelu',
                    'swish',
                    'mish',
                    'sqrt',
                    'hard_swish',
                    'sigmoid',
                    'abs',
                    'relu6',
                    'clip',
                    'tanh',
                    'hard_sigmoid',
                    'leaky_relu',
                    'scale',
                ]
            )
        )

        def generate_input(type):
            broadcast_X = st.booleans()
            channel_X = 1 if broadcast_X else channel
            channel_Y = channel if broadcast_X else 1
            batch_size_X = 1 if broadcast_X else batch_size
            batch_size_Y = batch_size if broadcast_X else 1

            if transpose_X and transpose_Y:
                shape_x = [batch_size_X, channel_X, input_dim, 32]
                shape_y = [batch_size_Y, channel_Y, 64, input_dim]
            elif transpose_X:
                shape_x = [batch_size_X, channel_X, input_dim, 32]
                shape_y = [batch_size_Y, channel_Y, input_dim, 64]
            elif transpose_Y:
                shape_x = [batch_size_X, channel_X, 32, input_dim]
                shape_y = [batch_size_Y, channel_Y, 8, input_dim]
            else:
                shape_x = [batch_size_X, channel_X, 32, input_dim]
                shape_y = [batch_size_Y, channel_Y, input_dim, 16]

            if type == 'X':
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        matmul_op = OpConfig(
            type='matmul_v2',
            inputs={'X': ['matmul_X'], 'Y': ['matmul_Y']},
            outputs={'Out': ['matmul_output']},
            attrs={
                'trans_x': transpose_X,
                'trans_y': transpose_Y,
                'use_mkldnn': True,
            },
        )

        if activation_type == 'relu6':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['matmul_output']},
                outputs={'Out': ['activation_output']},
                threshold=6.0,
            )
        elif activation_type == "leaky_relu":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["matmul_output"]},
                outputs={"Out": ["activation_output"]},
                alpha=draw(st.floats(min_value=0.1, max_value=1.0)),
            )
        elif activation_type == "scale":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["matmul_output"]},
                outputs={"Out": ["activation_output"]},
                scale=draw(st.sampled_from([0.125, 0.4, 0.875, 2])),
            )
        elif activation_type == 'swish':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['matmul_output']},
                outputs={'Out': ['activation_output']},
                beta=1.0,
            )
        elif activation_type == 'clip':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['matmul_output']},
                outputs={'Out': ['activation_output']},
                min=draw(st.floats(min_value=0.1, max_value=0.49)),
                max=draw(st.floats(min_value=0.5, max_value=1.0)),
            )
        else:
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['matmul_output']},
                outputs={'Out': ['activation_output']},
            )

        model_net = [matmul_op, activation_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                'matmul_X': TensorConfig(data_gen=partial(generate_input, 'X')),
                'matmul_Y': TensorConfig(data_gen=partial(generate_input, 'Y')),
            },
            outputs=['activation_output'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                'matmul_activation_onednn_fuse_pass',
                'operator_scale_onednn_fuse_pass',
            ],
        )
        yield config, ['fused_matmul'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=50,
            passes=[
                'matmul_activation_onednn_fuse_pass',
                'operator_scale_onednn_fuse_pass',
            ],
        )


if __name__ == '__main__':
    unittest.main()
