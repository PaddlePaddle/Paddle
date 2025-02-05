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


class TestOneDNNFCGruFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_size = draw(st.integers(min_value=1, max_value=16))
        fc_input_shape = [batch_size, 64]
        fc_weight_shape = [64, 192]
        fc_bias_shape = [1, 192]
        lod = [[0, batch_size]]

        gru_weight_shape = [64, 192]
        gru_bias_shape = [1, 192]
        activation = draw(st.sampled_from(['tanh']))
        is_reverse = draw(st.booleans())
        gate_activation = draw(st.sampled_from(['sigmoid']))

        mul_op = OpConfig(
            type='mul',
            inputs={'X': ['fc_input'], 'Y': ['fc_weight']},
            outputs={'Out': ['mul_out']},
            attrs={'x_num_col_dims': 1, 'y_num_col_dims': 1},
        )

        elt_op = OpConfig(
            type='elementwise_add',
            inputs={'X': ['mul_out'], 'Y': ['fc_bias']},
            outputs={'Out': ['fc_output']},
            attrs={'axis': -1},
        )

        gru_op = OpConfig(
            type='gru',
            inputs={
                'Input': ['fc_output'],
                'Weight': ['gru_weight'],
                'Bias': ['gru_bias'],
            },
            outputs={
                'BatchGate': ['batch_gate'],
                'BatchHidden': ['batch_hidden'],
                'BatchResetHiddenPrev': ['batch_reset'],
                'Hidden': ['gru_hidden'],
            },
            attrs={
                'activation': activation,
                'is_reverse': is_reverse,
                'gate_activation': gate_activation,
                'is_test': True,
            },
        )

        model_net = [mul_op, elt_op, gru_op]

        program_config = ProgramConfig(
            ops=model_net,
            inputs={
                'fc_input': TensorConfig(
                    lod=lod, data_gen=partial(generate_input, fc_input_shape)
                )
            },
            weights={
                'fc_weight': TensorConfig(
                    data_gen=partial(generate_input, fc_weight_shape)
                ),
                'fc_bias': TensorConfig(
                    data_gen=partial(generate_input, fc_bias_shape)
                ),
                'gru_weight': TensorConfig(
                    data_gen=partial(generate_input, gru_weight_shape)
                ),
                'gru_bias': TensorConfig(
                    data_gen=partial(generate_input, gru_bias_shape)
                ),
            },
            outputs=['gru_hidden'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                'onednn_placement_pass',
                'fc_gru_fuse_pass',
            ],
        )
        yield config, ['fusion_gru'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=[
                'onednn_placement_pass',
                'fc_gru_fuse_pass',
            ],
            max_examples=100,
        )


if __name__ == '__main__':
    unittest.main()
