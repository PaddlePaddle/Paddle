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

import hypothesis.strategies as st
import numpy as np
import unittest
from auto_scan_test import PassAutoScanTest
from functools import partial
from program_config import TensorConfig, ProgramConfig, OpConfig


class TestSoftplusActivationOneDNNFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        activation_type = draw(
            st.sampled_from(
                [
                    'relu',
                    'gelu',
                    'tanh',
                    'sigmoid',
                    'swish',
                    'mish',
                    'sqrt',
                    'hard_swish',
                    'abs',
                    'relu6',
                    'clip',
                    'leaky_relu',
                ]
            )
        )

        def generate_input():
            return np.random.random([4, 3, 100, 100]).astype(np.float32)

        softplus_op = OpConfig(
            type='softplus',
            inputs={
                'X': ['activation_X'],
            },
            outputs={'Out': ['softplus_out']},
            attrs={
                'beta': draw(st.floats(min_value=0.5, max_value=2)),
                'threshold': draw(st.floats(min_value=15, max_value=30)),
            },
        )

        if activation_type == 'clip':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['softplus_out']},
                outputs={'Out': ['activation_output']},
                min=draw(st.floats(min_value=0.1, max_value=0.49)),
                max=draw(st.floats(min_value=0.5, max_value=1.0)),
            )
        elif activation_type == "gelu":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["softplus_out"]},
                outputs={"Out": ["activation_output"]},
                approximate=draw(st.booleans()),
            )
        elif activation_type == 'leaky_relu':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['softplus_out']},
                outputs={'Out': ['activation_output']},
                alpha=draw(st.floats(min_value=0.1, max_value=1.0)),
            )
        elif activation_type == 'relu6':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['softplus_out']},
                outputs={'Out': ['activation_output']},
                threshold=draw(st.floats(min_value=1.0, max_value=10.0)),
            )
        elif activation_type == 'swish':
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['softplus_out']},
                outputs={'Out': ['activation_output']},
                beta=draw(st.floats(min_value=0.1, max_value=10.0)),
            )
        else:
            activation_op = OpConfig(
                activation_type,
                inputs={'X': ['softplus_out']},
                outputs={'Out': ['activation_output']},
            )

        model_net = [softplus_op, activation_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                'activation_X': TensorConfig(data_gen=partial(generate_input))
            },
            outputs=['activation_output'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['softplus'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=40,
            passes=['softplus_activation_mkldnn_fuse_pass'],
        )


if __name__ == '__main__':
    unittest.main()
