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


class TestOneDNNMatmulTransposeReshapeFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        # If the problem has been fixed, the judgment
        # needs to be deleted!!!
        if 0 in attrs[2]['shape']:
            return False

        return True

    def sample_program_config(self, draw):
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        alpha = draw(st.floats(min_value=0.01, max_value=2))
        axis = draw(st.sampled_from([[0, 2, 1, 3]]))
        shape = draw(st.sampled_from([[0, -1, 128], [-1, 1, 64]]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim = draw(st.sampled_from([32, 64]))

        def generate_input(type):
            if transpose_X and transpose_Y:
                shape_x = [batch_size, channel, input_dim, 32]
                shape_y = [batch_size, channel, 64, input_dim]
            elif transpose_X:
                shape_x = [batch_size, channel, input_dim, 32]
                shape_y = [batch_size, channel, input_dim, 64]
            elif transpose_Y:
                shape_x = [batch_size, channel, 32, input_dim]
                shape_y = [batch_size, channel, 8, input_dim]
            else:
                shape_x = [batch_size, channel, 32, input_dim]
                shape_y = [batch_size, channel, input_dim, 16]

            if type == 'x':
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        matmul_op = OpConfig(
            type='matmul',
            inputs={'X': ['input_data1'], 'Y': ['input_data2']},
            outputs={'Out': ['matmul_output']},
            attrs={
                "transpose_X": transpose_X,
                "transpose_Y": transpose_Y,
                "alpha": alpha,
            },
        )

        transpose2_op = OpConfig(
            type='transpose2',
            inputs={'X': ['matmul_output']},
            outputs={
                'Out': ['transpose2_output'],
                'XShape': ['transpose2_xshape'],
            },
            attrs={'axis': axis},
        )

        reshape2_op = OpConfig(
            type='reshape2',
            inputs={'X': ['transpose2_output']},
            outputs={'Out': ['reshape2_output'], 'XShape': ['reshape2_xshape']},
            attrs={'shape': shape},
        )

        model_net = [matmul_op, transpose2_op, reshape2_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                'input_data1': TensorConfig(
                    data_gen=partial(generate_input, 'x')
                ),
                'input_data2': TensorConfig(
                    data_gen=partial(generate_input, 'y')
                ),
            },
            outputs=['reshape2_output'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['fused_matmul'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=['matmul_transpose_reshape_onednn_fuse_pass']
        )


if __name__ == '__main__':
    unittest.main()
