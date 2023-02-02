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


class TestOneDNNMultiGruFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        input_dim1 = draw(st.integers(min_value=1, max_value=128)) * 3
        input_dim2 = input_dim1 // 3
        frame_size = draw(st.integers(min_value=1, max_value=128))
        shape_wx = [input_dim2, frame_size * 3]
        shape_wh = [frame_size, frame_size * 3]
        with_bias = draw(st.booleans())
        shape_bias = [1, frame_size * 3]
        axis = 1
        lod = [[0, input_dim1]]

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_bias(shape):
            if with_bias:
                return np.random.random(shape).astype(np.float32)
            else:
                return np.zeros(shape).astype(np.float32)

        fusion_gru_op_1 = OpConfig(
            type='fusion_gru',
            inputs={
                'X': ['input_data'],
                'WeightX': ['input_weight_x_1'],
                'WeightH': ['input_weight_h_1'],
                'Bias': ['bias_1'],
            },
            outputs={
                'Hidden': ['fusion_gru_output_hidden_1'],
                'XX': ['fusion_gru_output_xx_1'],
            },
            attrs={
                'origin_mode': False,
                'is_reverse': False,
            },
        )

        fusion_gru_op_2 = OpConfig(
            type='fusion_gru',
            inputs={
                'X': ['input_data'],
                'WeightX': ['input_weight_x_2'],
                'WeightH': ['input_weight_h_2'],
                'Bias': ['bias_2'],
            },
            outputs={
                'Hidden': ['fusion_gru_output_hidden_2'],
                'XX': ['fusion_gru_output_xx_2'],
            },
            attrs={'origin_mode': False, 'is_reverse': True},
        )

        concat_op = OpConfig(
            type='concat',
            inputs={
                'X': [
                    'fusion_gru_output_hidden_1',
                    'fusion_gru_output_hidden_2',
                ]
            },
            outputs={'Out': ['concat_output']},
            attrs={'axis': axis},
        )

        program_config = ProgramConfig(
            ops=[fusion_gru_op_1, fusion_gru_op_2, concat_op],
            weights={
                "input_weight_x_1": TensorConfig(
                    data_gen=partial(generate_data, shape_wx)
                ),
                "input_weight_h_1": TensorConfig(
                    data_gen=partial(generate_data, shape_wh)
                ),
                'bias_1': TensorConfig(
                    data_gen=partial(generate_bias, shape_bias)
                ),
                "input_weight_x_2": TensorConfig(
                    data_gen=partial(generate_data, shape_wx)
                ),
                "input_weight_h_2": TensorConfig(
                    data_gen=partial(generate_data, shape_wh)
                ),
                'bias_2': TensorConfig(
                    data_gen=partial(generate_data, shape_bias)
                ),
            },
            inputs={
                'input_data': TensorConfig(
                    lod=lod,
                    data_gen=partial(generate_data, [input_dim1, input_dim2]),
                ),
            },
            outputs=['concat_output'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=['multi_gru_fuse_pass'],
        )
        yield config, ['multi_gru'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False, passes=['multi_gru_fuse_pass'])


if __name__ == '__main__':
    unittest.main()
