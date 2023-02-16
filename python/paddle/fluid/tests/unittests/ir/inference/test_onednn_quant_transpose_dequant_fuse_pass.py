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


class TestQuantTranspose2DequantOneDNNFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        transpose_X = draw(st.booleans())
        axis = draw(st.sampled_from([[0, 2, 1, 3]]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim = draw(st.sampled_from([32, 64]))
        scale = draw(st.floats(min_value=1, max_value=16))
        shift = draw(st.integers(min_value=1, max_value=3))
        is_negative_input = draw(st.booleans())

        def generate_input():
            if transpose_X:
                shape_x = [batch_size, channel, input_dim, 32]
            else:
                shape_x = [batch_size, channel, 32, input_dim]
            return np.random.random(shape_x).astype(np.float32)

        quantize_op = OpConfig(
            type='quantize',
            inputs={'Input': ['input_data']},
            outputs={'Output': ['quantize_output']},
            attrs={
                'is_negative_input': is_negative_input,
                'Scale': scale,
                'Shift': shift,
            },
        )

        transpose2_op_1 = OpConfig(
            type='transpose2',
            inputs={'X': ['quantize_output']},
            outputs={
                'Out': ['transpose2_output_1'],
                'XShape': ['transpose2_xshape'],
            },
            attrs={
                'axis': axis,
                'use_mkldnn': True,
                'mkldnn_data_type': 'int8',
            },
            use_mkldnn=True,
        )

        transpose2_op_2 = OpConfig(
            type='transpose2',
            inputs={'X': ['transpose2_output_1']},
            outputs={
                'Out': ['transpose2_output_2'],
                'XShape': ['transpose2_xshape'],
            },
            attrs={
                'axis': axis,
                'use_mkldnn': True,
                'mkldnn_data_type': 'int8',
            },
            use_mkldnn=True,
        )

        dequantize_op = OpConfig(
            type='dequantize',
            inputs={'Input': ['transpose2_output_2']},
            outputs={'Output': ['dequantize_output']},
            attrs={
                'Scale': scale,
                'Shift': shift,
            },
        )

        program_config = ProgramConfig(
            ops=[quantize_op, transpose2_op_1, transpose2_op_2, dequantize_op],
            weights={},
            inputs={
                'input_data': TensorConfig(data_gen=partial(generate_input))
            },
            outputs=['dequantize_output'],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=['quant_transpose2_dequant_onednn_fuse_pass'],
        )
        yield config, ['fused_transpose', 'fused_transpose'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=['quant_transpose2_dequant_onednn_fuse_pass']
        )


if __name__ == '__main__':
    unittest.main()
