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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import ProgramConfig, TensorConfig


def product(input):
    result = 1

    for value in input:
        result = result * value

    return result


class TestShuffleChannelMKLDNNDetectPass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        input_shape = program_config.inputs['input_data'].shape
        first_reshape2_shape = program_config.ops[0].attrs['shape']
        transpose2_axis = program_config.ops[1].attrs['axis']
        second_reshape2_shape = program_config.ops[2].attrs['shape']

        shape_prod = product(input_shape)
        img_h = input_shape[-2]
        img_w = input_shape[-1]

        if shape_prod != product(first_reshape2_shape) or shape_prod != product(
            second_reshape2_shape
        ):
            return False
        if (
            len(input_shape) != 4
            or len(first_reshape2_shape) != 5
            or len(second_reshape2_shape) != 4
        ):
            return False
        if transpose2_axis != [0, 2, 1, 3, 4]:
            return False
        if (
            first_reshape2_shape[-1] != img_w
            or first_reshape2_shape[-2] != img_h
        ):
            return False
        if (
            second_reshape2_shape[-1] != img_w
            or second_reshape2_shape[-2] != img_h
        ):
            return False

        return True

    def sample_program_config(self, draw):
        input_shape = draw(st.sampled_from([[128, 32, 32]]))
        first_reshape2_shape = draw(
            st.sampled_from([[2, 64, 32, 32], [8, 16, 32, 32]])
        )
        transpose2_axis = draw(st.sampled_from([[0, 2, 1, 3, 4], [0, 2, 1, 3]]))
        second_reshape2_shape = draw(
            st.sampled_from([[128, 32, 32], [128, 31, 32]])
        )
        batch_size = draw(st.integers(min_value=1, max_value=10))

        input_shape.insert(0, batch_size)
        first_reshape2_shape.insert(0, batch_size)
        second_reshape2_shape.insert(0, batch_size)

        def generate_input():
            return np.random.random(input_shape).astype(np.float32)

        ops_config = [
            {
                "op_type": "reshape2",
                "op_inputs": {"X": ["input_data"]},
                "op_outputs": {
                    "Out": ["first_reshape2_output"],
                    "XShape": ["first_reshape2_xshape"],
                },
                "op_attrs": {'shape': first_reshape2_shape},
            },
            {
                "op_type": "transpose2",
                "op_inputs": {"X": ["first_reshape2_output"]},
                "op_outputs": {
                    "Out": ["transpose2_output"],
                    "XShape": ["transpose2_xshape"],
                },
                "op_attrs": {'axis': transpose2_axis},
            },
            {
                "op_type": "reshape2",
                "op_inputs": {
                    "X": ["transpose2_output"],
                },
                "op_outputs": {
                    "Out": ["output_data"],
                    "XShape": ["second_reshape2_xshape"],
                },
                "op_attrs": {'shape': second_reshape2_shape},
            },
        ]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["shuffle_channel"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["shuffle_channel_onednn_detect_pass"]
        )


if __name__ == "__main__":
    unittest.main()
