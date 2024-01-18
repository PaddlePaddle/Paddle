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
from auto_scan_test import MkldnnAutoScanTest, PirMkldnnAutoScanTest
from hypothesis import given
from program_config import (
    OpConfig,
    ProgramConfig,
    TensorConfig,
)


class TestOneDNNPad3DOp(MkldnnAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_paddings():
            return np.random.randint(0, 4, size=(6)).astype(np.int32)

        pad3d_op = OpConfig(
            type="pad3d",
            inputs={"X": ["input_data"], "Paddings": ["paddings_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "mode": "constant",
                "data_format": kwargs['data_format'],
                "paddings": kwargs['paddings'],
            },
        )

        program_config = ProgramConfig(
            ops=[pad3d_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                ),
                "paddings_data": TensorConfig(data_gen=generate_paddings),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        data_format=st.sampled_from(['NCDHW', 'NDHWC']),
        use_paddings_tensor=st.sampled_from([True, False]),
        in_shape=st.sampled_from(
            [[2, 3, 4, 5, 6], [1, 4, 1, 3, 2], [4, 3, 2, 1, 1], [1, 1, 1, 1, 1]]
        ),
        paddings=st.sampled_from(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 2, 1],
                [2, 5, 11, 3, 4, 3],
                [0, 5, 0, 1, 0, 2],
            ]
        ),
    )
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


class TestPirOneDNNPad3DOp(PirMkldnnAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_paddings():
            return np.random.randint(0, 4, size=(6)).astype(np.int32)

        pad3d_op = OpConfig(
            type="pad3d",
            inputs={"X": ["input_data"], "Paddings": ["paddings_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "mode": "constant",
                "data_format": kwargs['data_format'],
                "paddings": kwargs['paddings'],
                "use_mkldnn": True,
            },
        )

        program_config = ProgramConfig(
            ops=[pad3d_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                ),
                "paddings_data": TensorConfig(data_gen=generate_paddings),
            },
            outputs=["output_data"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        data_format=st.sampled_from(['NCDHW', 'NDHWC']),
        use_paddings_tensor=st.sampled_from([True, False]),
        in_shape=st.sampled_from(
            [[2, 3, 4, 5, 6], [1, 4, 1, 3, 2], [4, 3, 2, 1, 1], [1, 1, 1, 1, 1]]
        ),
        paddings=st.sampled_from(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 2, 0, 1, 2, 1],
                [2, 5, 11, 3, 4, 3],
                [0, 5, 0, 1, 0, 2],
            ]
        ),
    )
    def test_pir(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
