# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMkldnnConv3dOp(MkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            if kwargs["data_format"] == "NCDHW":
                return np.random.random(
                    [kwargs["batch_size"], 48, 64, 32, 64]
                ).astype(np.float32)
            else:
                return np.random.random(
                    [kwargs["batch_size"], 64, 32, 64, 48]
                ).astype(np.float32)

        def generate_weight(*args, **kwargs):
            return np.random.random(
                [16, int(48 / kwargs["groups"]), 3, 3, 3]
            ).astype(np.float32)

        conv3d_op = OpConfig(
            type="conv3d",
            inputs={"Input": ["input_data"], "Filter": ["conv_weight"]},
            outputs={"Output": ["conv_output"]},
            attrs={
                "data_format": kwargs["data_format"],
                "dilations": kwargs["dilations"],
                "padding_algorithm": kwargs["padding_algorithm"],
                "groups": kwargs["groups"],
                "paddings": kwargs["paddings"],
                "strides": kwargs["strides"],
                "is_test": True,
            },
        )

        program_config = ProgramConfig(
            ops=[conv3d_op],
            weights={
                "conv_weight": TensorConfig(
                    data_gen=partial(generate_weight, *args, **kwargs)
                )
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                )
            },
            outputs=["conv_output"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        data_format=st.sampled_from(["NCDHW", "NDHWC"]),
        dilations=st.sampled_from([[1, 2, 1]]),
        padding_algorithm=st.sampled_from(["EXPLICIT"]),
        groups=st.sampled_from([2]),
        paddings=st.sampled_from([[0, 3, 2]]),
        strides=st.sampled_from([[1, 2, 1]]),
        batch_size=st.integers(min_value=1, max_value=4),
    )
    def test(self, *args, **kwargs):
        self.run_test(*args, **kwargs)


class TestPirOneDNNPad3DOp(PirMkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            if kwargs["data_format"] == "NCDHW":
                return np.random.random(
                    [kwargs["batch_size"], 48, 64, 32, 64]
                ).astype(np.float32)
            else:
                return np.random.random(
                    [kwargs["batch_size"], 64, 32, 64, 48]
                ).astype(np.float32)

        def generate_weight(*args, **kwargs):
            return np.random.random(
                [16, int(48 / kwargs["groups"]), 3, 3, 3]
            ).astype(np.float32)

        conv3d_op = OpConfig(
            type="conv3d",
            inputs={"Input": ["input_data"], "Filter": ["conv_weight"]},
            outputs={"Output": ["conv_output"]},
            attrs={
                "data_format": kwargs["data_format"],
                "dilations": kwargs["dilations"],
                "padding_algorithm": kwargs["padding_algorithm"],
                "groups": kwargs["groups"],
                "paddings": kwargs["paddings"],
                "strides": kwargs["strides"],
                "is_test": True,
                "use_mkldnn": True,
            },
        )

        program_config = ProgramConfig(
            ops=[conv3d_op],
            weights={
                "conv_weight": TensorConfig(
                    data_gen=partial(generate_weight, *args, **kwargs)
                )
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                )
            },
            outputs=["conv_output"],
        )

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        data_format=st.sampled_from(["NCDHW", "NDHWC"]),
        dilations=st.sampled_from([[1, 2, 1]]),
        padding_algorithm=st.sampled_from(["EXPLICIT"]),
        groups=st.sampled_from([2]),
        paddings=st.sampled_from([[0, 3, 2]]),
        strides=st.sampled_from([[1, 2, 1]]),
        batch_size=st.integers(min_value=1, max_value=4),
    )
    def test(self, *args, **kwargs):
        self.run_test(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
