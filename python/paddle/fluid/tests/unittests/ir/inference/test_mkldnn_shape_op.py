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

from auto_scan_test import MkldnnAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
from functools import partial
import unittest
from hypothesis import given
import hypothesis.strategies as st


class TestMkldnnShapeOp(MkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(kwargs[
                'in_dtype'])

        shape_op = OpConfig(
            type="shape",
            inputs={"Input": ["input_data"]},
            outputs={"Out": ["output_data"]})

        program_config = ProgramConfig(
            ops=[shape_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input,
                                                            *args, **kwargs)),
            },
            outputs=["output_data"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        in_shape=st.lists(
            st.integers(
                min_value=1, max_value=3), min_size=1, max_size=9),
        in_dtype=st.sampled_from([np.float32, np.uint16, np.int8, np.uint8]))
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
