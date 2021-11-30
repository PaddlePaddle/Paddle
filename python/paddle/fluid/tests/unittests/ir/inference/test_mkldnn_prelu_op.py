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

from auto_scan_test import MkldnnAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestMkldnnPreluOp(MkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # if mode is channel, and in_shape is 1 rank
        if len(program_config.inputs['input_data'].
               shape) == 1 and program_config.ops[0].attrs['mode'] == 'channel':
            return False
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_alpha(*args, **kwargs):
            if kwargs["mode"] == "all":
                return np.random.random(size=(1)).astype(np.float32)
            elif kwargs["mode"] == "channel":
                if len(kwargs['in_shape']) <= 1:
                    # not valid case, just return 0
                    return np.zeros((1)).astype(np.float32)
                if kwargs['data_format'] == 'NCHW':
                    return np.random.random(kwargs['in_shape'][1]).astype(
                        np.float32)
                else:
                    return np.random.random(kwargs['in_shape'][-1]).astype(
                        np.float32)
            else:
                if len(kwargs['in_shape']) <= 1:
                    # not valid case, just return 0
                    return np.zeros((1)).astype(np.float32)
                return np.random.random(kwargs['in_shape']).astype(np.float32)

        prelu_op = OpConfig(
            type="prelu",
            inputs={"X": ["input_data"],
                    "Alpha": ["alpha_weight"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "mode": kwargs['mode'],
                "data_format": kwargs['data_format']
            })

        program_config = ProgramConfig(
            ops=[prelu_op],
            weights={
                "alpha_weight":
                TensorConfig(data_gen=partial(generate_alpha, *args, **kwargs))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
            },
            outputs=["output_data"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    def add_skip_pass_case(self):
        pass

    @given(
        mode=st.sampled_from(['all', 'channel', 'element']),
        data_format=st.sampled_from(['NCHW', 'NHWC']),
        in_shape=st.lists(
            st.integers(
                min_value=1, max_value=32), min_size=1, max_size=4))
    def test(self, *args, **kwargs):
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
