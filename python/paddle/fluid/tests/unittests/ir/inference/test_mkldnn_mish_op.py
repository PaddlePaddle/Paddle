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

<<<<<<< HEAD
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import MkldnnAutoScanTest
from hypothesis import given
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestMkldnnMishOp(MkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # if mode is channel, and in_shape is 1 rank
        if (
            len(program_config.inputs['input_data'].shape) == 1
            and program_config.ops[0].attrs['mode'] == 'channel'
        ):
=======
from auto_scan_test import MkldnnAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
from functools import partial
import unittest
from hypothesis import given
import hypothesis.strategies as st


class TestMkldnnMishOp(MkldnnAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # if mode is channel, and in_shape is 1 rank
        if len(program_config.inputs['input_data'].shape
               ) == 1 and program_config.ops[0].attrs['mode'] == 'channel':
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return False
        return True

    def sample_program_configs(self, *args, **kwargs):
<<<<<<< HEAD
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        mish_op = OpConfig(
            type="mish",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "mode": kwargs['mode'],
                "data_format": kwargs['data_format'],
            },
        )
=======

        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        mish_op = OpConfig(type="mish",
                           inputs={"X": ["input_data"]},
                           outputs={"Out": ["output_data"]},
                           attrs={
                               "mode": kwargs['mode'],
                               "data_format": kwargs['data_format']
                           })
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        program_config = ProgramConfig(
            ops=[mish_op],
            weights={},
            inputs={
<<<<<<< HEAD
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, *args, **kwargs)
                ),
            },
            outputs=["output_data"],
        )
=======
                "input_data":
                TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
            },
            outputs=["output_data"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

<<<<<<< HEAD
    @given(
        mode=st.sampled_from(['all', 'channel', 'element']),
        data_format=st.sampled_from(['NCHW', 'NHWC']),
        in_shape=st.lists(
            st.integers(min_value=1, max_value=32), min_size=1, max_size=4
        ),
    )
=======
    @given(mode=st.sampled_from(['all', 'channel', 'element']),
           data_format=st.sampled_from(['NCHW', 'NHWC']),
           in_shape=st.lists(st.integers(min_value=1, max_value=32),
                             min_size=1,
                             max_size=4))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
