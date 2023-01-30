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


class TestMKLDNNShuffleChannelOp(MkldnnAutoScanTest):
=======
from auto_scan_test import MkldnnAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
from functools import partial
import unittest
from hypothesis import given
import hypothesis.strategies as st


class TestMKLDNNShuffleChannelOp(MkldnnAutoScanTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
<<<<<<< HEAD
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        shuffle_channel_op = OpConfig(
            type="shuffle_channel",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"group": kwargs['group']},
        )
=======

        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        shuffle_channel_op = OpConfig(type="shuffle_channel",
                                      inputs={"X": ["input_data"]},
                                      outputs={"Out": ["output_data"]},
                                      attrs={"group": kwargs['group']})
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        program_config = ProgramConfig(
            ops=[shuffle_channel_op],
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
        group=st.sampled_from([1, 2, 8, 32, 128]),
        in_shape=st.sampled_from([[5, 512, 2, 3], [2, 256, 5, 4]]),
    )
=======
    @given(group=st.sampled_from([1, 2, 8, 32, 128]),
           in_shape=st.sampled_from([[5, 512, 2, 3], [2, 256, 5, 4]]))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
