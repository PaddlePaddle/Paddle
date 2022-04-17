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

from auto_scan_test import PassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestShuffleChannelDetectPass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if attrs[0]['input_shape'] != attrs[2]['shape']:
            return False

        return True

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        out_channel = draw(st.integers(min_value=1, max_value=16))
        group = draw(st.integers(min_value=1, max_value=4))
        in_channel = group * out_channel
        x_shape = [batch_size, in_channel, 64, 64]
        shape = [0, group, out_channel, -1, 64]
        axis_v = [0, 2, 1, 3, 4]

        def generate_reshape2_Input():
            return np.random.random(x_shape).astype(np.float32)

        reshape2_op1 = OpConfig(
            "reshape2",
            inputs={"X": ["reshape2_input1"], },
            outputs={
                "Out": ["reshape2_output1"],
                "XShape": ["reshape2_xshape1"]
            },
            shape=shape,
            input_shape=x_shape)
        transpose2_op = OpConfig(
            "transpose2",
            inputs={"X": ["reshape2_output1"], },
            outputs={
                "Out": ["transpose2_ouput"],
                "XShape": ["transpose2_xshape"]
            },
            axis=axis_v)
        reshape2_op2 = OpConfig(
            "reshape2",
            inputs={"X": ["transpose2_ouput"], },
            outputs={
                "Out": ["reshape2_output2"],
                "XShape": ["reshape2_xshape2"]
            },
            shape=x_shape)
        ops = [reshape2_op1, transpose2_op, reshape2_op2]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "reshape2_input1":
                TensorConfig(data_gen=partial(generate_reshape2_Input)),
            },
            weights={},
            outputs=["reshape2_output2"])
        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=4,
            min_subgraph_size=1,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['shuffle_channel'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=["shuffle_channel_detect_pass"], )


if __name__ == "__main__":
    unittest.main()
