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

import numpy as np
from program_config import OpConfig
from test_xpu_multi_encoder_xpu_fuse_pass import TestMultiEncoderXPUFusePass


class TestMultiEncoderXPUFusePass(TestMultiEncoderXPUFusePass):
    def sample_program_config(self, draw):
        slice_op = OpConfig(
            "slice",
            inputs={"Input": ["ln_2_out"]},
            outputs={"Out": ["slice_out"]},
            axes=[1],
            decrease_axis=[1],
            starts=[0],
            ends=[1],
        )
        program_config = self.multi_encoder_xpu_program_config(draw)
        program_config.ops.append(slice_op)
        program_config.outputs = ["slice_out"]
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=2,
            min_success_num=2,
            passes=[
                "multi_encoder_xpu_fuse_pass",
                "multi_encoder_xpu_slice_fuse_pass",
            ],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
