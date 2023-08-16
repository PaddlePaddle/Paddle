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


class TestSqueeze2Transpose2OneDNNFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        channel = draw(st.sampled_from([1, 2, 4, 8, 16]))
        transpose_axis = draw(
            st.sampled_from(
                [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 1, 0], [2, 1, 0]]
            )
        )

        squeeze2_op = OpConfig(
            type="squeeze2",
            inputs={"X": ["squeeze_x"]},
            outputs={
                "Out": ["squeeze_out"],
                "XShape": ["squeeze2_xshape"],
            },
            attrs={
                "axes": [2],
                "use_mkldnn": True,
            },
        )

        transpose2_op = OpConfig(
            type="transpose2",
            inputs={
                "X": ["squeeze_out"],
            },
            outputs={
                "Out": ["trans_out"],
                "XShape": ['transpose2_xshape'],
            },
            attrs={
                "axis": transpose_axis,
                "use_mkldnn": True,
            },
        )

        model_net = [squeeze2_op, transpose2_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                "squeeze_x": TensorConfig(
                    data_gen=partial(generate_input, [channel, 16, 1, 32])
                )
            },
            outputs=["trans_out"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                "squeeze2_transpose2_onednn_fuse_pass",
            ],
        )
        yield config, ["fused_transpose"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=[
                "squeeze2_transpose2_onednn_fuse_pass",
            ],
        )


if __name__ == "__main__":
    unittest.main()
