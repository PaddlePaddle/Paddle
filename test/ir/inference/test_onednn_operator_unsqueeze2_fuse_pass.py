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


class TestTranspose2Unsqueeze2OneDNNFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        channel = draw(st.sampled_from([1, 2, 4]))
        transpose_axis = draw(
            st.sampled_from([[0, 1, 2, 3], [2, 1, 3, 0], [3, 2, 1, 0]])
        )
        unsqueeze_axes = draw(st.sampled_from([[0, 1], [0, 4], [1, 2], [3]]))

        transpose2_op = OpConfig(
            type="transpose2",
            inputs={
                "X": ["transpose_x"],
            },
            outputs={
                "Out": ["transpose_out"],
                "XShape": ['transpose2_xshape'],
            },
            attrs={
                "axis": transpose_axis,
                "use_mkldnn": True,
            },
        )

        unsqueeze2_op = OpConfig(
            type="unsqueeze2",
            inputs={"X": ["transpose_out"]},
            outputs={"Out": ["unsqueeze_out"]},
            attrs={
                "axes": unsqueeze_axes,
            },
        )

        model_net = [transpose2_op, unsqueeze2_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                "transpose_x": TensorConfig(
                    data_gen=partial(generate_input, [channel, 16, 64, 32])
                )
            },
            outputs=["unsqueeze_out"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                "operator_unsqueeze2_onednn_fuse_pass",
            ],
        )
        yield config, ["fused_transpose"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=[
                "operator_unsqueeze2_onednn_fuse_pass",
            ],
        )


class TestElementwiseMulUnsqueeze2OneDNNFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        batch_size = draw(st.sampled_from([1, 2, 4]))
        channel = draw(st.sampled_from([1, 3, 16]))
        unsqueeze_axes = draw(st.sampled_from([[0, 1], [0, 4], [1, 2], [3]]))

        elementwise_op = OpConfig(
            type='elementwise_mul',
            inputs={'X': ['eltwise_X'], 'Y': ['eltwise_Y']},
            outputs={'Out': ['eltwise_output']},
            attrs={"use_mkldnn": True},
        )

        unsqueeze2_op = OpConfig(
            type="unsqueeze2",
            inputs={"X": ["eltwise_output"]},
            outputs={"Out": ["unsqueeze_out"]},
            attrs={
                "axes": unsqueeze_axes,
            },
        )

        model_net = [elementwise_op, unsqueeze2_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={},
            inputs={
                "eltwise_X": TensorConfig(
                    data_gen=partial(
                        generate_input, [batch_size, channel, 100, 100]
                    )
                ),
                "eltwise_Y": TensorConfig(
                    data_gen=partial(
                        generate_input, [batch_size, channel, 100, 100]
                    )
                ),
            },
            outputs=["unsqueeze_out"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                "operator_unsqueeze2_onednn_fuse_pass",
            ],
        )
        yield config, ["fused_elementwise_mul"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=[
                "operator_unsqueeze2_onednn_fuse_pass",
            ],
        )


if __name__ == "__main__":
    unittest.main()
