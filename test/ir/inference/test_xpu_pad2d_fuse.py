# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Unsupported on XPU3",
)
class TestXpuUnSqueezPad3dUnsqueezeFusePass2(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["pad2d_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        x_shape = draw(st.sampled_from([[6, 6, 6, 6]]))
        # x_shape = draw(st.sampled_from([[1, 1, 3, 3]]))

        # 1.unsqeeze
        axes = [2]
        # 2.pad3d
        data_format = draw(st.sampled_from(['NCDHW', 'NDHWC']))
        value = 0.0
        mode = draw(st.sampled_from(['constant', 'reflect', 'replicate']))
        paddings = draw(
            st.sampled_from(
                [
                    [1, 1, 1, 1, 0, 0],
                    [2, 2, 2, 2, 0, 0],
                    [0, 1, 2, 3, 0, 0],
                    [4, 3, 2, 1, 0, 0],
                    [2, 3, 4, 5, 0, 0],
                    [1, 2, 2, 1, 0, 0],
                    [2, 0, 1, 1, 0, 0],
                    [1, 1, 2, 0, 0, 0],
                ]
            )
        )

        if data_format == 'NDHWC':
            axes = [1]

        unsqueeze_op = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["unsqueeze_input"],
            },
            outputs={"Out": ["unsqueeze_out"]},
            axes=axes,
        )
        pad3d_op = OpConfig(
            "pad3d",
            inputs={
                "X": ["unsqueeze_out"],
            },
            outputs={"Out": ["pad3d_out"]},
            attrs={
                "paddings": paddings,
                "mode": mode,
                "pad_value": value,
                "data_format": data_format,
            },
        )
        squeeze_op = OpConfig(
            "squeeze2",
            inputs={
                "X": ["pad3d_out"],
            },
            outputs={"Out": ["squeeze_out"]},
            axes=axes,
        )

        ops = [
            unsqueeze_op,
            pad3d_op,
            squeeze_op,
        ]

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "unsqueeze_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={},
            outputs=["squeeze_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            min_success_num=1,
            passes=["pad2d_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
