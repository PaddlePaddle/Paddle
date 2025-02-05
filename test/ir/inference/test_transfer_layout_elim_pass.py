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

import os
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import CutlassAutoScanTest, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


class TestTransferElimPass0(PassAutoScanTest):
    r"""input0                    input1
           |                         |
     transfer_layout         transfer_layout
           |                       |
    transfer_layout_out0    transfer_layout_out1
                  \          /
                elementwise_add
                       |
                elementwise_add_out

    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["elementwise_add", "transfer_layout"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        transfer_layout0 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input0"]},
            outputs={"Out": ["transfer_layout_out0"]},
            dst_layout=1,
            src_layout=2,
        )
        transfer_layout1 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input1"]},
            outputs={"Out": ["transfer_layout_out1"]},
            dst_layout=1,
            src_layout=2,
        )

        add_op = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["transfer_layout_out0"],
                "Y": ["transfer_layout_out1"],
            },
            outputs={"Out": ["elementwise_add_out"]},
            axis=-1,
        )

        ops = [transfer_layout0, transfer_layout1, add_op]
        x_shape = draw(
            st.lists(
                st.integers(min_value=10, max_value=100), min_size=4, max_size=4
            )
        )
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input0": TensorConfig(shape=x_shape),
                "input1": TensorConfig(shape=x_shape),
            },
            outputs=["elementwise_add_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=30,
            passes=["transfer_layout_elim_pass"],
        )


class TestTransferElimPass1(PassAutoScanTest):
    r"""input0                    input1
           |                         |
     transfer_layout         transfer_layout
           |                       |
    transfer_layout_out0    transfer_layout_out1
                  \          /
                elementwise_add
                       |
                elementwise_add_out
                       |
                transfer_layout
                       |
                transfer_layout2
    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["elementwise_add"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        transfer_layout0 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input0"]},
            outputs={"Out": ["transfer_layout_out0"]},
            dst_layout=1,
            src_layout=2,
        )
        transfer_layout1 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input1"]},
            outputs={"Out": ["transfer_layout_out1"]},
            dst_layout=1,
            src_layout=2,
        )

        add_op = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["transfer_layout_out0"],
                "Y": ["transfer_layout_out1"],
            },
            outputs={"Out": ["elementwise_add_out"]},
            axis=-1,
        )

        transfer_layout2 = OpConfig(
            "transfer_layout",
            inputs={"X": ["elementwise_add_out"]},
            outputs={"Out": ["transfer_layout_out2"]},
            dst_layout=2,
            src_layout=1,
        )
        ops = [transfer_layout0, transfer_layout1, add_op, transfer_layout2]
        x_shape = draw(
            st.lists(
                st.integers(min_value=10, max_value=100), min_size=4, max_size=4
            )
        )
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input0": TensorConfig(shape=x_shape),
                "input1": TensorConfig(shape=x_shape),
            },
            outputs=["transfer_layout_out2"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=30,
            passes=["transfer_layout_elim_pass"],
        )


class TestTransferElimPass2(PassAutoScanTest):
    r"""input0                    input1
           |                         |
     transfer_layout         transfer_layout
           |                       |
    transfer_layout_out0    transfer_layout_out1
                  \          /
                     concat
                       |
                   concat_out

    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["concat", "transfer_layout"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        # nhwc -> nchw

        transfer_layout0 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input0"]},
            outputs={"Out": ["transfer_layout_out0"]},
            dst_layout=1,
            src_layout=2,
        )
        transfer_layout1 = OpConfig(
            "transfer_layout",
            inputs={"X": ["input1"]},
            outputs={"Out": ["transfer_layout_out1"]},
            dst_layout=1,
            src_layout=2,
        )

        concat_op = OpConfig(
            "concat",
            inputs={"X": ["transfer_layout_out0", "transfer_layout_out1"]},
            outputs={"Out": ["concat_out"]},
            axis=1,
        )

        ops = [transfer_layout0, transfer_layout1, concat_op]
        x_shape = draw(
            st.lists(
                st.integers(min_value=10, max_value=100), min_size=4, max_size=4
            )
        )
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input0": TensorConfig(shape=x_shape),
                "input1": TensorConfig(shape=x_shape),
            },
            outputs=["concat_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=30,
            passes=["transfer_layout_elim_pass"],
        )


class TestTransferElimPass3(CutlassAutoScanTest):
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(input_shape):
            return (np.random.random(input_shape) - 0.5).astype(np.float32)

        # src_layout should be NCHW, because it is the model's input
        for dst_layout, src_layout in [[1, 2]]:
            for axis in [0, 1, 2, 3]:
                ops_config = [
                    {
                        "op_type": "transfer_layout",
                        "op_inputs": {"X": ["input0"]},
                        "op_outputs": {"Out": ["transfer_layout_out0"]},
                        "op_attrs": {
                            "dst_layout": dst_layout,
                            "src_layout": src_layout,
                        },
                    },
                    {
                        "op_type": "transfer_layout",
                        "op_inputs": {"X": ["input1"]},
                        "op_outputs": {"Out": ["transfer_layout_out1"]},
                        "op_attrs": {
                            "dst_layout": dst_layout,
                            "src_layout": src_layout,
                        },
                        # nchw -> nhwc
                    },
                    {
                        "op_type": "concat",
                        "op_inputs": {
                            "X": [
                                "transfer_layout_out0",
                                "transfer_layout_out1",
                            ]
                        },
                        "op_outputs": {"Out": ["concat_out0"]},
                        "op_attrs": {"axis": axis},
                    },
                ]

                ops = self.generate_op_config(ops_config)
                input_shape = [12, 13, 14, 15]

                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "input0": TensorConfig(
                            data_gen=partial(generate_input, input_shape)
                        ),
                        "input1": TensorConfig(
                            data_gen=partial(generate_input, input_shape)
                        ),
                    },
                    outputs=["concat_out0"],
                )

                yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        config.enable_use_gpu(256, 0)
        yield config, (1e-2, 1e-2)

    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
