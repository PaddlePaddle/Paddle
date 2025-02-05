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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestFCActivationOneDNNFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        activation_type = draw(
            st.sampled_from(
                [
                    'relu',
                    'gelu',
                    'swish',
                    'mish',
                    'sqrt',
                    'hard_swish',
                    'sigmoid',
                    'abs',
                    'relu6',
                    'clip',
                    'tanh',
                    'hard_sigmoid',
                    'leaky_relu',
                    'scale',
                ]
            )
        )

        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        fc_op = OpConfig(
            type="fc",
            inputs={
                "Input": ["fc_input"],
                "W": ["fc_weight"],
                "Bias": ["fc_bias"],
            },
            outputs={"Out": ["fc_output"]},
            attrs={
                "use_mkldnn": True,
                "padding_weights": False,
                "in_num_col_dims": 1,
            },
        )

        if activation_type == "clip":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                min=draw(st.floats(min_value=0.1, max_value=0.49)),
                max=draw(st.floats(min_value=0.5, max_value=1.0)),
            )
        elif activation_type == "gelu":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                approximate=draw(st.booleans()),
            )
        elif activation_type == "leaky_relu":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                alpha=draw(st.floats(min_value=0.1, max_value=1.0)),
            )
        elif activation_type == "relu6":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                threshold=6,
            )
        elif activation_type == "scale":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                scale=draw(st.sampled_from([0.125, 0.4, 0.875, 2])),
            )
        elif activation_type == "swish":
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
                beta=1.0,
            )
        else:
            activation_op = OpConfig(
                activation_type,
                inputs={"X": ["fc_output"]},
                outputs={"Out": ["activation_output"]},
            )

        model_net = [fc_op, activation_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={
                "fc_weight": TensorConfig(
                    data_gen=partial(generate_input, [64, 64])
                ),
                "fc_bias": TensorConfig(data_gen=partial(generate_input, [64])),
            },
            inputs={
                "fc_input": TensorConfig(
                    data_gen=partial(generate_input, [32, 64])
                )
            },
            outputs=["activation_output"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True,
            passes=[
                "fc_act_onednn_fuse_pass",
                "operator_scale_onednn_fuse_pass",
            ],
        )
        yield config, ["fc"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=[
                "fc_act_onednn_fuse_pass",
                "operator_scale_onednn_fuse_pass",
            ],
        )


if __name__ == "__main__":
    unittest.main()
