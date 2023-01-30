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

<<<<<<< HEAD
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestSeqconvEltaddReluFusePass(PassAutoScanTest):
=======
from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
from functools import reduce


class TestSeqconvEltaddReluFusePass(PassAutoScanTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        contextLength = draw(st.sampled_from([1, 2, 3, 4]))
        contextStart = draw(st.sampled_from([1, 2, 3]))
        contextStride = draw(st.sampled_from([1]))
        paddingTrainable = False
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            shape = [batch_size, 128, 6, 120]
            return np.random.random(shape).astype(np.float32)

        def generate_weight(shape):
            return np.random.random(shape).astype(np.float32)

<<<<<<< HEAD
        im2sequence_op = OpConfig(
            type="im2sequence",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["seq_out"]},
            attrs={
                "kernels": [6, 1],
                "out_stride": [1, 1],
                "paddings": [0, 0, 0, 0],
                "strides": [1, 1],
            },
        )

        sequence_conv_op = OpConfig(
            type="sequence_conv",
            inputs={"X": ["seq_out"], "Filter": ["conv_weight"]},
            outputs={"Out": ["conv_out"]},
            attrs={
                "contextLength": contextLength,
                "contextStart": contextStart,
                "contextStride": contextStride,
                "paddingTrainable": paddingTrainable,
            },
        )

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["conv_out"], "Y": ["elt_weight"]},
            outputs={"Out": ["elt_output"]},
            attrs={'axis': axis},
        )

        relu_op = OpConfig(
            type="relu",
            inputs={"X": ["elt_output"]},
            outputs={"Out": ["relu_output"]},
            attrs={},
        )

        model_net = [
            im2sequence_op,
            sequence_conv_op,
            elementwise_add_op,
            relu_op,
=======
        im2sequence_op = OpConfig(type="im2sequence",
                                  inputs={"X": ["input_data"]},
                                  outputs={"Out": ["seq_out"]},
                                  attrs={
                                      "kernels": [6, 1],
                                      "out_stride": [1, 1],
                                      "paddings": [0, 0, 0, 0],
                                      "strides": [1, 1]
                                  })

        sequence_conv_op = OpConfig(type="sequence_conv",
                                    inputs={
                                        "X": ["seq_out"],
                                        "Filter": ["conv_weight"]
                                    },
                                    outputs={"Out": ["conv_out"]},
                                    attrs={
                                        "contextLength": contextLength,
                                        "contextStart": contextStart,
                                        "contextStride": contextStride,
                                        "paddingTrainable": paddingTrainable
                                    })

        elementwise_add_op = OpConfig(type="elementwise_add",
                                      inputs={
                                          "X": ["conv_out"],
                                          "Y": ["elt_weight"]
                                      },
                                      outputs={"Out": ["elt_output"]},
                                      attrs={'axis': axis})

        relu_op = OpConfig(type="relu",
                           inputs={"X": ["elt_output"]},
                           outputs={"Out": ["relu_output"]},
                           attrs={})

        model_net = [
            im2sequence_op, sequence_conv_op, elementwise_add_op, relu_op
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]

        program_config = ProgramConfig(
            ops=model_net,
            weights={
<<<<<<< HEAD
                "conv_weight": TensorConfig(
                    data_gen=partial(generate_weight, [768 * contextLength, 16])
                ),
                "elt_weight": TensorConfig(
                    data_gen=partial(generate_weight, [16])
                ),
=======
                "conv_weight":
                TensorConfig(
                    data_gen=partial(generate_weight, [768 *
                                                       contextLength, 16])),
                "elt_weight":
                TensorConfig(data_gen=partial(generate_weight, [16]))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
<<<<<<< HEAD
            outputs=["relu_output"],
        )
=======
            outputs=["relu_output"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
<<<<<<< HEAD
        yield config, ["im2sequence", "fusion_seqconv_eltadd_relu"], (
            1e-5,
            1e-5,
        )

    def test(self):
        self.run_and_statis(
            quant=False, passes=["seqconv_eltadd_relu_fuse_pass"]
        )
=======
        yield config, ["im2sequence",
                       "fusion_seqconv_eltadd_relu"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False,
                            passes=["seqconv_eltadd_relu_fuse_pass"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
