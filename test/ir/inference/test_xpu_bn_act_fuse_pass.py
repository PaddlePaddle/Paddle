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


class TestXpuBNActFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["bn_act_xpu"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(["NCHW"]))
        n = draw(st.integers(min_value=1, max_value=64))
        c = draw(st.integers(min_value=1, max_value=64))
        h = draw(st.integers(min_value=1, max_value=64))
        w = draw(st.integers(min_value=1, max_value=64))
        epsilon = draw(st.floats(min_value=0.0, max_value=0.001))

        x_shape = [n, c, h, w]
        scale_shape = [c]
        bias_shape = [c]
        var_shape = [c]
        mean_shape = [c]

        bn_op = OpConfig(
            "batch_norm",
            inputs={
                "X": ["bn_input"],
                "Scale": ["batch_norm_Scale"],
                "Bias": ["batch_norm_Bias"],
                "Mean": ["batch_norm_Mean"],
                "Variance": ["batch_norm_Variance"],
            },
            outputs={
                "Y": ["batch_norm_Y"],
                "MeanOut": ["batch_norm_Mean"],
                "VarianceOut": ["batch_norm_Variance"],
                "SavedMean": ["batch_norm_SavedMean"],
                "SavedVariance": ["batch_norm_SavedVariance"],
                "ReserveSpace": ["batch_norm_ReserveSpace"],
            },
            epsilon=epsilon,
            trainable_statistics=False,
            data_layout=data_format,
            is_test=True,
        )

        relu_op = OpConfig(
            "relu",
            inputs={
                "X": ["batch_norm_Y"],
            },
            outputs={
                "Out": ["relu_out"],
            },
        )
        ops = [bn_op, relu_op]

        def generate_bn_Input():
            return np.random.random(x_shape).astype(np.float32)

        def generate_bn_Scale():
            return np.random.random(scale_shape).astype(np.float32)

        def generate_bn_Bias():
            return np.random.random(bias_shape).astype(np.float32)

        def generate_bn_Mean():
            return np.random.random(mean_shape).astype(np.float32)

        def generate_bn_Var():
            return np.random.random(var_shape).astype(np.float32)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "bn_input": TensorConfig(data_gen=partial(generate_bn_Input)),
            },
            weights={
                "batch_norm_Scale": TensorConfig(data_gen=generate_bn_Scale),
                "batch_norm_Bias": TensorConfig(data_gen=generate_bn_Bias),
                "batch_norm_Mean": TensorConfig(data_gen=generate_bn_Mean),
                "batch_norm_Variance": TensorConfig(data_gen=generate_bn_Var),
            },
            outputs=["relu_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["bn_act_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
