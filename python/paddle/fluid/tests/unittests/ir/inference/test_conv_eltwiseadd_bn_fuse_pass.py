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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvEltwiseaddBnFusePass(PassAutoScanTest):
    """
          x_var   f_var(persistable)
            \       /
                conv2d
                |
                conv2d_var    bias_var(persistable)
                    \          /
                elementwise_add
                        |
                elementwise_add_var Scale(persistable) Bias(persistable) Mean(persistable) Variance(persistable)
                        |
                    batch_norm
                        |
     Y  MeanOut VarianceOut  SavedMeanSavedVariance
    """

    def sample_predictor_configs(self, program_config):
        # cpu
        config = self.create_inference_config(use_gpu=False)
        yield config, ["conv2d", "elementwise_add"], (1e-4, 1e-5)

        # MKLDNN
        config = self.create_inference_config(use_gpu=False)
        config.enable_mkldnn()
        yield config, ["conv2d", "elementwise_add"], (1e-4, 1e-5)

        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["conv2d", "elementwise_add"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        paddings = prog_config.ops[0].attrs["paddings"]
        strides = prog_config.ops[0].attrs["strides"]
        groups = prog_config.ops[0].attrs["groups"]
        padding_algorithm = prog_config.ops[0].attrs["padding_algorithm"]
        dilations = prog_config.ops[0].attrs["dilations"]
        data_format = prog_config.ops[0].attrs["data_format"]
        filter_shape = prog_config.weights["filter"].shape
        input_shape = prog_config.inputs["input_x"].shape
        if data_format != "NCHW":
            return False
        if padding_algorithm == "VALID":
            if ((input_shape[2] - (dilations[0] * (filter_shape[2] - 1) + 1)) / strides[0] + 1) <= 1 or \
            ((input_shape[3] - (dilations[1] * (filter_shape[3] - 1) + 1)) / strides[1] + 1) <= 1:
                return False
        if padding_algorithm == "EXPLICIT":
            if ((input_shape[2] + paddings[0] + paddings[1] - (dilations[0] * (filter_shape[2] - 1) + 1)) / strides[0] + 1) <= 1 or \
                ((input_shape[3] + paddings[2] + paddings[3] - (dilations[1] * (filter_shape[3] - 1) + 1)) / strides[1] + 1) <= 1:
                return False

        if data_format == "NCHW":
            if input_shape[1] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False
        else:
            if input_shape[3] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False

        bn_scale = np.array(prog_config.weights["scale_in"].data)
        bn_bias = np.array(prog_config.weights["bias_in"].data)
        bn_mean = np.array(prog_config.weights["mean_in"].data)
        bn_variance = np.array(prog_config.weights["variance_in"].data)
        epsilon = np.array(prog_config.ops[-1].attrs["epsilon"])
        bn_variance = bn_variance + epsilon

        if np.isnan(bn_variance).any():
            return False
        bn_variance = np.sqrt(bn_variance)
        if np.sum(bn_variance == 0.0) > 0:
            return False
        bn_variance = bn_scale / bn_variance
        if np.isnan(bn_variance).any():
            return False
        return True

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X of conv2d
        x_shape = draw(
            st.lists(st.integers(min_value=10, max_value=100),
                     min_size=4,
                     max_size=4))
        x_shape[1] = draw(st.integers(min_value=1, max_value=10))

        # 2. Generate legal attr:data_format of conv2d
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        # 2. Generate legal shape of input:Y of conv2d
        f_shape = draw(
            st.lists(st.integers(min_value=1, max_value=7),
                     min_size=4,
                     max_size=4))
        if data_format == "NCHW":
            f_shape[1] = x_shape[1]
        else:
            f_shape[1] = x_shape[3]

        # 3. Generate legal attr:strides of conv2d
        strides = draw(
            st.lists(st.integers(min_value=1, max_value=5),
                     min_size=2,
                     max_size=2))

        # 4. Generate legal attr:padding_algorithm of conv2d
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))

        # 5. Generate legal attr:padding of conv2d
        padding = draw(
            st.lists(st.integers(min_value=1, max_value=5),
                     min_size=4,
                     max_size=4))

        # 6. Generate legal attr:groups of conv2d
        groups = draw(st.integers(min_value=1, max_value=3))

        # 7. Generate legal attr:dilations of conv2d
        dilations = draw(
            st.lists(st.integers(min_value=1, max_value=5),
                     min_size=2,
                     max_size=2))

        # 9. Generate legal input:ResidualData of conv2d
        res_shape = []
        if draw(st.booleans()):
            res_shape = draw(
                st.lists(st.integers(min_value=1, max_value=100),
                         min_size=4,
                         max_size=4))

        # 10. Generate legal shape of input:bias of elementwise_add
        bias_shape = [f_shape[0]]

        # 11. Generate legal attr:axis of elementwise_add
        axis = 1

        # 12. Generate legal input:Scale of batch_norm
        bn_scale_shape = [f_shape[0]]

        # 13. Generate legal input:Bias of batch_norm
        bn_bias_shape = [f_shape[0]]

        # 14. Generate legal input:Mean of batch_norm
        bn_mean_shape = [f_shape[0]]

        # 15. Generate legal input:Variance of batch_norm
        bn_variance_shape = [f_shape[0]]

        # 16. Generate legal attr:epsilon of batch_norm
        epsilon = draw(st.floats(min_value=0.00001, max_value=0.001))

        def generate_batch_variance():
            return (0.1 +
                    (1.0 - 0.1) * np.random.random(bn_variance_shape)).astype(
                        np.float32)

        conv2d_op = OpConfig("conv2d",
                             inputs={
                                 "Input": ["input_x"],
                                 "Filter": ["filter"],
                                 "ResidualData": ["residualdata"]
                             },
                             outputs={"Output": ["conv2d_out"]},
                             strides=strides,
                             padding_algorithm=padding_algorithm,
                             paddings=padding,
                             groups=groups,
                             dilations=dilations,
                             data_format=data_format)
        add_op = OpConfig("elementwise_add",
                          inputs={
                              "X": ["conv2d_out"],
                              "Y": ["bias"]
                          },
                          outputs={"Out": ["add_out"]},
                          axis=axis)

        bn_op = OpConfig("batch_norm",
                         inputs={
                             "X": ["add_out"],
                             "Scale": ["scale_in"],
                             "Bias": ["bias_in"],
                             "Mean": ["mean_in"],
                             "Variance": ["variance_in"]
                         },
                         outputs={
                             "Y": ["y_out"],
                             "MeanOut": ["mean_in"],
                             "VarianceOut": ["variance_in"],
                             "SavedMean": ["SavedMean_out"],
                             "SavedVariance": ["SavedVariance_out"],
                             "ReserveSpace": ["ReserveSpace_out"]
                         },
                         epsilon=epsilon,
                         is_test=True,
                         trainable_statistics=False,
                         data_layout=data_format)

        ops = [conv2d_op, add_op, bn_op]

        # 17. if the output of bias is more than one
        if draw(st.booleans()):
            outputs = ops[-1].outputs["Y"]
        else:
            outputs = ops[-1].outputs["Y"] + ["bias"]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "filter": TensorConfig(shape=f_shape),
                "bias": TensorConfig(shape=bias_shape),
                "scale_in": TensorConfig(shape=bn_scale_shape),
                "bias_in": TensorConfig(shape=bn_bias_shape),
                "mean_in": TensorConfig(shape=bn_mean_shape),
                "variance_in": TensorConfig(data_gen=generate_batch_variance),
            },
            inputs={
                "input_x": TensorConfig(shape=x_shape),
                "residualdata": TensorConfig(shape=res_shape)
            },
            outputs=outputs)
        return program_config

    def test(self):
        self.run_and_statis(quant=False,
                            max_examples=300,
                            passes=["conv_eltwiseadd_bn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
