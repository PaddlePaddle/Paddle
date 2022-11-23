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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
import paddle.fluid as fluid
from paddle.fluid.core import PassVersionChecker


class SoftplusActivationReluOneDNNFusePassTest(InferencePassTest):
    fuse_alpha = None
    fuse_beta = None
    pass_name = 'softplus_activation_mkldnn_fuse_pass'

    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            softplus_out = fluid.layers.softplus(data)
            if self.fuse_beta is not None:
                activation_out = self.fuse_activation(softplus_out,
                                                      self.fuse_alpha,
                                                      self.fuse_beta)
            elif self.fuse_alpha is not None:
                activation_out = self.fuse_activation(softplus_out,
                                                      self.fuse_alpha)
            else:
                activation_out = self.fuse_activation(softplus_out)

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32"),
        }
        self.fetch_list = [activation_out]
        self.enable_mkldnn = True

    def set_params(self):
        self.fuse_activation = fluid.layers.relu

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)

    def test_pass_compatible(self):
        self.assertTrue(PassVersionChecker.IsCompatible(self.pass_name))


class SoftplusActivationTanhOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.tanh


class SoftplusActivationLeakyReluOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.leaky_relu
        self.fuse_alpha = 0.3


class SoftplusActivationSwishOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.swish
        self.fuse_alpha = 3


class SoftplusActivationHardSwishOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.hard_swish


class SoftplusActivationSqrtOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.hard_swish


class SoftplusActivationAbsOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.abs


class SoftplusActivationClipOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.clip
        self.fuse_alpha = 1.1
        self.fuse_beta = 5.2


class SoftplusActivationGeluErfOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.gelu


class SoftplusActivationGeluTanhOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.gelu
        self.fuse_alpha = True  # simulated "Approximate" attr


class SoftplusActivationRelu6OneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.relu6


class SoftplusActivationSigmoidOneDNNFusePassTest(
        SoftplusActivationReluOneDNNFusePassTest):

    def set_params(self):
        self.fuse_activation = fluid.layers.sigmoid


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
