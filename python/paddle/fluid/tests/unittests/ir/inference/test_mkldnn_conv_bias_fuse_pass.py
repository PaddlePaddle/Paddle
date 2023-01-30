# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from inference_pass_test import InferencePassTest

import paddle
import paddle.fluid as fluid
from paddle.fluid.core import PassVersionChecker


# padding SAME
class ConvBiasMkldnnFusePassSamePadTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32"
            )
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                padding="SAME",
                bias_attr=param_attr,
            )
=======
from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import PassVersionChecker


#padding SAME
class ConvBiasMkldnnFusePassSamePadTest(InferencePassTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           padding="SAME",
                                           bias_attr=param_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)
        self.assertTrue(
<<<<<<< HEAD
            PassVersionChecker.IsCompatible("conv_bias_mkldnn_fuse_pass")
        )


# padding VALID
class ConvBiasMkldnnFusePassValidPadTest(ConvBiasMkldnnFusePassSamePadTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32"
            )
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                padding="VALID",
                bias_attr=param_attr,
            )
=======
            PassVersionChecker.IsCompatible("conv_bias_mkldnn_fuse_pass"))


#padding VALID
class ConvBiasMkldnnFusePassValidPadTest(ConvBiasMkldnnFusePassSamePadTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           padding="VALID",
                                           bias_attr=param_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True


<<<<<<< HEAD
# padding EXPLICT NUMBER
class ConvBiasMkldnnFusePassExplictPadTest(ConvBiasMkldnnFusePassSamePadTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32"
            )
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                padding=[2, 4, 6, 8],
                bias_attr=param_attr,
            )
=======
#padding EXPLICT NUMBER
class ConvBiasMkldnnFusePassExplictPadTest(ConvBiasMkldnnFusePassSamePadTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           padding=[2, 4, 6, 8],
                                           bias_attr=param_attr)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True


class ConvBiasMkldnnFusePassGroupTest(ConvBiasMkldnnFusePassSamePadTest):
<<<<<<< HEAD
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32"
            )
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                padding="VALID",
                groups=3,
                bias_attr=param_attr,
                use_cudnn=False,
                act="softmax",
                data_format="NCHW",
            )
=======

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           padding="VALID",
                                           groups=3,
                                           bias_attr=param_attr,
                                           use_cudnn=False,
                                           act="softmax",
                                           data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True


class ConvBiasMkldnnFusePassDialtionsGroupsTest(
<<<<<<< HEAD
    ConvBiasMkldnnFusePassSamePadTest
):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32"
            )
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                padding="VALID",
                dilation=2,
                groups=3,
                bias_attr=param_attr,
                use_cudnn=False,
                act="softmax",
                data_format="NCHW",
            )
=======
        ConvBiasMkldnnFusePassSamePadTest):

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data",
                              shape=[-1, 3, 100, 100],
                              dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d(input=data,
                                           num_filters=3,
                                           filter_size=3,
                                           padding="VALID",
                                           dilation=2,
                                           groups=3,
                                           bias_attr=param_attr,
                                           use_cudnn=False,
                                           act="softmax",
                                           data_format="NCHW")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True


class ConvTransposeMkldnnFusePassDialtionsGroupsTest(InferencePassTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(name="data", shape=[-1, 3, 5, 5], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False),
<<<<<<< HEAD
                learning_rate=0.001,
            )
            conv_out = paddle.static.nn.conv2d_transpose(
                input=data,
                num_filters=3,
                filter_size=3,
                padding="SAME",
                dilation=1,
                bias_attr=param_attr,
                use_cudnn=False,
            )
=======
                learning_rate=0.001)
            conv_out = fluid.layers.conv2d_transpose(input=data,
                                                     num_filters=3,
                                                     filter_size=3,
                                                     padding="SAME",
                                                     dilation=1,
                                                     bias_attr=param_attr,
                                                     use_cudnn=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.feeds = {"data": np.random.random((1, 3, 5, 5)).astype("float32")}
        self.fetch_list = [conv_out]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)
        self.assertTrue(
            PassVersionChecker.IsCompatible(
<<<<<<< HEAD
                "conv_transpose_bias_mkldnn_fuse_pass"
            )
        )
=======
                "conv_transpose_bias_mkldnn_fuse_pass"))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
