#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import math
import unittest

import conv2d_utils
import numpy as np
import pool_utils
from test_utils import SingleOpTester

from paddle.cinn import framework


class OpTest_relu(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return np.maximum(X, np.zeros(X.shape).astype("float32"))

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[32]], [[32]], "relu", attrs)


class OpTest_relu6(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return np.minimum(
            np.maximum(X, np.zeros(np.array(X).shape).astype("float32")), 6
        )

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[32, 32]], [[32, 32]], "relu6", attrs)


class OpTest_conv2d_nchw(SingleOpTester):
    def init_testcase(self):
        self.input_size = [1, 3, 10, 10]
        self.groups = 1
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [2, f_c, 2, 2]
        assert np.mod(self.filter_size[0], self.groups) == 0
        self.data_format = "NCHW"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [2, 2]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_conv2d_nchw_1(SingleOpTester):
    def init_testcase(self):
        self.input_size = [1, 3, 224, 224]
        self.groups = 1
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [64, f_c, 7, 7]
        self.data_format = "NCHW"
        self.attrs = framework.NodeAttr()
        self.padding = [3, 3]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_conv2d_nchw_group(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 8, 10, 10]
        self.groups = 4
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]
        self.data_format = "NCHW"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_conv2d_nchw_depthwise(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 8, 10, 10]
        self.groups = 8
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]
        self.data_format = "NCHW"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_conv2d_nhwc_group(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 10, 10, 8]
        self.groups = 4
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [16, f_c, 7, 7]
        self.data_format = "NHWC"
        self.attrs = framework.NodeAttr()
        self.padding = [2, 2]
        self.stride = [2, 2]
        self.dilation = [2, 2]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_conv2d_nhwc_depthwise(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 10, 10, 8]
        self.groups = 8
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [16, f_c, 7, 7]
        self.data_format = "NHWC"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, False
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "conv2d",
            self.attrs,
            0,
            True,
        )


# test channel multiplier format
class OpTest_depthwise_conv2d_nchw(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 8, 10, 10]
        self.groups = self.input_size[1]
        assert np.mod(self.input_size[1], self.groups) == 0
        channel_multiplier = 1
        self.filter_size = [self.input_size[1], channel_multiplier, 7, 7]
        self.data_format = "NCHW"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, True
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "depthwise_conv2d",
            self.attrs,
            0,
            True,
        )


# test channel multiplier format
class OpTest_depthwise_conv2d_nhwc(SingleOpTester):
    def init_testcase(self):
        self.input_size = [2, 10, 10, 8]
        self.groups = self.input_size[3]
        assert np.mod(self.input_size[3], self.groups) == 0
        channel_multiplier = 4
        self.filter_size = [self.input_size[3], channel_multiplier, 7, 7]
        self.data_format = "NHWC"
        self.attrs = framework.NodeAttr()
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.attrs.set_attr("stride", self.stride)
        self.attrs.set_attr("padding", self.padding)
        self.attrs.set_attr("dilation", self.dilation)
        self.attrs.set_attr("groups", self.groups)
        self.attrs.set_attr("data_format", self.data_format)

    def create_target_data(self, inputs_data, attrs):
        return conv2d_utils.conv2d_native(
            inputs_data, self.input_size, self.filter_size, self.attrs, True
        )

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [self.input_size, self.filter_size],
            None,
            "depthwise_conv2d",
            self.attrs,
            0,
            True,
        )


class OpTest_pool1d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2])
    attrs.set_attr("stride_size", [2])
    attrs.set_attr("padding_size", [1, 1])
    attrs.set_attr("pool_type", "max")
    attrs.set_attr("ceil_mode", False)
    attrs.set_attr("exclusive", True)
    attrs.set_attr("data_format", "NCW")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2])
    attrs.set_attr("stride_size", [2])
    attrs.set_attr("padding_size", [2, 3])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", False)
    attrs.set_attr("exclusive", True)
    attrs.set_attr("data_format", "NCW")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool1d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2])
    attrs.set_attr("stride_size", [3])
    attrs.set_attr("padding_size", [4, 5])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", True)
    attrs.set_attr("exclusive", False)
    attrs.set_attr("data_format", "NWC")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool1d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 3]
        self.to_test_op([input_shape], None, "pool1d", self.attrs)


class OpTest_pool2d(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2, 2])
    attrs.set_attr("stride_size", [2, 2])
    attrs.set_attr("padding_size", [1, 1, 1, 1])
    attrs.set_attr("pool_type", "max")
    attrs.set_attr("ceil_mode", False)
    attrs.set_attr("exclusive", True)
    attrs.set_attr("data_format", "NCHW")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2, 2])
    attrs.set_attr("stride_size", [2, 2])
    attrs.set_attr("padding_size", [2, 3, 4, 5])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", False)
    attrs.set_attr("exclusive", True)
    attrs.set_attr("data_format", "NCHW")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


class OpTest_pool2d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2, 2])
    attrs.set_attr("stride_size", [3, 3])
    attrs.set_attr("padding_size", [2, 3, 4, 5])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", True)
    attrs.set_attr("exclusive", False)
    attrs.set_attr("data_format", "NHWC")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool2d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool2d", self.attrs)


# The following test is temporarily broken

# class OpTest_pool3d(SingleOpTester):
#     attrs = framework.NodeAttr()
#     attrs.attr_store = {
#         "kernel_size": [2, 2, 2],
#         "stride_size": [2, 2, 2],
#         "padding_size": [1, 2, 3, 4, 5, 6],
#         "pool_type": "max",
#         "ceil_mode": False,
#         "exclusive": True,
#         "data_format": "NCDHW"
#     }

#     def create_target_data(self, inputs_data, attrs):
#         return pool_utils.pool3d(inputs_data[0], self.attrs)

#     def test_op(self):
#         input_shape = [2, 3, 8, 8, 8]
#         self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_1(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2, 2, 2])
    attrs.set_attr("stride_size", [2, 2, 2])
    attrs.set_attr("padding_size", [1, 1, 1, 1, 1, 1])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", False)
    attrs.set_attr("exclusive", True)
    attrs.set_attr("data_format", "NCDHW")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 3, 8, 8, 8]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_pool3d_2(SingleOpTester):
    attrs = framework.NodeAttr()
    attrs.set_attr("kernel_size", [2, 2, 2])
    attrs.set_attr("stride_size", [2, 2, 2])
    attrs.set_attr("padding_size", [1, 2, 3, 4, 5, 6])
    attrs.set_attr("pool_type", "avg")
    attrs.set_attr("ceil_mode", True)
    attrs.set_attr("exclusive", False)
    attrs.set_attr("data_format", "NDHWC")

    def create_target_data(self, inputs_data, attrs):
        return pool_utils.pool3d(inputs_data[0], self.attrs)

    def test_op(self):
        input_shape = [1, 8, 8, 8, 3]
        self.to_test_op([input_shape], None, "pool3d", self.attrs)


class OpTest_batchnorm(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Scale, Bias, Mean, Variance] = inputs_data
        c = X.shape[1]
        for i in range(0, c):
            X[:, i, :, :] = (X[:, i, :, :] - Mean[i]) / math.sqrt(
                Variance[i] + 0.00001
            ) * Scale[i] + Bias[i]
        return X

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op(
            [[1, 64, 112, 112], [64], [64], [64], [64]],
            [[1, 64, 112, 112]],
            "batch_norm",
            attrs,
        )


class OpTest_softmax_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[1]):
            Y[:, i, :] = (
                np.exp(X[:, i, :])
                / np.sum(np.exp(X), axis=1, keepdims=True)[:, 0, :]
            )
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op(
            [[12, 224, 224]],
            [[12, 224, 224], [12, 224, 224]],
            "softmax",
            attrs,
            0,
        )


class OpTest_softmax_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[2]):
            Y[:, :, i] = (
                np.exp(X[:, :, i])
                / np.sum(np.exp(X), axis=2, keepdims=True)[:, :, 0]
            )
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", -1)
        self.to_test_op(
            [[12, 224, 224]],
            [[12, 224, 224], [12, 224, 224]],
            "softmax",
            attrs,
            0,
        )


class OpTest_softmax_2(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = np.zeros(X.shape).astype("float32")
        for i in range(0, Y.shape[0]):
            Y[i, :, :] = (
                np.exp(X[i, :, :])
                / np.sum(np.exp(X), axis=0, keepdims=True)[0, :, :]
            )
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op(
            [[12, 224, 224]],
            [[12, 224, 224], [12, 224, 224]],
            "softmax",
            attrs,
            0,
        )


class OpTest_sigmoid(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        x = np.array(inputs_data[0])
        y = 1 / (1 + np.exp(-x))
        return y

    def test_op(self):
        attrs = framework.NodeAttr()
        self.to_test_op([[3, 224, 224]], [[3, 224, 224]], "sigmoid", attrs)


class OpTest_slice_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = X[:, 0:2, 2:4, :]
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axes", [0, 1, 2])
        attrs.set_attr("starts", [-3, 0, 2])
        attrs.set_attr("ends", [3, 2, 4])
        self.to_test_op([[3, 4, 5, 6]], [[3, 2, 2, 6]], "slice", attrs)


class OpTest_slice_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        Y = X[:, 0:3, 1:2, 2:4]
        return Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axes", [1, 2, 3])
        attrs.set_attr("starts", [0, 1, 2])
        attrs.set_attr("ends", [3, 2, 4])
        self.to_test_op([[3, 4, 5, 6]], [[3, 3, 1, 2]], "slice", attrs)


class OpTest_dropout_infer_0(SingleOpTester):
    def init_testcase(self):
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("dropout_prob", 0.2)
        self.attrs.set_attr("dropout_implementation", "downgrade_in_infer")

    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        assert "dropout_implementation" in self.attrs.attr_store
        if (
            self.attrs.attr_store["dropout_implementation"]
            == "downgrade_in_infer"
        ):
            return X * (1 - self.attrs.attr_store["dropout_prob"])
        else:
            return X

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [[2, 1280, 2, 2]], [[2, 1280, 2, 2]], "dropout_infer", self.attrs
        )


class OpTest_dropout_infer_1(SingleOpTester):
    def init_testcase(self):
        self.attrs = framework.NodeAttr()
        self.attrs.set_attr("dropout_prob", 0.2)
        self.attrs.set_attr("dropout_implementation", "upscale_in_train")

    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        assert "dropout_implementation" in self.attrs.attr_store
        if (
            self.attrs.attr_store["dropout_implementation"]
            == "downgrade_in_infer"
        ):
            return X * (1 - self.attrs.attr_store["dropout_prob"])
        else:
            return X

    def test_op(self):
        self.init_testcase()
        self.to_test_op(
            [[2, 1280, 2, 2]], [[2, 1280, 2, 2]], "dropout_infer", self.attrs
        )


if __name__ == "__main__":
    unittest.main()
