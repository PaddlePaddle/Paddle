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

import sys
import unittest

from dygraph_to_static_utils import Dy2StTestBase

import paddle
from paddle import nn


class MyConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.inner_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.inner_conv(x)


class ExampleCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = MyConv2D(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2D(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(256 * 28 * 28, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = paddle.flatten(x, start_axis=1)
        x = self.fc1(x)
        return x


class TestCircularReference(Dy2StTestBase):
    def test_circular_reference(self):
        model = ExampleCNN()
        model_ref_before_to_static = sys.getrefcount(model)
        inner_model_ref_before_to_static = sys.getrefcount(model.conv1)
        paddle.jit.to_static(model)
        model_ref_after_to_static = sys.getrefcount(model)
        inner_model_ref_after_to_static = sys.getrefcount(model.conv1)
        # NOTE(SigureMo): The reference count of `model` must be the same before and after `to_static`.
        # Otherwise, it may cause memory leak.
        self.assertEqual(model_ref_before_to_static, model_ref_after_to_static)
        self.assertEqual(
            inner_model_ref_before_to_static, inner_model_ref_after_to_static
        )


if __name__ == '__main__':
    unittest.main()
