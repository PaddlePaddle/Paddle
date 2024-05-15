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

import numpy as np
from api_base import ApiBase

import paddle


class TestInstanceNorm2DOp(unittest.TestCase):
    def setUp(self):
        self.init_test_case()
        self.init_wb()

    def test_api(self):

        test = ApiBase(
            func=paddle.nn.InstanceNorm2D,
            feed_names=['X'],
            feed_shapes=[self.input_size],
            is_train=True,
        )
        np.random.seed(1)
        data = np.random.uniform(-1, 1, self.input_size).astype('float32')
        test.run(
            feed=[data],
            num_features=self.C,
            weight_attr=self.weight_attr,
            bias_attr=self.bias_attr,
        )

    def init_test_case(self):
        self.input_size = [4, 2, 16, 16]
        self.C = self.input_size[1]

    def init_wb(self):
        self.weight_attr = None
        self.bias_attr = None


class TestInstanceNorm1D(TestInstanceNorm2DOp):
    def init_test_case(self):
        self.input_size = [4, 16, 16]
        self.C = self.input_size[1]

    def test_api(self):

        test = ApiBase(
            func=paddle.nn.InstanceNorm1D,
            feed_names=['X'],
            feed_shapes=[self.input_size],
            is_train=True,
        )
        np.random.seed(1)
        data = np.random.uniform(-1, 1, self.input_size).astype('float32')
        test.run(
            feed=[data],
            num_features=self.C,
            weight_attr=self.weight_attr,
            bias_attr=self.bias_attr,
        )


class TestInstanceNorm3D(TestInstanceNorm2DOp):
    def init_test_case(self):
        self.input_size = [16, 4, 9, 9, 9]
        self.C = self.input_size[1]

    def test_api(self):

        test = ApiBase(
            func=paddle.nn.InstanceNorm3D,
            feed_names=['X'],
            feed_shapes=[self.input_size],
            is_train=True,
        )
        np.random.seed(1)
        data = np.random.uniform(-1, 1, self.input_size).astype('float32')
        test.run(
            feed=[data],
            num_features=self.C,
            weight_attr=self.weight_attr,
            bias_attr=self.bias_attr,
        )


if __name__ == "__main__":
    a = TestInstanceNorm3D()
    a.setUp()
    a.test_api()
