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

import unittest

import numpy as np
from test_prim_sub_graph_backward_dynamic_shape import (
    TestPrimBaseWithGrad,
    TestPrimTwoWithGrad,
)

import paddle


def unsqueeze_net(x):
    return paddle.unsqueeze(x, axis=[1, 2])


def where_net(x, y):
    return paddle.where(x > y, x, y)


class TestPrimUnsqueezeWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.unsqueeze_grad"
        self.dtype = "float32"
        self.x_shape = [20, 30, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = unsqueeze_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimWhereWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.where_grad"
        self.dtype = "float32"
        self.x_shape = [30, 30, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 30, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = where_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimWhereWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.where_grad"
        self.dtype = "float32"
        self.x_shape = [30, 30, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 30, 40]
        self.init_y_shape = [30, 30, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = where_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimWhereWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.where_grad"
        self.dtype = "float32"
        self.x_shape = [30, 30, 40]
        self.init_x_shape = [30, 30, 40]
        self.y_shape = [30, 30, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = where_net
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
