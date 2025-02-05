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

# BUILD_SLICE (new)

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase, test_with_faster_guard

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


def build_list_slice(x: list, y: paddle.Tensor):
    x[2:4] = [0, 1]
    return x[0] + y


def build_list_slice_with_step(x: list, y: paddle.Tensor):
    x[1:5:2] = [0, 1]
    return x[0] + y


def build_tuple_slice(x: list, y: paddle.Tensor):
    x[2:4] = (0, 1)
    return x[0] + y


def build_tuple_slice_with_step(x: list, y: paddle.Tensor):
    x[1:5:2] = (0, 1)
    return x[0] + y


def tensor_subscript_ellipsis(x: paddle.Tensor, y: paddle.Tensor):
    return x[...] + y[...]


@check_no_breakgraph
def tensor_subscript_tensor(x: paddle.Tensor):
    d0, d1 = paddle.shape(x)
    return x[: d0 // 2, d1 // 2 : d1]


class TestSlice(TestCaseBase):
    @test_with_faster_guard
    def test_simple(self):
        x = list(range(10))
        y = paddle.arange(10)
        self.assert_results_with_side_effects(build_list_slice, x, y)
        self.assert_results_with_side_effects(build_list_slice_with_step, x, y)
        self.assert_results_with_side_effects(build_tuple_slice, x, y)
        self.assert_results_with_side_effects(build_tuple_slice_with_step, x, y)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linears = paddle.nn.LayerList(
            [paddle.nn.Linear(10, 10) for i in range(10)]
        )

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


def layer_list_slice(layer, x):
    out = layer(x)
    return out


class TestLayerList(TestCaseBase):
    @test_with_faster_guard
    def test_layer_list_slice(self):
        layer = MyLayer()
        x = paddle.randn([5, 10])
        self.assert_results(layer_list_slice, layer, x)


def tensor_slice(x: paddle.Tensor):
    return x[1, 1, 1] + 1


class TestTensorSlice(TestCaseBase):
    def test_tensor_slice(self):
        x = paddle.randn([4, 3, 10])
        self.assert_results(tensor_slice, x)


class TestTensorEllipsis(TestCaseBase):
    def test_tensor_subscript_ellipsis(self):
        x = paddle.rand((10,))
        y = paddle.rand((10, 10))
        self.assert_results(tensor_subscript_ellipsis, x, y)


class TestTensorSubscriptTensor(TestCaseBase):
    def test_tensor_subscript_tensor(self):
        x = paddle.rand((10, 10))
        self.assert_results(tensor_subscript_tensor, x)


class LayerListNet(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.layer_list = paddle.nn.LayerList(
            [paddle.nn.Linear(5, 5), paddle.nn.Linear(5, 5)]
        )

    def forward(self, x):
        out = self.layer_list[0](x)
        for layer in self.layer_list[1:]:
            out = layer(out)
        return out


class TestLayerListSlice(TestCaseBase):
    @test_with_faster_guard
    def test_layer_list_slice(self):
        x = paddle.randn([2, 5])
        net = LayerListNet()
        self.assert_results(layer_list_slice, net, x)


@check_no_breakgraph
def string_slice(x: str):
    return x[2:7:2] + x[1:5] + x[4]


class TestStringSlice(TestCaseBase):
    def test_string_slice(self):
        x = "1234567"
        self.assert_results(string_slice, x)


if __name__ == "__main__":
    unittest.main()
