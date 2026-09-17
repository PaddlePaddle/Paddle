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

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle
from paddle.static.input import InputSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def group_norm_with_linspace(x):
    weight = paddle.linspace(0.75, 1.25, 6, dtype='float32')
    bias = paddle.linspace(-0.3, 0.3, 6, dtype='float32')
    return paddle.nn.functional.group_norm(
        x,
        num_groups=3,
        weight=weight,
        bias=bias,
        epsilon=1e-5,
    )


def instance_norm_with_linspace(x):
    weight = paddle.linspace(0.8, 1.2, 6, dtype='float32')
    bias = paddle.linspace(-0.2, 0.1, 6, dtype='float32')
    return paddle.nn.functional.instance_norm(
        x,
        weight=weight,
        bias=bias,
        use_input_stats=True,
        eps=1e-5,
    )


def instance_norm_with_affine_inputs(x, weight, bias):
    return paddle.nn.functional.instance_norm(
        x,
        weight=weight,
        bias=bias,
        use_input_stats=True,
        eps=1e-5,
    )


class TestDynamicShapeInfermeta(Dy2StTestBase):
    def check_dynamic_shape(
        self,
        fn: Callable[..., Any],
        inputs: Sequence[paddle.Tensor],
        input_specs: list[InputSpec],
    ):
        static_fn = paddle.jit.to_static(
            fn,
            full_graph=True,
            input_spec=input_specs,
        )
        np.testing.assert_allclose(static_fn(*inputs), fn(*inputs), rtol=1e-05)

    @test_ast_only
    def test_conv2d(self):
        self.check_dynamic_shape(
            paddle.nn.Conv2D(3, 3, 3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_ast_only
    def test_bn(self):
        self.check_dynamic_shape(
            paddle.nn.BatchNorm2D(3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_ast_only
    def test_depthwise_conv2d(self):
        self.check_dynamic_shape(
            paddle.nn.Conv2D(3, 3, 3, groups=3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_ast_only
    def test_group_norm(self):
        self.check_dynamic_shape(
            paddle.nn.GroupNorm(3, 3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_ast_only
    def test_group_norm_with_linspace(self):
        x = paddle.arange(-96, 96, dtype='float32').reshape([2, 6, 4, 4])
        self.check_dynamic_shape(
            group_norm_with_linspace,
            [x / 17.0],
            [InputSpec(shape=[2, 6, 4, 4], dtype='float32')],
        )

    @test_ast_only
    def test_instance_norm_with_linspace(self):
        x = paddle.arange(-150, 150, dtype='float32').reshape([2, 6, 5, 5])
        self.check_dynamic_shape(
            instance_norm_with_linspace,
            [x / 23.0],
            [InputSpec(shape=[2, 6, 5, 5], dtype='float32')],
        )

    @test_ast_only
    def test_instance_norm_with_dynamic_channel(self):
        x = paddle.arange(-150, 150, dtype='float32').reshape([2, 6, 5, 5])
        weight = paddle.linspace(0.8, 1.2, 6, dtype='float32')
        bias = paddle.linspace(-0.2, 0.1, 6, dtype='float32')
        self.check_dynamic_shape(
            instance_norm_with_affine_inputs,
            [x / 23.0, weight, bias],
            [
                InputSpec(shape=[None, None, None, None], dtype='float32'),
                InputSpec(shape=[6], dtype='float32'),
                InputSpec(shape=[6], dtype='float32'),
            ],
        )

    @test_ast_only
    def test_functional_conv(self):
        self.check_dynamic_shape(
            paddle.nn.functional.conv2d,
            [paddle.randn([1, 3, 32, 32]), paddle.randn([3, 3, 3, 3])],
            [
                InputSpec(shape=[None, None, None, None], dtype='float32'),
                InputSpec(shape=[None, None, None, None], dtype='float32'),
            ],
        )


if __name__ == '__main__':
    unittest.main()
