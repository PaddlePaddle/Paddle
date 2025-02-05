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
    test_pir_only,
)

import paddle
from paddle.static.input import InputSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


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

    @test_pir_only
    @test_ast_only
    def test_conv2d(self):
        self.check_dynamic_shape(
            paddle.nn.Conv2D(3, 3, 3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_pir_only
    @test_ast_only
    def test_bn(self):
        self.check_dynamic_shape(
            paddle.nn.BatchNorm2D(3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_pir_only
    @test_ast_only
    def test_depthwise_conv2d(self):
        self.check_dynamic_shape(
            paddle.nn.Conv2D(3, 3, 3, groups=3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_pir_only
    @test_ast_only
    def test_group_norm(self):
        self.check_dynamic_shape(
            paddle.nn.GroupNorm(3, 3),
            [paddle.randn([1, 3, 32, 32])],
            [InputSpec(shape=[None, None, None, None], dtype='float32')],
        )

    @test_pir_only
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
