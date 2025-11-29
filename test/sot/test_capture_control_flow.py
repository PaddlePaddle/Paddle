# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle import nn


@paddle.jit.marker.capture_control_flow
def inner_fn_with_control_flow_explicit_capture(x):
    if x.sum() > 0:
        x += 1
    else:
        x -= 1
    return x


def fn_with_control_flow_explicit_capture(x):
    x = inner_fn_with_control_flow_explicit_capture(x)
    return x + 1


def fn_without_capture(x):
    if x.sum() > 0:
        x += 1
    else:
        x -= 1
    return x + 1


class TestCaptureControlFlow(TestCaseBase):
    def test_case_without_capture_control_flow(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x = paddle.full([3, 3], 1)
            self.assert_results(fn_without_capture, x)
            self.assertEqual(ctx.translate_count, 2)
            x = paddle.full([3, 3], -1)
            self.assert_results(fn_without_capture, x)
            self.assertEqual(ctx.translate_count, 3)

    def test_case_capture_control_flow(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x = paddle.full([3, 3], 1)
            self.assert_results(fn_with_control_flow_explicit_capture, x)
            self.assertEqual(ctx.translate_count, 1)
            x = paddle.full([3, 3], -1)
            self.assert_results(fn_with_control_flow_explicit_capture, x)
            self.assertEqual(ctx.translate_count, 1)


class NetWithCaptureControlFlow(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8)

    @paddle.jit.marker.capture_control_flow
    def fn(self, x):
        x = self.layer(x)
        if x.sum() > 0:
            x += paddle.ones_like(x)
        else:
            x -= paddle.zeros_like(x)
        return x

    def forward(self, x):
        return self.fn(x) + 1


def model_call(x: paddle.Tensor, net: paddle.nn.Layer):
    return net(x)


class TestEagerParamsToPirValue(TestCaseBase):
    def test_case_without_capture_control_flow(self):
        model = NetWithCaptureControlFlow()
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x = paddle.randn([4, 8])
            self.assert_results(model_call, x, model)
            self.assertEqual(ctx.translate_count, 1)
            x = paddle.randn([4, 8])
            self.assert_results(model_call, x, model)
            self.assertEqual(ctx.translate_count, 1)


if __name__ == "__main__":
    unittest.main()
