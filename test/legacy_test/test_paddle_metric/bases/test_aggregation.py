# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for aggregation metrics."""

import paddle
from paddle.metric.aggregation import (
    CatMetric,
    MaxMetric,
    MeanMetric,
    MinMetric,
    SumMetric,
)


def _assert_scalar_close(actual, expected, atol=1e-6):
    if not isinstance(actual, paddle.Tensor):
        actual = paddle.to_tensor(actual)
    if not isinstance(expected, paddle.Tensor):
        expected = paddle.to_tensor(expected)
    actual = actual.flatten()
    expected = expected.flatten()
    assert paddle.allclose(actual, expected, atol=atol), (
        f"Expected {expected.numpy()}, got {actual.numpy()}"
    )


class TestSumMetric:
    def test_basic(self):
        m = SumMetric()
        m.update(paddle.to_tensor(1.0))
        m.update(paddle.to_tensor(2.0))
        m.update(paddle.to_tensor(3.0))
        _assert_scalar_close(m.compute(), 6.0)

    def test_reset(self):
        m = SumMetric()
        m.update(paddle.to_tensor(5.0))
        m.reset()
        _assert_scalar_close(m.compute(), 0.0)

    def test_forward(self):
        m = SumMetric()
        val = m(paddle.to_tensor(4.0))
        _assert_scalar_close(val, 4.0)
        val2 = m(paddle.to_tensor(3.0))
        _assert_scalar_close(val2, 3.0)
        _assert_scalar_close(m.compute(), 7.0)


class TestMeanMetric:
    def test_basic(self):
        m = MeanMetric()
        m.update(paddle.to_tensor(2.0))
        m.update(paddle.to_tensor(4.0))
        _assert_scalar_close(m.compute(), 3.0)

    def test_single_value(self):
        m = MeanMetric()
        m.update(paddle.to_tensor(5.0))
        _assert_scalar_close(m.compute(), 5.0)


class TestMaxMetric:
    def test_basic(self):
        m = MaxMetric()
        m.update(paddle.to_tensor(1.0))
        m.update(paddle.to_tensor(5.0))
        m.update(paddle.to_tensor(3.0))
        _assert_scalar_close(m.compute(), 5.0)


class TestMinMetric:
    def test_basic(self):
        m = MinMetric()
        m.update(paddle.to_tensor(5.0))
        m.update(paddle.to_tensor(1.0))
        m.update(paddle.to_tensor(3.0))
        _assert_scalar_close(m.compute(), 1.0)


class TestCatMetric:
    def test_basic(self):
        m = CatMetric()
        m.update(paddle.to_tensor([1.0, 2.0]))
        m.update(paddle.to_tensor([3.0, 4.0]))
        result = m.compute()
        expected = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        assert paddle.allclose(result, expected)
