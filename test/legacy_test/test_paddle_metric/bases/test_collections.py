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

"""Tests for MetricCollection."""

import paddle
from paddle.metric import Accuracy, MetricCollection, Precision, Recall


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


class TestMetricCollection:
    def test_basic_list(self):
        metrics = MetricCollection(
            [
                Accuracy(task="binary"),
                Precision(task="binary"),
                Recall(task="binary"),
            ]
        )
        preds = paddle.to_tensor([0.9, 0.2, 0.8, 0.1])
        target = paddle.to_tensor([1, 0, 1, 0])
        results = metrics(preds, target)
        assert isinstance(results, dict)
        assert len(results) == 3

    def test_basic_dict(self):
        metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary"),
                "prec": Precision(task="binary"),
            }
        )
        preds = paddle.to_tensor([0.9, 0.2, 0.8, 0.1])
        target = paddle.to_tensor([1, 0, 1, 0])
        results = metrics(preds, target)
        assert "acc" in results
        assert "prec" in results

    def test_prefix(self):
        metrics = MetricCollection(
            [Accuracy(task="binary")],
            prefix="test_",
        )
        preds = paddle.to_tensor([0.9, 0.2])
        target = paddle.to_tensor([1, 0])
        results = metrics(preds, target)
        assert any(k.startswith("test_") for k in results.keys())

    def test_reset(self):
        metrics = MetricCollection([Accuracy(task="binary")])
        preds = paddle.to_tensor([0.9, 0.2])
        target = paddle.to_tensor([1, 0])
        metrics.update(preds, target)
        metrics.reset()
        for m in metrics.values():
            assert not m.update_called

    def test_compute(self):
        metrics = MetricCollection([Accuracy(task="binary")])
        preds = paddle.to_tensor([0.9, 0.2, 0.8, 0.1])
        target = paddle.to_tensor([1, 0, 1, 0])
        metrics.update(preds, target)
        results = metrics.compute()
        assert isinstance(results, dict)
        _assert_scalar_close(next(iter(results.values())), 1.0)
