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

"""Tests for the core Metric base class."""

import pickle

import pytest

import paddle
from paddle.metric.metric import Metric


def _assert_scalar_close(actual, expected, atol=1e-6):
    """Compare metric result with expected scalar value, handling shape differences."""
    if not isinstance(actual, paddle.Tensor):
        actual = paddle.to_tensor(actual)
    if not isinstance(expected, paddle.Tensor):
        expected = paddle.to_tensor(expected)
    # Flatten both to 1-d for comparison
    actual = actual.flatten()
    expected = expected.flatten()
    assert paddle.allclose(actual, expected, atol=atol), (
        f"Expected {expected.numpy()}, got {actual.numpy()}"
    )


class DummyMetric(Metric):
    def __init__(self, dist_reduce_fn="sum", **kwargs):
        super().__init__(dist_reduce_fn=dist_reduce_fn, **kwargs)
        self.declare(
            "x", default=paddle.zeros([1]), dist_reduce_fn=dist_reduce_fn
        )

    def update(self, val):
        self.x += val

    def compute(self):
        return self.x


class DummyListMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.declare("values", default=[], dist_reduce_fn="cat")

    def update(self, val):
        self.values.append(val)

    def compute(self):
        return paddle.concat(self.values).sum()


class TestMetric:
    def test_basic_update_compute(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([1.0]))
        m.update(paddle.to_tensor([2.0]))
        result = m.compute()
        _assert_scalar_close(result, 3.0)

    def test_reset(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([5.0]))
        m.reset()
        assert m.update_count == 0
        assert not m.update_called
        result = m.compute()
        _assert_scalar_close(result, 0.0)

    def test_forward(self):
        m = DummyMetric()
        batch_val = m(paddle.to_tensor([3.0]))
        _assert_scalar_close(batch_val, 3.0)
        batch_val2 = m(paddle.to_tensor([2.0]))
        _assert_scalar_close(batch_val2, 2.0)
        _assert_scalar_close(m.compute(), 5.0)

    def test_name_property(self):
        m = DummyMetric()
        assert m.name == "DummyMetric"

    def test_name_custom(self):
        m = DummyMetric(name="my_metric")
        assert m.name == "my_metric"

    def test_update_called(self):
        m = DummyMetric()
        assert not m.update_called
        assert m.update_count == 0
        m.update(paddle.to_tensor([1.0]))
        assert m.update_called
        assert m.update_count == 1

    def test_metric_state(self):
        m = DummyMetric()
        state = m.metric_state
        assert "x" in state

    def test_list_state(self):
        m = DummyListMetric()
        m.update(paddle.to_tensor([1.0]))
        m.update(paddle.to_tensor([2.0]))
        result = m.compute()
        _assert_scalar_close(result, 3.0)

    def test_clone(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([3.0]))
        m2 = m.clone()
        _assert_scalar_close(m2.compute(), 3.0)

    def test_persistent(self):
        m = DummyMetric()
        m.persistent(True)
        sd = m.state_dict()
        assert any("x" in k for k in sd.keys()), (
            f"Expected 'x' in state_dict keys: {list(sd.keys())}"
        )

    def test_pickle(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([3.0]))
        data = pickle.dumps(m)
        m2 = pickle.loads(data)
        _assert_scalar_close(m2.compute(), 3.0)

    def test_hash(self):
        m1 = DummyMetric()
        m2 = DummyMetric()
        assert hash(m1) != hash(m2)

    def test_declare_alias_declare(self):
        class MyMetric(Metric):
            def __init__(self):
                super().__init__()
                self.declare(
                    "y", default=paddle.zeros([1]), dist_reduce_fn="sum"
                )

            def update(self, val):
                self.y += val

            def compute(self):
                return self.y

        m = MyMetric()
        m.update(paddle.to_tensor([5.0]))
        _assert_scalar_close(m.compute(), 5.0)

    def test_protected_attrs(self):
        m = DummyMetric()
        with pytest.raises(RuntimeError):
            m.higher_is_better = True

    def test_state_dict_load(self):
        m = DummyMetric()
        m.persistent(True)
        m.update(paddle.to_tensor([7.0]))
        sd = m.state_dict()
        assert len(sd) > 0, (
            "state_dict should not be empty after persistent(True)"
        )
        m2 = DummyMetric()
        m2.persistent(True)
        m2.set_state_dict(sd)
        _assert_scalar_close(m2.compute(), 7.0)


class TestCompositionalMetric:
    def test_add(self):
        m1 = DummyMetric()
        m2 = DummyMetric()
        m1.update(paddle.to_tensor([2.0]))
        m2.update(paddle.to_tensor([3.0]))
        composed = m1 + m2
        result = composed.compute()
        _assert_scalar_close(result, 5.0)

    def test_mul_scalar(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([3.0]))
        composed = m * 2.0
        result = composed.compute()
        _assert_scalar_close(result, 6.0)

    def test_sub(self):
        m1 = DummyMetric()
        m2 = DummyMetric()
        m1.update(paddle.to_tensor([5.0]))
        m2.update(paddle.to_tensor([3.0]))
        composed = m1 - m2
        result = composed.compute()
        _assert_scalar_close(result, 2.0)

    def test_div(self):
        m = DummyMetric()
        m.update(paddle.to_tensor([6.0]))
        composed = m / 2.0
        result = composed.compute()
        _assert_scalar_close(result, 3.0)

    def test_reset(self):
        m1 = DummyMetric()
        m2 = DummyMetric()
        m1.update(paddle.to_tensor([5.0]))
        m2.update(paddle.to_tensor([3.0]))
        composed = m1 + m2
        composed.reset()
        assert not m1.update_called
        assert not m2.update_called


class TestEdgeCases:
    def test_compute_before_update_warns(self):
        m = DummyMetric()
        with pytest.warns(UserWarning, match="before the.*update"):
            m.compute()

    def test_forward_while_synced_raises(self):
        m = DummyMetric()
        m._is_synced = True
        with pytest.raises(RuntimeError, match="synced"):
            m(paddle.to_tensor([1.0]))

    def test_unsync_not_synced_raises(self):
        m = DummyMetric()
        with pytest.raises(RuntimeError, match="un-synced"):
            m.unsync()

    def test_declare_invalid_default(self):
        with pytest.raises(ValueError):
            m = DummyMetric()
            m.declare("bad", default=[1, 2, 3])

    def test_declare_invalid_reduce_fx(self):
        with pytest.raises(ValueError):
            m = DummyMetric()
            m.declare(
                "bad", default=paddle.zeros([1]), dist_reduce_fn="invalid"
            )
