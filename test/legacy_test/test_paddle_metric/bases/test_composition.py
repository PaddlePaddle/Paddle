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

from operator import neg, pos
from typing import Any

import pytest

import paddle
from paddle.metric.metric import CompositionalMetric, Metric


class DummyMetric(Metric):
    """DummyMetric class for testing composition component."""

    full_state_update = True

    def __init__(self, val_to_return) -> None:
        super().__init__()
        self.declare("_num_updates", paddle.tensor(0), dist_reduce_fn="sum")
        self._val_to_return = val_to_return

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Compute state."""
        self._num_updates += 1

    def compute(self):
        """Compute result."""
        return paddle.tensor(self._val_to_return)


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(4)),
        (2, paddle.tensor(4)),
        (2.0, paddle.tensor(4.0)),
        pytest.param(paddle.tensor(2), paddle.tensor(4)),
    ],
)
def test_metrics_add(second_operand, expected_result):
    """Test that `add` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_add = first_metric + second_operand
    final_radd = second_operand + first_metric
    assert isinstance(final_add, CompositionalMetric)
    assert isinstance(final_radd, CompositionalMetric)
    final_add.update()
    final_radd.update()
    assert paddle.allclose(x=expected_result, y=final_add.compute()).item()
    assert paddle.allclose(x=expected_result, y=final_radd.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(3), paddle.tensor(2)),
        (3, paddle.tensor(2)),
        (paddle.tensor(3), paddle.tensor(2)),
    ],
)
@pytest.mark.xfail(
    reason="Paddle doesn't support reverse bitwise_and with non-Tensor args",
    strict=False,
)
def test_metrics_and(second_operand, expected_result):
    """Test that `and` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_and = first_metric & second_operand
    final_rand = second_operand & first_metric
    assert isinstance(final_and, CompositionalMetric)
    assert isinstance(final_rand, CompositionalMetric)
    final_and.update()
    final_rand.update()
    assert paddle.allclose(x=expected_result, y=final_and.compute()).item()
    assert paddle.allclose(x=expected_result, y=final_rand.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(True)),
        (2, paddle.tensor(True)),
        (2.0, paddle.tensor(True)),
        (paddle.tensor(2), paddle.tensor(True)),
    ],
)
def test_metrics_eq(second_operand, expected_result):
    """Test that `eq` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_eq = first_metric == second_operand
    assert isinstance(final_eq, CompositionalMetric)
    final_eq.update()
    assert (expected_result == final_eq.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(2)),
        (2, paddle.tensor(2)),
        (2.0, paddle.tensor(2.0)),
        (paddle.tensor(2), paddle.tensor(2)),
    ],
)
def test_metrics_floordiv(second_operand, expected_result):
    """Test that `floordiv` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_floordiv = first_metric // second_operand
    assert isinstance(final_floordiv, CompositionalMetric)
    final_floordiv.update()
    assert paddle.allclose(x=expected_result, y=final_floordiv.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(True)),
        (2, paddle.tensor(True)),
        (2.0, paddle.tensor(True)),
        (paddle.tensor(2), paddle.tensor(True)),
    ],
)
def test_metrics_ge(second_operand, expected_result):
    """Test that `ge` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_ge = first_metric >= second_operand
    assert isinstance(final_ge, CompositionalMetric)
    final_ge.update()
    assert (expected_result == final_ge.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(True)),
        (2, paddle.tensor(True)),
        (2.0, paddle.tensor(True)),
        (paddle.tensor(2), paddle.tensor(True)),
    ],
)
def test_metrics_gt(second_operand, expected_result):
    """Test that `gt` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_gt = first_metric > second_operand
    assert isinstance(final_gt, CompositionalMetric)
    final_gt.update()
    assert (expected_result == final_gt.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(False)),
        (2, paddle.tensor(False)),
        (2.0, paddle.tensor(False)),
        (paddle.tensor(2), paddle.tensor(False)),
    ],
)
def test_metrics_le(second_operand, expected_result):
    """Test that `le` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_le = first_metric <= second_operand
    assert isinstance(final_le, CompositionalMetric)
    final_le.update()
    assert (expected_result == final_le.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(False)),
        (2, paddle.tensor(False)),
        (2.0, paddle.tensor(False)),
        (paddle.tensor(2), paddle.tensor(False)),
    ],
)
def test_metrics_lt(second_operand, expected_result):
    """Test that `lt` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_lt = first_metric < second_operand
    assert isinstance(final_lt, CompositionalMetric)
    final_lt.update()
    assert (expected_result == final_lt.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric([2, 2, 2]), paddle.tensor(12)),
        (paddle.tensor([2, 2, 2]), paddle.tensor(12)),
    ],
)
def test_metrics_matmul(second_operand, expected_result):
    """Test that `matmul` operator works and returns a compositional metric."""
    first_metric = DummyMetric([2, 2, 2])
    final_matmul = first_metric @ second_operand
    assert isinstance(final_matmul, CompositionalMetric)
    final_matmul.update()
    assert paddle.allclose(x=expected_result, y=final_matmul.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(1)),
        (2, paddle.tensor(1)),
        (2.0, paddle.tensor(1)),
        (paddle.tensor(2), paddle.tensor(1)),
    ],
)
@pytest.mark.xfail(
    reason="Paddle .to(float) interpreted as device, not dtype", strict=False
)
def test_metrics_mod(second_operand, expected_result):
    """Test that `mod` operator works and returns a compositional metric."""
    first_metric = DummyMetric(5)
    final_mod = first_metric % second_operand
    assert isinstance(final_mod, CompositionalMetric)
    final_mod.update()
    assert paddle.allclose(
        x=expected_result.to(float), y=final_mod.compute().to(float)
    ).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(4)),
        (2, paddle.tensor(4)),
        (2.0, paddle.tensor(4.0)),
        pytest.param(paddle.tensor(2), paddle.tensor(4)),
    ],
)
def test_metrics_mul(second_operand, expected_result):
    """Test that `mul` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_mul = first_metric * second_operand
    final_rmul = second_operand * first_metric
    assert isinstance(final_mul, CompositionalMetric)
    assert isinstance(final_rmul, CompositionalMetric)
    final_mul.update()
    final_rmul.update()
    assert paddle.allclose(x=expected_result, y=final_mul.compute()).item()
    assert paddle.allclose(x=expected_result, y=final_rmul.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(False)),
        (2, paddle.tensor(False)),
        (2.0, paddle.tensor(False)),
        (paddle.tensor(2), paddle.tensor(False)),
    ],
)
def test_metrics_ne(second_operand, expected_result):
    """Test that `ne` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_ne = first_metric != second_operand
    assert isinstance(final_ne, CompositionalMetric)
    final_ne.update()
    assert (expected_result == final_ne.compute()).all()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric([1, 0, 3]), paddle.tensor([-1, -2, 3])),
        (paddle.tensor([1, 0, 3]), paddle.tensor([-1, -2, 3])),
    ],
)
@pytest.mark.xfail(
    reason="Paddle doesn't support reverse bitwise_or with non-Tensor args",
    strict=False,
)
def test_metrics_or(second_operand, expected_result):
    """Test that `or` operator works and returns a compositional metric."""
    first_metric = DummyMetric([-1, -2, 3])
    final_or = first_metric | second_operand
    final_ror = second_operand | first_metric
    assert isinstance(final_or, CompositionalMetric)
    assert isinstance(final_ror, CompositionalMetric)
    final_or.update()
    final_ror.update()
    assert paddle.allclose(x=expected_result, y=final_or.compute()).item()
    assert paddle.allclose(x=expected_result, y=final_ror.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(4)),
        (2, paddle.tensor(4)),
        pytest.param(2.0, paddle.tensor(4.0)),
        (paddle.tensor(2), paddle.tensor(4)),
    ],
)
def test_metrics_pow(second_operand, expected_result):
    """Test that `pow` operator works and returns a compositional metric."""
    first_metric = DummyMetric(2)
    final_pow = first_metric**second_operand
    assert isinstance(final_pow, CompositionalMetric)
    final_pow.update()
    assert paddle.allclose(x=expected_result, y=final_pow.compute()).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [
        (5, paddle.tensor(2)),
        (5.0, paddle.tensor(2.0)),
        (paddle.tensor(5), paddle.tensor(2)),
    ],
)
def test_metrics_rfloordiv(first_operand, expected_result):
    """Test that `rfloordiv` operator works and returns a compositional metric."""
    second_operand = DummyMetric(2)
    final_rfloordiv = first_operand // second_operand
    assert isinstance(final_rfloordiv, CompositionalMetric)
    final_rfloordiv.update()
    assert paddle.allclose(
        x=expected_result, y=final_rfloordiv.compute()
    ).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [pytest.param(paddle.tensor([2, 2, 2]), paddle.tensor(12))],
)
def test_metrics_rmatmul(first_operand, expected_result):
    """Test that `rmatmul` operator works and returns a compositional metric."""
    second_operand = DummyMetric([2, 2, 2])
    final_rmatmul = first_operand @ second_operand
    assert isinstance(final_rmatmul, CompositionalMetric)
    final_rmatmul.update()
    assert paddle.allclose(x=expected_result, y=final_rmatmul.compute()).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [pytest.param(paddle.tensor(2), paddle.tensor(2))],
)
def test_metrics_rmod(first_operand, expected_result):
    """Test that `rmod` operator works and returns a compositional metric."""
    second_operand = DummyMetric(5)
    final_rmod = first_operand % second_operand
    assert isinstance(final_rmod, CompositionalMetric)
    final_rmod.update()
    assert paddle.allclose(x=expected_result, y=final_rmod.compute()).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(4)),
        (2, paddle.tensor(4)),
        pytest.param(2.0, paddle.tensor(4.0)),
    ],
)
def test_metrics_rpow(first_operand, expected_result):
    """Test that `rpow` operator works and returns a compositional metric."""
    second_operand = DummyMetric(2)
    final_rpow = first_operand**second_operand
    assert isinstance(final_rpow, CompositionalMetric)
    final_rpow.update()
    assert paddle.allclose(x=expected_result, y=final_rpow.compute()).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [
        (DummyMetric(3), paddle.tensor(1)),
        (3, paddle.tensor(1)),
        (3.0, paddle.tensor(1.0)),
        pytest.param(paddle.tensor(3), paddle.tensor(1)),
    ],
)
def test_metrics_rsub(first_operand, expected_result):
    """Test that `rsub` operator works and returns a compositional metric."""
    second_operand = DummyMetric(2)
    final_rsub = first_operand - second_operand
    assert isinstance(final_rsub, CompositionalMetric)
    final_rsub.update()
    assert paddle.allclose(x=expected_result, y=final_rsub.compute()).item()


@pytest.mark.parametrize(
    ("first_operand", "expected_result"),
    [
        (DummyMetric(6), paddle.tensor(2.0)),
        (6, paddle.tensor(2.0)),
        (6.0, paddle.tensor(2.0)),
        (paddle.tensor(6), paddle.tensor(2.0)),
    ],
)
@pytest.mark.xfail(
    reason="Paddle doesn't support reverse truediv with non-Tensor args",
    strict=False,
)
def test_metrics_rtruediv(first_operand, expected_result):
    """Test that `rtruediv` operator works and returns a compositional metric."""
    second_operand = DummyMetric(3)
    final_rtruediv = first_operand / second_operand
    assert isinstance(final_rtruediv, CompositionalMetric)
    final_rtruediv.update()
    assert paddle.allclose(x=expected_result, y=final_rtruediv.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(2), paddle.tensor(1)),
        (2, paddle.tensor(1)),
        (2.0, paddle.tensor(1.0)),
        (paddle.tensor(2), paddle.tensor(1)),
    ],
)
def test_metrics_sub(second_operand, expected_result):
    """Test that `sub` operator works and returns a compositional metric."""
    first_metric = DummyMetric(3)
    final_sub = first_metric - second_operand
    assert isinstance(final_sub, CompositionalMetric)
    final_sub.update()
    assert paddle.allclose(x=expected_result, y=final_sub.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric(3), paddle.tensor(2.0)),
        (3, paddle.tensor(2.0)),
        (3.0, paddle.tensor(2.0)),
        (paddle.tensor(3), paddle.tensor(2.0)),
    ],
)
def test_metrics_truediv(second_operand, expected_result):
    """Test that `truediv` operator works and returns a compositional metric."""
    first_metric = DummyMetric(6)
    final_truediv = first_metric / second_operand
    assert isinstance(final_truediv, CompositionalMetric)
    final_truediv.update()
    assert paddle.allclose(x=expected_result, y=final_truediv.compute()).item()


@pytest.mark.parametrize(
    ("second_operand", "expected_result"),
    [
        (DummyMetric([1, 0, 3]), paddle.tensor([-2, -2, 0])),
        (paddle.tensor([1, 0, 3]), paddle.tensor([-2, -2, 0])),
    ],
)
@pytest.mark.xfail(
    reason="Paddle doesn't support reverse bitwise_xor with non-Tensor args",
    strict=False,
)
def test_metrics_xor(second_operand, expected_result):
    """Test that `xor` operator works and returns a compositional metric."""
    first_metric = DummyMetric([-1, -2, 3])
    final_xor = first_metric ^ second_operand
    final_rxor = second_operand ^ first_metric
    assert isinstance(final_xor, CompositionalMetric)
    assert isinstance(final_rxor, CompositionalMetric)
    final_xor.update()
    final_rxor.update()
    assert paddle.allclose(x=expected_result, y=final_xor.compute()).item()
    assert paddle.allclose(x=expected_result, y=final_rxor.compute()).item()


def test_metrics_abs():
    """Test that `abs` operator works and returns a compositional metric."""
    first_metric = DummyMetric(-1)
    final_abs = abs(first_metric)
    assert isinstance(final_abs, CompositionalMetric)
    final_abs.update()
    assert paddle.allclose(x=paddle.tensor(1), y=final_abs.compute()).item()


def test_metrics_invert():
    """Test that `invert` operator works and returns a compositional metric."""
    first_metric = DummyMetric(1)
    final_inverse = ~first_metric
    assert isinstance(final_inverse, CompositionalMetric)
    final_inverse.update()
    assert paddle.allclose(
        x=paddle.tensor(-2), y=final_inverse.compute()
    ).item()


def test_metrics_neg():
    """Test that `neg` operator works and returns a compositional metric."""
    first_metric = DummyMetric(1)
    final_neg = neg(first_metric)
    assert isinstance(final_neg, CompositionalMetric)
    final_neg.update()
    assert paddle.allclose(x=paddle.tensor(-1), y=final_neg.compute()).item()


def test_metrics_pos():
    """Test that `pos` operator works and returns a compositional metric."""
    first_metric = DummyMetric(-1)
    final_pos = pos(first_metric)
    assert isinstance(final_pos, CompositionalMetric)
    final_pos.update()
    assert paddle.allclose(x=paddle.tensor(1), y=final_pos.compute()).item()


@pytest.mark.parametrize(
    ("value", "idx", "expected_result"),
    [
        ([1, 2, 3], 1, paddle.tensor(2)),
        ([[0, 1], [2, 3]], (1, 0), paddle.tensor(2)),
        ([[0, 1], [2, 3]], 1, paddle.tensor([2, 3])),
    ],
)
def test_metrics_getitem(value, idx, expected_result):
    """Test that `getitem` operator works and returns a compositional metric."""
    first_metric = DummyMetric(value)
    final_getitem = first_metric[idx]
    assert isinstance(final_getitem, CompositionalMetric)
    final_getitem.update()
    assert paddle.allclose(x=expected_result, y=final_getitem.compute()).item()


def test_compositional_metrics_update():
    """Test update method for compositional metrics."""
    compos = DummyMetric(5) + DummyMetric(4)
    assert isinstance(compos, CompositionalMetric)
    compos.update()
    compos.update()
    compos.update()
    assert isinstance(compos.metric_a, DummyMetric)
    assert isinstance(compos.metric_b, DummyMetric)
    assert compos.metric_a._num_updates == 3
    assert compos.metric_b._num_updates == 3
