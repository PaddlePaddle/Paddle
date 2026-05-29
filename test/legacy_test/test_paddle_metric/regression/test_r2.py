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

from functools import partial

import pytest
from sklearn.metrics import r2_score as sk_r2score
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

import paddle
from paddle.metric.functional import r2_score
from paddle.metric.regression import R2Score

seed_all(42)
NUM_TARGETS = 5
_single_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE),
)
_multi_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _single_target_ref_wrapper(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (
            sk_preds.shape[0] - adjusted - 1
        )
    return r2_score


def _multi_target_ref_wrapper(preds, target, adjusted, multioutput):
    sk_preds = preds.view(-1, NUM_TARGETS).numpy()
    sk_target = target.view(-1, NUM_TARGETS).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        return 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (
            sk_preds.shape[0] - adjusted - 1
        )
    return r2_score


@pytest.mark.parametrize("adjusted", [0, 5, 10])
@pytest.mark.parametrize(
    "multioutput", ["raw_values", "uniform_average", "variance_weighted"]
)
@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (
            _single_target_inputs.preds,
            _single_target_inputs.target,
            _single_target_ref_wrapper,
        ),
        (
            _multi_target_inputs.preds,
            _multi_target_inputs.target,
            _multi_target_ref_wrapper,
        ),
    ],
)
class TestR2Score(MetricTester):
    """Test class for `R2Score` metric."""

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_r2(self, adjusted, multioutput, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            R2Score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_functional(
        self, adjusted, multioutput, preds, target, ref_metric
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            r2_score,
            partial(ref_metric, adjusted=adjusted, multioutput=multioutput),
            metric_args={"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_differentiability(
        self, adjusted, multioutput, preds, target, ref_metric
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds,
            target,
            R2Score,
            r2_score,
            {"adjusted": adjusted, "multioutput": multioutput},
        )

    def test_r2_half_cpu(
        self, adjusted, multioutput, preds, target, ref_metric
    ):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            R2Score,
            r2_score,
            {"adjusted": adjusted, "multioutput": multioutput},
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    def test_r2_half_gpu(
        self, adjusted, multioutput, preds, target, ref_metric
    ):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds,
            target,
            R2Score,
            r2_score,
            {"adjusted": adjusted, "multioutput": multioutput},
        )


def test_error_on_different_shape(metric_class=R2Score):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100), paddle.randn(50))


def test_error_on_multidim_tensors(metric_class=R2Score):
    """Test that error is raised if a larger than 2D tensor is given as input."""
    metric = metric_class()
    with pytest.raises(
        ValueError,
        match="Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension .",
    ):
        metric(paddle.randn(10, 20, 5), paddle.randn(10, 20, 5))


def test_error_on_too_few_samples(metric_class=R2Score):
    """Test that error is raised if too few samples are provided."""
    metric = metric_class()
    with pytest.raises(
        ValueError, match="Needs at least two samples to calculate r2 score."
    ):
        metric(paddle.randn(1), paddle.randn(1))
    metric.reset()
    metric.update(paddle.randn(1), paddle.randn(1))
    metric.update(paddle.randn(1), paddle.randn(1))
    assert metric.compute()


def test_warning_on_too_large_adjusted(metric_class=R2Score):
    """Test that warning is raised if adjusted argument is set to more than or equal to the number of datapoints."""
    metric = metric_class(adjusted=10)
    with pytest.warns(
        UserWarning,
        match="More independent regressions than data points in adjusted r2 score. Falls back to standard r2 score.",
    ):
        metric(paddle.randn(10), paddle.randn(10))
    with pytest.warns(
        UserWarning,
        match="Division by zero in adjusted r2 score. Falls back to standard r2 score.",
    ):
        metric(paddle.randn(11), paddle.randn(11))


def test_constant_target():
    """Check for a near constant target that a value of 0 is returned."""
    y_true = paddle.tensor(
        [-5.1608, -5.1609, -5.1608, -5.1608, -5.1608, -5.1608]
    )
    y_pred = paddle.tensor(
        [-3.9865, -5.4648, -5.0238, -4.3899, -5.6672, -4.7336]
    )
    score = r2_score(preds=y_pred, target=y_true)
    assert score == 0
