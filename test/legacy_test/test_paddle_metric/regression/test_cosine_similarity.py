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

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

import paddle
from paddle.metric.functional.regression.cosine_similarity import (
    cosine_similarity,
)
from paddle.metric.regression.cosine_similarity import CosineSimilarity

seed_all(42)
NUM_TARGETS = 5
_single_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, 1),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, 1),
)
_multi_target_inputs = _Input(
    preds=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=paddle.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _reference_sklearn_cosine(preds, target, reduction):
    sk_preds = preds.numpy()
    sk_target = target.numpy()
    result_array = sk_cosine(sk_target, sk_preds)
    col = np.diagonal(result_array)
    col_sum = col.sum()
    if reduction == "sum":
        return col_sum
    if reduction == "mean":
        return col_sum / len(col)
    return col


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (
            _single_target_inputs.preds,
            _single_target_inputs.target,
            _reference_sklearn_cosine,
        ),
        (
            _multi_target_inputs.preds,
            _multi_target_inputs.target,
            _reference_sklearn_cosine,
        ),
    ],
)
class TestCosineSimilarity(MetricTester):
    """Test class for `CosineSimilarity` metric."""

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_cosine_similarity(self, reduction, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            CosineSimilarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_cosine_similarity_functional(
        self, reduction, preds, target, ref_metric
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            cosine_similarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )


def test_error_on_different_shape(metric_class=CosineSimilarity):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(
        RuntimeError,
        match="Predictions and targets are expected to have the same shape",
    ):
        metric(paddle.randn(100, 2), paddle.randn(50, 2))


def test_error_on_non_2d_input():
    """Test that error is raised if input is not 2-dimensional."""
    metric = CosineSimilarity()
    with pytest.raises(
        ValueError,
        match="Expected input to cosine similarity to be 2D tensors of shape.*",
    ):
        metric(paddle.randn(100), paddle.randn(100))
