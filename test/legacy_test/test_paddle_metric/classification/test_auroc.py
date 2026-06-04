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
from scipy.special import expit as sigmoid, softmax
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import (
    MetricTester,
    inject_ignore_index,
    remove_ignore_index,
)
from unittests.classification._inputs import (
    _binary_cases,
    _multiclass_cases,
    _multilabel_cases,
)

import paddle
from paddle.metric.classification.auroc import (
    AUROC,
    BinaryAUROC,
    MulticlassAUROC,
    MultilabelAUROC,
)
from paddle.metric.functional.classification.auroc import (
    binary_auroc,
    multiclass_auroc,
    multilabel_auroc,
)
from paddle.metric.functional.classification.roc import binary_roc
from paddle.metric.metric import Metric

seed_all(42)


def _reference_sklearn_auroc_binary(
    preds, target, max_fpr=None, ignore_index=None
):
    preds = preds.flatten().numpy()
    target = target.flatten().numpy()
    if not ((preds > 0) & (preds < 1)).all():
        preds = sigmoid(preds)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_roc_auc_score(target, preds, max_fpr=max_fpr)


@pytest.mark.parametrize(
    "inputs",
    [_binary_cases[1], _binary_cases[2], _binary_cases[4], _binary_cases[5]],
)
class TestBinaryAUROC(MetricTester):
    """Test class for `BinaryAUROC` metric."""

    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_binary_auroc(self, inputs, ddp, max_fpr, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAUROC,
            reference_metric=partial(
                _reference_sklearn_auroc_binary,
                max_fpr=max_fpr,
                ignore_index=ignore_index,
            ),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_binary_auroc_functional(self, inputs, max_fpr, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_auroc,
            reference_metric=partial(
                _reference_sklearn_auroc_binary,
                max_fpr=max_fpr,
                ignore_index=ignore_index,
            ),
            metric_args={
                "max_fpr": max_fpr,
                "thresholds": None,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_auroc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_auroc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_auroc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAUROC,
            metric_functional=binary_auroc,
            metric_args={"thresholds": None},
            dtype=dtype,
        )

    @pytest.mark.parametrize(
        "threshold_fn",
        [lambda x: x, lambda x: x.numpy().tolist()],
        ids=["as tensor", "as list"],
    )
    def test_binary_auroc_threshold_arg(self, inputs, threshold_fn):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        for pred, true in zip(preds, target):
            _, _, t = binary_roc(pred, true, thresholds=None)
            ap1 = binary_auroc(pred, true, thresholds=None)
            ap2 = binary_auroc(
                pred, true, thresholds=threshold_fn(t.flip(axis=0))
            )
            assert paddle.allclose(x=ap1, y=ap2).item()


def _reference_sklearn_auroc_multiclass(
    preds, target, average="macro", ignore_index=None
):
    preds = np.moveaxis(preds.numpy(), 1, -1).reshape((-1, preds.shape[1]))
    target = target.numpy().flatten()
    if not ((preds > 0) & (preds < 1)).all():
        preds = softmax(preds, 1)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_roc_auc_score(
        target,
        preds,
        average=average,
        multi_class="ovr",
        labels=list(range(NUM_CLASSES)),
    )


@pytest.mark.parametrize(
    "inputs",
    [
        _multiclass_cases[1],
        _multiclass_cases[2],
        _multiclass_cases[4],
        _multiclass_cases[5],
    ],
)
class TestMulticlassAUROC(MetricTester):
    """Test class for `MulticlassAUROC` metric."""

    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multiclass_auroc(self, inputs, average, ddp, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAUROC,
            reference_metric=partial(
                _reference_sklearn_auroc_multiclass,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["macro", "weighted"])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multiclass_auroc_functional(self, inputs, average, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_auroc,
            reference_metric=partial(
                _reference_sklearn_auroc_multiclass,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_classes": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_auroc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == paddle.float16 and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(
                reason="half support for paddle.softmax on cpu not implemented"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_auroc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAUROC,
            metric_functional=multiclass_auroc,
            metric_args={"thresholds": None, "num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["macro", "weighted", None])
    def test_multiclass_auroc_threshold_arg(self, inputs, average):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = preds.softmax(dim=-1)
        for pred, true in zip(preds, target):
            pred = paddle.tensor(np.round(pred.numpy(), 2)) + 1e-06
            ap1 = multiclass_auroc(
                pred,
                true,
                num_classes=NUM_CLASSES,
                average=average,
                thresholds=None,
            )
            ap2 = multiclass_auroc(
                pred,
                true,
                num_classes=NUM_CLASSES,
                average=average,
                thresholds=paddle.linspace(0, 1, 100),
            )
            assert paddle.allclose(x=ap1, y=ap2).item()


def _reference_sklearn_auroc_multilabel(
    preds, target, average="macro", ignore_index=None
):
    if ignore_index is None:
        if preds.ndim > 2:
            target = target.transpose(2, 1).reshape(-1, NUM_CLASSES)
            preds = preds.transpose(2, 1).reshape(-1, NUM_CLASSES)
        target = target.numpy()
        preds = preds.numpy()
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        return sk_roc_auc_score(target, preds, average=average, max_fpr=None)
    if average == "micro":
        return _reference_sklearn_auroc_binary(
            preds.flatten(),
            target.flatten(),
            max_fpr=None,
            ignore_index=ignore_index,
        )
    res = [
        _reference_sklearn_auroc_binary(
            preds[:, i], target[:, i], max_fpr=None, ignore_index=ignore_index
        )
        for i in range(NUM_CLASSES)
    ]
    if average == "macro":
        return np.array(res)[~np.isnan(res)].mean()
    if average == "weighted":
        weights = (
            (target == 1).sum([0, 2])
            if target.ndim == 3
            else (target == 1).sum(0)
        ).numpy()
        weights = weights / sum(weights)
        return (np.array(res) * weights)[~np.isnan(res)].sum()
    return res


@pytest.mark.parametrize(
    "inputs",
    [
        _multilabel_cases[1],
        _multilabel_cases[2],
        _multilabel_cases[4],
        _multilabel_cases[5],
    ],
)
class TestMultilabelAUROC(MetricTester):
    """Test class for `MultilabelAUROC` metric."""

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multilabel_auroc(self, inputs, ddp, average, ignore_index):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelAUROC,
            reference_metric=partial(
                _reference_sklearn_auroc_multilabel,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ignore_index", [None, -1])
    def test_multilabel_auroc_functional(self, inputs, average, ignore_index):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_auroc,
            reference_metric=partial(
                _reference_sklearn_auroc_multilabel,
                average=average,
                ignore_index=ignore_index,
            ),
            metric_args={
                "thresholds": None,
                "num_labels": NUM_CLASSES,
                "average": average,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_auroc_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_auroc_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if dtype == paddle.float16 and not ((preds > 0) & (preds < 1)).all():
            pytest.xfail(
                reason="half support for paddle.softmax on cpu not implemented"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_auroc_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAUROC,
            metric_functional=multilabel_auroc,
            metric_args={"thresholds": None, "num_labels": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_auroc_threshold_arg(self, inputs, average):
        """Test that different types of `thresholds` argument lead to same result."""
        preds, target = inputs
        if (preds < 0).any():
            preds = sigmoid(preds)
        for pred, true in zip(preds, target):
            pred = paddle.tensor(np.round(pred.numpy(), 1)) + 1e-06
            ap1 = multilabel_auroc(
                pred,
                true,
                num_labels=NUM_CLASSES,
                average=average,
                thresholds=None,
            )
            ap2 = multilabel_auroc(
                pred,
                true,
                num_labels=NUM_CLASSES,
                average=average,
                thresholds=paddle.linspace(0, 1, 100),
            )
            assert paddle.allclose(x=ap1, y=ap2).item()


@pytest.mark.parametrize(
    "metric",
    [
        BinaryAUROC,
        partial(MulticlassAUROC, num_classes=NUM_CLASSES),
        partial(MultilabelAUROC, num_labels=NUM_CLASSES),
    ],
)
@pytest.mark.parametrize(
    "thresholds", [None, 100, [0.3, 0.5, 0.7, 0.9], paddle.linspace(0, 1, 10)]
)
def test_valid_input_thresholds(recwarn, metric, thresholds):
    """Test valid formats of the threshold argument."""
    metric(thresholds=thresholds)
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


@pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
def test_corner_case_max_fpr(max_fpr):
    """Check that metric returns 0 when one class is missing and `max_fpr` is set."""
    preds = paddle.tensor([0.1, 0.2, 0.3, 0.4])
    target = paddle.tensor([0, 0, 0, 0])
    metric = BinaryAUROC(max_fpr=max_fpr)
    assert metric(preds, target) == 0.0
    preds = paddle.tensor([0.5, 0.6, 0.7, 0.8])
    target = paddle.tensor([1, 1, 1, 1])
    metric = BinaryAUROC(max_fpr=max_fpr)
    assert metric(preds, target) == 0.0


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryAUROC, {"task": "binary"}),
        (MulticlassAUROC, {"task": "multiclass", "num_classes": 3}),
        (MultilabelAUROC, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=AUROC):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
