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
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    confusion_matrix as sk_confusion_matrix,
)
from unittests import NUM_CLASSES, THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import (
    MetricTester,
    inject_ignore_index,
    remove_ignore_index,
)
from unittests.classification._inputs import (
    _binary_cases,
    _input_binary,
    _multiclass_cases,
    _multilabel_cases,
)

import paddle
from paddle.metric.classification.accuracy import (
    Accuracy,
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
)
from paddle.metric.functional.classification.accuracy import (
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from paddle.metric.metric import Metric

seed_all(42)


def _reference_sklearn_accuracy(target, preds):
    score = sk_accuracy(target, preds)
    return score if not np.isnan(score) else 0.0


def _reference_sklearn_accuracy_binary(
    preds, target, ignore_index, multidim_average
):
    if multidim_average == "global":
        preds = preds.view(-1).numpy()
        target = target.view(-1).numpy()
    else:
        preds = preds.numpy()
        target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    if multidim_average == "global":
        target, preds = remove_ignore_index(
            target=target, preds=preds, ignore_index=ignore_index
        )
        return _reference_sklearn_accuracy(target, preds)
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        res.append(_reference_sklearn_accuracy(true, pred))
    return np.stack(res)


def test_accuracy_functional_raises_invalid_task():
    """Tests accuracy task enum from functional.accuracy."""
    preds, target = _input_binary
    task = "NotValidTask"
    ignore_index = None
    multidim_average = "global"
    with pytest.raises(ValueError, match="Invalid *"):
        accuracy(
            preds,
            target,
            threshold=THRESHOLD,
            task=task,
            ignore_index=ignore_index,
            multidim_average=multidim_average,
        )


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryAccuracy(MetricTester):
    """Test class for `BinaryAccuracy` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_binary_accuracy(self, ddp, inputs, ignore_index, multidim_average):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryAccuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_accuracy_functional(
        self, inputs, ignore_index, multidim_average
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_accuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_accuracy_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_accuracy_half_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_accuracy_half_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryAccuracy,
            metric_functional=binary_accuracy,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_accuracy_multiclass(
    preds, target, ignore_index, multidim_average, average
):
    if preds.ndim == target.ndim + 1:
        preds = paddle.argmax(preds, 1)
    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()
        target, preds = remove_ignore_index(
            target=target, preds=preds, ignore_index=ignore_index
        )
        if average == "micro":
            return _reference_sklearn_accuracy(target, preds)
        confmat = sk_confusion_matrix(
            target, preds, labels=list(range(NUM_CLASSES))
        )
        acc_per_class = confmat.diagonal() / confmat.sum(axis=1)
        acc_per_class[np.isnan(acc_per_class)] = 0.0
        if average == "macro":
            acc_per_class = acc_per_class[
                np.bincount(preds, minlength=NUM_CLASSES)
                + np.bincount(target, minlength=NUM_CLASSES)
                != 0.0
            ]
            return acc_per_class.mean()
        if average == "weighted":
            weights = confmat.sum(1)
            return (weights * acc_per_class / weights.sum()).sum()
        return acc_per_class
    preds = preds.numpy()
    target = target.numpy()
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        if average == "micro":
            res.append(_reference_sklearn_accuracy(true, pred))
        else:
            confmat = sk_confusion_matrix(
                true, pred, labels=list(range(NUM_CLASSES))
            )
            acc_per_class = confmat.diagonal() / confmat.sum(axis=1)
            acc_per_class[np.isnan(acc_per_class)] = 0.0
            if average == "macro":
                acc_per_class = acc_per_class[
                    np.bincount(pred, minlength=NUM_CLASSES)
                    + np.bincount(true, minlength=NUM_CLASSES)
                    != 0.0
                ]
                res.append(
                    acc_per_class.mean() if len(acc_per_class) > 0 else 0.0
                )
            elif average == "weighted":
                weights = confmat.sum(1)
                score = (weights * acc_per_class / weights.sum()).sum()
                res.append(0.0 if np.isnan(score) else score)
            else:
                res.append(acc_per_class)
    return np.stack(res, 0)


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassAccuracy(MetricTester):
    """Test class for `MulticlassAccuracy` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multiclass_accuracy(
        self, ddp, inputs, ignore_index, multidim_average, average
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassAccuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multiclass_accuracy_functional(
        self, inputs, ignore_index, multidim_average, average
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_accuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_multiclass,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    def test_multiclass_accuracy_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_accuracy_half_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_accuracy_half_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    @pytest.mark.parametrize(
        ("average", "use_deterministic_algorithms"),
        [
            (None),
            ("macro"),
            ("micro", False),
            ("micro"),
            ("weighted"),
        ],
    )
    def test_multiclass_accuracy_gpu_sync_points(
        self,
        inputs,
        dtype: paddle.dtype,
        average: str,
        use_deterministic_algorithms: bool,
    ):
        """Test GPU support of the metric, avoiding CPU sync points."""
        preds, target = inputs

        def wrapped_multiclass_accuracy(
            preds: paddle.Tensor, target: paddle.Tensor, num_classes: int
        ) -> paddle.Tensor:
            prev_sync_debug_mode = paddle.cuda.get_sync_debug_mode()
            paddle.cuda.set_sync_debug_mode("error")
            try:
                validate_args = False
                paddle.use_deterministic_algorithms(
                    mode=use_deterministic_algorithms
                )
                return multiclass_accuracy(
                    preds,
                    target,
                    num_classes,
                    validate_args=validate_args,
                    average=average,
                )
            finally:
                paddle.cuda.set_sync_debug_mode(prev_sync_debug_mode)

        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassAccuracy,
            metric_functional=wrapped_multiclass_accuracy,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    @pytest.mark.parametrize(
        ("average", "use_deterministic_algorithms"),
        [(None, False), ("macro", False), ("weighted", False)],
    )
    def test_multiclass_accuracy_gpu_sync_points_uptodate(
        self,
        inputs,
        dtype: paddle.dtype,
        average: str,
        use_deterministic_algorithms: bool,
    ):
        """Negative test for `test_multiclass_accuracy_gpu_sync_points`, to confirm completeness.

        Tests that `test_multiclass_accuracy_gpu_sync_points` is kept up to date, explicitly validating that known
        failures still fail, so that if they're fixed they must be added to
        `test_multiclass_accuracy_gpu_sync_points`.

        """
        with pytest.raises(
            RuntimeError, match="called a synchronizing CUDA operation"
        ):
            self.test_multiclass_accuracy_gpu_sync_points(
                inputs=inputs,
                dtype=dtype,
                average=average,
                use_deterministic_algorithms=use_deterministic_algorithms,
            )


_mc_k_target = paddle.tensor([0, 1, 2])
_mc_k_preds = paddle.tensor(
    [[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]]
)
_mc_k_targets2 = paddle.tensor([0, 0, 2])
_mc_k_preds2 = paddle.tensor(
    [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]]
)
_mc_k_targets3 = paddle.arange(0, 512) % 10
_mc_k_preds3 = paddle.rand(512, 10)
_mc_k_targets4 = _mc_k_targets3[:1]
_mc_k_preds4 = _mc_k_preds3[:1, :]
_mc_k_targets5 = paddle.randint(low=0, high=10, shape=(2, 50))
_mc_k_preds5 = paddle.rand(2, 10, 50)
_mc_k_targets6 = paddle.tensor([[3, 2], [1, 0]])
_mc_k_preds6 = paddle.tensor(
    [
        [[0.0, 0.1, 0.5, 0.4], [0.0, 0.2, 0.7, 0.1]],
        [[0.0, 0.4, 0.3, 0.3], [1.0, 0.0, 0.0, 0.0]],
    ]
).transpose(2, 1)


@pytest.mark.parametrize(
    ("k", "preds", "target", "average", "num_classes", "expected"),
    [
        (1, _mc_k_preds, _mc_k_target, "micro", 3, paddle.tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", 3, paddle.tensor(3 / 3)),
        (1, _mc_k_preds2, _mc_k_targets2, "macro", 3, paddle.tensor(1 / 2)),
        (2, _mc_k_preds2, _mc_k_targets2, "macro", 3, paddle.tensor(1 / 2)),
        (5, _mc_k_preds3, _mc_k_targets3, "macro", 10, paddle.tensor(0.5175)),
        (5, _mc_k_preds3, _mc_k_targets3, "micro", 10, paddle.tensor(0.5176)),
        (5, _mc_k_preds4, _mc_k_targets4, "macro", 10, paddle.tensor(1.0)),
        (5, _mc_k_preds4, _mc_k_targets4, "micro", 10, paddle.tensor(1.0)),
        (5, _mc_k_preds5, _mc_k_targets5, "micro", 10, paddle.tensor(0.42)),
    ],
)
def test_top_k(k, preds, target, average, num_classes, expected):
    """A simple test to check that top_k works as expected."""
    class_metric = MulticlassAccuracy(
        top_k=k, average=average, num_classes=num_classes
    )
    class_metric.update(preds, target)
    assert paddle.isclose(
        class_metric.compute(), expected, rtol=0.0001, atol=0.0001
    )
    assert paddle.isclose(
        multiclass_accuracy(
            preds, target, top_k=k, average=average, num_classes=num_classes
        ),
        expected,
        rtol=0.0001,
        atol=0.0001,
    )


@pytest.mark.parametrize(
    ("preds", "target", "k", "expected"),
    [
        (_mc_k_preds6, _mc_k_targets6, 1, paddle.tensor(0.6667)),
        (_mc_k_preds6, _mc_k_targets6, 2, paddle.tensor(1.0)),
        (_mc_k_preds6, _mc_k_targets6, 3, paddle.tensor(1.0)),
        (_mc_k_preds6, _mc_k_targets6, 4, paddle.tensor(1.0)),
    ],
)
def test_top_k_with_ignore_index(preds, target, k, expected):
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/3068."""
    num_classes = 4
    average = "micro"
    ignore_index = 0
    class_metric = Accuracy(
        task="multiclass",
        ignore_index=ignore_index,
        num_classes=num_classes,
        multidim_average="global",
        average=average,
        top_k=k,
    )
    class_metric.update(preds, target)
    assert paddle.isclose(
        class_metric.compute(), expected, rtol=0.0001, atol=0.0001
    )
    assert paddle.isclose(
        multiclass_accuracy(
            preds,
            target,
            num_classes=num_classes,
            average=average,
            top_k=k,
            ignore_index=ignore_index,
        ),
        expected,
        rtol=0.0001,
        atol=0.0001,
    )


@pytest.mark.parametrize("num_classes", [5])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_multiclass_accuracy_with_top_k(num_classes, average):
    """Test that Accuracy increases monotonically with top_k and equals 1 when top_k equals num_classes.

    Args:
        num_classes: Number of classes in the classification task.
        average: The averaging method to use (macro, micro, weighted).

    The test verifies two properties:
    1. Accuracy increases or stays the same as top_k increases
    2. Accuracy equals 1 when top_k equals num_classes

    """
    preds = paddle.randn(200, num_classes).softmax(dim=-1)
    target = paddle.randint(low=0, high=num_classes, shape=(200,))
    previous_accuracy = 0.0
    for k in range(1, num_classes + 1):
        accuracy_score = MulticlassAccuracy(
            num_classes=num_classes, top_k=k, average=average
        )
        accuracy = accuracy_score(preds, target)
        assert accuracy >= previous_accuracy, (
            f"Accuracy did not increase for top_k={k}"
        )
        previous_accuracy = accuracy
        if k == num_classes:
            assert paddle.isclose(accuracy, paddle.tensor(1.0)), (
                f"Accuracy is not 1 for top_k={k} when num_classes={num_classes}"
            )


@pytest.mark.parametrize(("num_classes", "k"), [(5, 3), (10, 5)])
@pytest.mark.parametrize("average", ["macro", "micro", "weighted"])
def test_multiclass_accuracy_top_k_equivalence(num_classes, k, average):
    """Test that top-k Accuracy scores are equivalent to corrected top-1 scores.

    Args:
        num_classes: Number of classes in the classification task.
        k: The top-k value to test.
        average: The averaging method to use (macro, micro, weighted).

    """
    preds = paddle.randn(200, num_classes).softmax(dim=-1)
    target = paddle.randint(low=0, high=num_classes, shape=(200,))
    accuracy_top_k = MulticlassAccuracy(
        num_classes=num_classes, top_k=k, average=average
    )
    accuracy_top_1 = MulticlassAccuracy(
        num_classes=num_classes, top_k=1, average=average
    )
    pred_top_k = paddle.argsort(preds, axis=1, descending=True)[:, :k]
    pred_top_1 = pred_top_k[:, 0]
    target_in_top_k = (target.unsqueeze(1) == pred_top_k).any(dim=1)
    pred_corrected_top_k = paddle.where(target_in_top_k, target, pred_top_1)
    accuracy_score_top_k = accuracy_top_k(preds, target)
    accuracy_score_corrected = accuracy_top_1(pred_corrected_top_k, target)
    assert paddle.isclose(accuracy_score_top_k, accuracy_score_corrected), (
        f"Top-{k} Accuracy ({accuracy_score_top_k}) does not match corrected top-1 Accuracy ({accuracy_score_corrected})"
    )


def _reference_sklearn_accuracy_multilabel(
    preds, target, ignore_index, multidim_average, average
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if multidim_average == "global":
        if average == "micro":
            preds = preds.flatten()
            target = target.flatten()
            target, preds = remove_ignore_index(
                target=target, preds=preds, ignore_index=ignore_index
            )
            return _reference_sklearn_accuracy(target, preds)
        accuracy, weights = [], []
        for i in range(preds.shape[1]):
            pred, true = preds[:, i].flatten(), target[:, i].flatten()
            true, pred = remove_ignore_index(
                target=true, preds=pred, ignore_index=ignore_index
            )
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            accuracy.append(_reference_sklearn_accuracy(true, pred))
            weights.append(confmat[1, 1] + confmat[1, 0])
        res = np.stack(accuracy, axis=0)
        if average == "macro":
            return res.mean(0)
        if average == "weighted":
            weights = np.stack(weights, 0).astype(float)
            weights_norm = weights.sum(-1, keepdims=True)
            weights_norm[weights_norm == 0] = 1.0
            return (weights * res / weights_norm).sum(-1)
        if average is None or average == "none":
            return res
        return None
    accuracy, weights = [], []
    for i in range(preds.shape[0]):
        if average == "micro":
            pred, true = preds[i].flatten(), target[i].flatten()
            true, pred = remove_ignore_index(
                target=true, preds=pred, ignore_index=ignore_index
            )
            accuracy.append(_reference_sklearn_accuracy(true, pred))
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            weights.append(confmat[1, 1] + confmat[1, 0])
        else:
            scores, w = [], []
            for j in range(preds.shape[1]):
                pred, true = preds[i, j], target[i, j]
                true, pred = remove_ignore_index(
                    target=true, preds=pred, ignore_index=ignore_index
                )
                scores.append(_reference_sklearn_accuracy(true, pred))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                w.append(confmat[1, 1] + confmat[1, 0])
            accuracy.append(np.stack(scores))
            weights.append(np.stack(w))
    if average == "micro":
        return np.array(accuracy)
    res = np.stack(accuracy, 0)
    if average == "macro":
        return res.mean(-1)
    if average == "weighted":
        weights = np.stack(weights, 0).astype(float)
        weights_norm = weights.sum(-1, keepdims=True)
        weights_norm[weights_norm == 0] = 1.0
        return (weights * res / weights_norm).sum(-1)
    if average is None or average == "none":
        return res
    return None


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelAccuracy(MetricTester):
    """Test class for `MultilabelAccuracy` metric."""

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_accuracy(
        self, ddp, inputs, ignore_index, multidim_average, average
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelAccuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_accuracy_functional(
        self, inputs, ignore_index, multidim_average, average
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_accuracy,
            reference_metric=partial(
                _reference_sklearn_accuracy_multilabel,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    def test_multilabel_accuracy_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_accuracy_half_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_accuracy_half_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelAccuracy,
            metric_functional=multilabel_accuracy,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_corner_cases():
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/1691."""
    target = paddle.tensor([0, 1, 2, 0, 1, 2])
    preds = target
    metric = MulticlassAccuracy(num_classes=3, average="none", ignore_index=0)
    res = metric(preds, target)
    assert paddle.allclose(x=res, y=paddle.tensor([0.0, 1.0, 1.0])).item()
    metric = MulticlassAccuracy(num_classes=3, average="macro", ignore_index=0)
    res = metric(preds, target)
    assert res == 1.0
    metric_micro1 = MulticlassAccuracy(
        num_classes=None, average="micro", ignore_index=0
    )
    metric_micro2 = MulticlassAccuracy(
        num_classes=3, average="micro", ignore_index=0
    )
    res1 = metric_micro1(preds, target)
    res2 = metric_micro2(preds, target)
    assert res1 == res2


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryAccuracy, {"task": "binary"}),
        (MulticlassAccuracy, {"task": "multiclass", "num_classes": 3}),
        (MultilabelAccuracy, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=Accuracy):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
