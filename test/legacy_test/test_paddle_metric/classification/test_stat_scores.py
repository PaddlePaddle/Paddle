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
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from unittests import NUM_CLASSES, THRESHOLD
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
from paddle.metric.classification.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
    StatScores,
)
from paddle.metric.functional.classification.stat_scores import (
    _refine_preds_oh,
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
)
from paddle.metric.metric import Metric

seed_all(42)


def _reference_sklearn_stat_scores_binary(
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
        tn, fp, fn, tp = sk_confusion_matrix(
            y_true=target, y_pred=preds, labels=[0, 1]
        ).ravel()
        return np.array([tp, fp, tn, fn, tp + fn])
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        tn, fp, fn, tp = sk_confusion_matrix(
            y_true=true, y_pred=pred, labels=[0, 1]
        ).ravel()
        res.append(np.array([tp, fp, tn, fn, tp + fn]))
    return np.stack(res)


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryStatScores(MetricTester):
    """Test class for `BinaryStatScores` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_binary_stat_scores(
        self, ddp, inputs, ignore_index, multidim_average
    ):
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
            metric_class=BinaryStatScores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_stat_scores_functional(
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
            metric_functional=binary_stat_scores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_binary,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_stat_scores_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryStatScores,
            metric_functional=binary_stat_scores,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_stat_scores_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryStatScores,
            metric_functional=binary_stat_scores,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_stat_scores_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryStatScores,
            metric_functional=binary_stat_scores,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_stat_scores_multiclass_global(
    preds, target, ignore_index, average
):
    preds = preds.numpy().flatten()
    target = target.numpy().flatten()
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    confmat = sk_confusion_matrix(
        y_true=target, y_pred=preds, labels=list(range(NUM_CLASSES))
    )
    tp = np.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    tn = confmat.sum() - (fp + fn + tp)
    res = np.stack([tp, fp, tn, fn, tp + fn], 1)
    if average == "micro":
        return res.sum(0)
    if average == "macro":
        return res.mean(0)
    if average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
    if average is None or average == "none":
        return res
    return None


def _reference_sklearn_stat_scores_multiclass_local(
    preds, target, ignore_index, average
):
    preds = preds.numpy()
    target = target.numpy()
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        confmat = sk_confusion_matrix(
            y_true=true, y_pred=pred, labels=list(range(NUM_CLASSES))
        )
        tp = np.diag(confmat)
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
        r = np.stack([tp, fp, tn, fn, tp + fn], 1)
        if average == "micro":
            res.append(r.sum(0))
        elif average == "macro":
            res.append(r.mean(0))
        elif average == "weighted":
            w = tp + fn
            res.append((r * (w / w.sum()).reshape(-1, 1)).sum(0))
        elif average is None or average == "none":
            res.append(r)
    return np.stack(res, 0)


def _reference_sklearn_stat_scores_multiclass(
    preds, target, ignore_index, multidim_average, average
):
    if preds.ndim == target.ndim + 1:
        preds = paddle.argmax(preds, 1)
    if multidim_average == "global":
        return _reference_sklearn_stat_scores_multiclass_global(
            preds, target, ignore_index, average
        )
    return _reference_sklearn_stat_scores_multiclass_local(
        preds, target, ignore_index, average
    )


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassStatScores(MetricTester):
    """Test class for `MulticlassStatScores` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multiclass_stat_scores(
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
            metric_class=MulticlassStatScores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_multiclass,
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
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multiclass_stat_scores_functional(
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
            metric_functional=multiclass_stat_scores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_multiclass,
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

    def test_multiclass_stat_scores_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassStatScores,
            metric_functional=multiclass_stat_scores,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_stat_scores_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassStatScores,
            metric_functional=multiclass_stat_scores,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_stat_scores_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassStatScores,
            metric_functional=multiclass_stat_scores,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("preds", "target", "ignore_index", "error_message"),
    [
        (
            paddle.randint(low=0, high=NUM_CLASSES + 1, shape=(100,)),
            paddle.randint(low=0, high=NUM_CLASSES, shape=(100,)),
            None,
            f"Detected more unique values in `preds` than expected. Expected only {NUM_CLASSES}.*",
        ),
        (
            paddle.randint(low=0, high=NUM_CLASSES, shape=(100,)),
            paddle.randint(low=0, high=NUM_CLASSES + 1, shape=(100,)),
            None,
            f"Detected more unique values in `target` than expected. Expected only {NUM_CLASSES}.*",
        ),
        (
            paddle.randint(low=0, high=NUM_CLASSES + 2, shape=(100,)),
            paddle.randint(low=0, high=NUM_CLASSES, shape=(100,)),
            1,
            f"Detected more unique values in `preds` than expected. Expected only {NUM_CLASSES + 1}.*",
        ),
        (
            paddle.randint(low=0, high=NUM_CLASSES, shape=(100,)),
            paddle.randint(low=0, high=NUM_CLASSES + 2, shape=(100,)),
            1,
            f"Detected more unique values in `target` than expected. Expected only {NUM_CLASSES + 1}.*",
        ),
    ],
)
def test_raises_error_on_too_many_classes(
    preds, target, ignore_index, error_message
):
    """Test that an error is raised if the number of classes in preds or target is larger than expected."""
    with pytest.raises(RuntimeError, match=error_message):
        multiclass_stat_scores(
            preds, target, num_classes=NUM_CLASSES, ignore_index=ignore_index
        )


@pytest.mark.parametrize(
    ("top_k", "expected_result"),
    [
        (
            1,
            paddle.tensor(
                [[[0, 0, 1]], [[1, 0, 0]], [[0, 1, 0]], [[1, 0, 0]]],
                dtype=paddle.int32,
            ),
        ),
        (
            2,
            paddle.tensor(
                [[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]],
                dtype=paddle.int32,
            ),
        ),
        (
            3,
            paddle.tensor(
                [[[1, 0, 0]], [[0, 1, 0]], [[0, 1, 0]], [[0, 0, 1]]],
                dtype=paddle.int32,
            ),
        ),
    ],
)
def test_refine_preds_oh(top_k, expected_result):
    """Test the _refine_preds_oh function.

    This function tests the behavior of the _refine_preds_oh function with various top_k values
    and checks if the output matches the expected one-hot encoded results.

    Args:
        top_k: The number of top predictions to consider.
        expected_result: The expected one-hot encoded tensor result after refinement.

    """
    preds = paddle.tensor(
        [
            [[0.2917], [0.0682], [0.6401]],
            [[0.2582], [0.0614], [0.0704]],
            [[0.0725], [0.6015], [0.326]],
            [[0.465], [0.2448], [0.2902]],
        ]
    )
    preds_oh = paddle.tensor(
        [[[1, 0, 1]], [[1, 0, 1]], [[0, 1, 1]], [[1, 0, 1]]], dtype=paddle.int32
    )
    target = paddle.tensor([[0], [1], [1], [2]])
    result = _refine_preds_oh(preds, preds_oh, target, top_k)
    assert paddle.equal(result, expected_result), (
        f"Test failed for top_k={top_k}. Expected result: {expected_result}, but got: {result}"
    )


_mc_k_target = paddle.tensor([0, 1, 2])
_mc_k_preds = paddle.tensor(
    [[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]]
)
_mc_k_target2 = paddle.tensor([0, 1, 2, 0])
_mc_k_preds2 = paddle.tensor(
    [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2], [0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]
)


@pytest.mark.parametrize(
    ("k", "preds", "target", "average", "expected"),
    [
        (1, _mc_k_preds, _mc_k_target, "micro", paddle.tensor([2, 1, 5, 1, 3])),
        (2, _mc_k_preds, _mc_k_target, "micro", paddle.tensor([3, 0, 6, 0, 3])),
        (
            1,
            _mc_k_preds,
            _mc_k_target,
            None,
            paddle.tensor(
                [[0, 1, 1], [0, 1, 0], [2, 1, 2], [1, 0, 0], [1, 1, 1]]
            ),
        ),
        (
            2,
            _mc_k_preds,
            _mc_k_target,
            None,
            paddle.tensor(
                [[1, 1, 1], [0, 0, 0], [2, 2, 2], [0, 0, 0], [1, 1, 1]]
            ),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor([0.3333, 1.0, 1.6667, 1.0, 1.3333]),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor([1.0, 0.3333, 2.3333, 0.3333, 1.3333]),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor([1.3333, 0.0, 2.6667, 0.0, 1.3333]),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor([1, 3, 5, 3, 4]),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor([3, 1, 7, 1, 4]),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor([4, 0, 8, 0, 4]),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor([0.25, 1.0, 1.5, 1.25, 1.5]),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor([1.0, 0.25, 2.25, 0.5, 1.5]),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor([1.5, 0.0, 2.5, 0.0, 1.5]),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            None,
            paddle.tensor(
                [[0, 0, 1], [1, 0, 2], [1, 3, 1], [2, 1, 0], [2, 1, 1]]
            ),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            None,
            paddle.tensor(
                [[1, 1, 1], [0, 0, 1], [2, 3, 2], [1, 0, 0], [2, 1, 1]]
            ),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            None,
            paddle.tensor(
                [[2, 1, 1], [0, 0, 0], [2, 3, 3], [0, 0, 0], [2, 1, 1]]
            ),
        ),
    ],
)
def test_top_k_multiclass(k, preds, target, average, expected):
    """A simple test to check that top_k works as expected."""
    class_metric = MulticlassStatScores(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)
    assert paddle.allclose(
        x=class_metric.compute(), y=expected.T, atol=0.0001, rtol=0.0001
    ).item()
    assert paddle.allclose(
        x=multiclass_stat_scores(
            preds, target, top_k=k, average=average, num_classes=3
        ),
        y=expected.T,
        atol=0.0001,
        rtol=0.0001,
    ).item()


def test_top_k_ignore_index_multiclass():
    """Test that top_k argument works together with ignore_index."""
    preds_without = paddle.randn(10, 3).softmax(dim=-1)
    target_without = paddle.randint(low=0, high=3, shape=(10,))
    preds_with = paddle.concat(
        [preds_without, paddle.randn(10, 3).softmax(dim=-1)], 0
    )
    target_with = paddle.concat(
        [target_without, -100 * paddle.ones(10)], 0
    ).long()
    res_without = multiclass_stat_scores(
        preds_without, target_without, num_classes=3, average="micro", top_k=2
    )
    res_with = multiclass_stat_scores(
        preds_with,
        target_with,
        num_classes=3,
        average="micro",
        top_k=2,
        ignore_index=-100,
    )
    assert paddle.allclose(x=res_without, y=res_with).item()


def test_multiclass_overflow():
    """Test that multiclass computations does not overflow even on byte input."""
    preds = paddle.randint(low=0, high=20, shape=(100,)).byte()
    target = paddle.randint(low=0, high=20, shape=(100,)).byte()
    m = MulticlassStatScores(num_classes=20, average=None)
    res = m(preds, target)
    confmat = sk_confusion_matrix(target, preds)
    fp = confmat.sum(axis=0) - np.diag(confmat)
    fn = confmat.sum(axis=1) - np.diag(confmat)
    tp = np.diag(confmat)
    tn = confmat.sum() - (fp + fn + tp)
    compare = np.stack([tp, fp, tn, fn, tp + fn]).T
    assert paddle.allclose(x=res, y=paddle.tensor(compare)).item()


def _reference_sklearn_stat_scores_multilabel(
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
        stat_scores = []
        for i in range(preds.shape[1]):
            pred, true = preds[:, i].flatten(), target[:, i].flatten()
            true, pred = remove_ignore_index(
                target=true, preds=pred, ignore_index=ignore_index
            )
            tn, fp, fn, tp = sk_confusion_matrix(
                true, pred, labels=[0, 1]
            ).ravel()
            stat_scores.append(np.array([tp, fp, tn, fn, tp + fn]))
        res = np.stack(stat_scores, axis=0)
        if average == "micro":
            return res.sum(0)
        if average == "macro":
            return res.mean(0)
        if average == "weighted":
            w = res[:, 0] + res[:, 3]
            return (res * (w / w.sum()).reshape(-1, 1)).sum(0)
        if average is None or average == "none":
            return res
        return None
    stat_scores = []
    for i in range(preds.shape[0]):
        scores = []
        for j in range(preds.shape[1]):
            pred, true = preds[i, j], target[i, j]
            true, pred = remove_ignore_index(
                target=true, preds=pred, ignore_index=ignore_index
            )
            tn, fp, fn, tp = sk_confusion_matrix(
                true, pred, labels=[0, 1]
            ).ravel()
            scores.append(np.array([tp, fp, tn, fn, tp + fn]))
        stat_scores.append(np.stack(scores, 1))
    res = np.stack(stat_scores, 0)
    if average == "micro":
        return res.sum(-1)
    if average == "macro":
        return res.mean(-1)
    if average == "weighted":
        w = res[:, 0, :] + res[:, 3, :]
        return (res * (w / w.sum())[:, np.newaxis]).sum(-1)
    if average is None or average == "none":
        return np.moveaxis(res, 1, -1)
    return None


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelStatScores(MetricTester):
    """Test class for `MultilabelStatScores` metric."""

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multilabel_stat_scores(
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
            metric_class=MultilabelStatScores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_multilabel,
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

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", None])
    def test_multilabel_stat_scores_functional(
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
            metric_functional=multilabel_stat_scores,
            reference_metric=partial(
                _reference_sklearn_stat_scores_multilabel,
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

    def test_multilabel_stat_scores_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelStatScores,
            metric_functional=multilabel_stat_scores,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_stat_scores_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelStatScores,
            metric_functional=multilabel_stat_scores,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_stat_scores_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelStatScores,
            metric_functional=multilabel_stat_scores,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_support_for_int():
    """See issue: https://github.com/Lightning-AI/paddlemetrics/issues/1970."""
    seed_all(42)
    metric = MulticlassStatScores(
        num_classes=4,
        average="none",
        multidim_average="samplewise",
        ignore_index=0,
    )
    prediction = paddle.randint(low=0, high=4, shape=(1, 50, 50)).to(
        paddle.uint8
    )
    label = paddle.randint(low=0, high=4, shape=(1, 50, 50)).to(paddle.uint8)
    score = metric(preds=prediction, target=label)
    assert score.shape == (1, 4, 5)


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryStatScores, {"task": "binary"}),
        (MulticlassStatScores, {"task": "multiclass", "num_classes": 3}),
        (MultilabelStatScores, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=StatScores):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
