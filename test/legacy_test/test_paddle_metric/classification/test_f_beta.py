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
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1_score,
    fbeta_score as sk_fbeta_score,
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
    _multiclass_cases,
    _multilabel_cases,
)

import paddle
from paddle.metric.classification.f_beta import (
    BinaryF1Score,
    BinaryFBetaScore,
    F1Score,
    FBetaScore,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MultilabelF1Score,
    MultilabelFBetaScore,
)
from paddle.metric.functional.classification.f_beta import (
    binary_f1_score,
    binary_fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from paddle.metric.metric import Metric

seed_all(42)


def _reference_sklearn_fbeta_score_binary(
    preds, target, sk_fn, ignore_index, multidim_average, zero_division=0
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
        return sk_fn(target, preds, zero_division=zero_division)
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        res.append(sk_fn(true, pred, zero_division=zero_division))
    return np.stack(res)


@pytest.mark.parametrize("inputs", _binary_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (BinaryF1Score, binary_f1_score, sk_f1_score),
        (
            partial(BinaryFBetaScore, beta=2.0),
            partial(binary_fbeta_score, beta=2.0),
            partial(sk_fbeta_score, beta=2.0),
        ),
    ],
    ids=["f1", "fbeta"],
)
class TestBinaryFBetaScore(MetricTester):
    """Test class for `BinaryFBetaScore` metric."""

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_fbeta_score(
        self,
        ddp,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        zero_division,
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_binary,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_binary_fbeta_score_functional(
        self,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        zero_division,
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
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_binary,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                zero_division=zero_division,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "zero_division": zero_division,
            },
        )

    def test_binary_fbeta_score_differentiability(
        self, inputs, module, functional, compare
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_fbeta_score_half_cpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_fbeta_score_half_gpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_fbeta_score_multiclass(
    preds,
    target,
    sk_fn,
    ignore_index,
    multidim_average,
    average,
    zero_division=0,
):
    if preds.ndim == target.ndim + 1:
        preds = paddle.argmax(preds, 1)
    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()
        target, preds = remove_ignore_index(
            target=target, preds=preds, ignore_index=ignore_index
        )
        return sk_fn(
            target,
            preds,
            average=average,
            labels=list(range(NUM_CLASSES)) if average is None else None,
            zero_division=zero_division,
        )
    preds = preds.numpy()
    target = target.numpy()
    res = []
    for pred, true in zip(preds, target):
        pred = pred.flatten()
        true = true.flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        if len(pred) == 0 and average == "weighted":
            r = 0.0
        else:
            r = sk_fn(
                true,
                pred,
                average=average,
                labels=list(range(NUM_CLASSES)) if average is None else None,
                zero_division=zero_division,
            )
        res.append(0.0 if np.isnan(r).any() else r)
    return np.stack(res, 0)


@pytest.mark.parametrize("inputs", _multiclass_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (MulticlassF1Score, multiclass_f1_score, sk_f1_score),
        (
            partial(MulticlassFBetaScore, beta=2.0),
            partial(multiclass_fbeta_score, beta=2.0),
            partial(sk_fbeta_score, beta=2.0),
        ),
    ],
    ids=["f1", "fbeta"],
)
class TestMulticlassFBetaScore(MetricTester):
    """Test class for `MulticlassFBetaScore` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_fbeta_score(
        self,
        ddp,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        average,
        zero_division,
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multiclass_fbeta_score_functional(
        self,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        average,
        zero_division,
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
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
                "zero_division": zero_division,
            },
        )

    def test_multiclass_fbeta_score_differentiability(
        self, inputs, module, functional, compare
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_fbeta_score_half_cpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_fbeta_score_half_gpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
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
    ("metric_class", "metric_fn"),
    [
        (
            partial(MulticlassFBetaScore, beta=2.0),
            partial(multiclass_fbeta_score, beta=2.0),
        ),
        (MulticlassF1Score, multiclass_f1_score),
    ],
)
@pytest.mark.parametrize(
    ("k", "preds", "target", "average", "expected_fbeta", "expected_f1"),
    [
        (
            1,
            _mc_k_preds,
            _mc_k_target,
            "micro",
            paddle.tensor(2 / 3),
            paddle.tensor(2 / 3),
        ),
        (
            2,
            _mc_k_preds,
            _mc_k_target,
            "micro",
            paddle.tensor(1.0),
            paddle.tensor(1.0),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor(0.25),
            paddle.tensor(0.25),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor(0.75),
            paddle.tensor(0.75),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "micro",
            paddle.tensor(1.0),
            paddle.tensor(1.0),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor(0.2381),
            paddle.tensor(0.1667),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor(0.7963),
            paddle.tensor(0.7778),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "macro",
            paddle.tensor(1.0),
            paddle.tensor(1.0),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor(0.1786),
            paddle.tensor(0.125),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor(0.7361),
            paddle.tensor(0.75),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "weighted",
            paddle.tensor(1.0),
            paddle.tensor(1.0),
        ),
        (
            1,
            _mc_k_preds2,
            _mc_k_target2,
            "none",
            paddle.tensor([0.0, 0.0, 0.7143]),
            paddle.tensor([0.0, 0.0, 0.5]),
        ),
        (
            2,
            _mc_k_preds2,
            _mc_k_target2,
            "none",
            paddle.tensor([0.5556, 1.0, 0.8333]),
            paddle.tensor([0.6667, 1.0, 0.6667]),
        ),
        (
            3,
            _mc_k_preds2,
            _mc_k_target2,
            "none",
            paddle.tensor([1.0, 1.0, 1.0]),
            paddle.tensor([1.0, 1.0, 1.0]),
        ),
    ],
)
def test_top_k(
    metric_class,
    metric_fn,
    k: int,
    preds: paddle.Tensor,
    target: paddle.Tensor,
    average: str,
    expected_fbeta: paddle.Tensor,
    expected_f1: paddle.Tensor,
):
    """A comprehensive test to check that top_k works as expected."""
    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)
    result = expected_fbeta if class_metric.beta != 1.0 else expected_f1
    assert paddle.allclose(
        x=class_metric.compute(), y=result, atol=0.0001, rtol=0.0001
    ).item()
    assert paddle.allclose(
        x=metric_fn(preds, target, top_k=k, average=average, num_classes=3),
        y=result,
        atol=0.0001,
        rtol=0.0001,
    ).item()


@pytest.mark.parametrize("num_classes", [5])
def test_multiclassf1score_with_top_k(num_classes):
    """Test that F1 score increases monotonically with top_k and equals 1 when top_k equals num_classes.

    Args:
        num_classes: Number of classes in the classification task.

    The test verifies two properties:
    1. F1 score increases or stays the same as top_k increases
    2. F1 score equals 1 when top_k equals num_classes

    """
    preds = paddle.randn(200, num_classes).softmax(dim=-1)
    target = paddle.randint(low=0, high=num_classes, shape=(200,))
    previous_score = 0.0
    for k in range(1, num_classes + 1):
        f1_score = MulticlassF1Score(
            num_classes=num_classes, top_k=k, average="macro"
        )
        score = f1_score(preds, target)
        assert score >= previous_score, (
            f"F1 score did not increase for top_k={k}"
        )
        previous_score = score
        if k == num_classes:
            assert paddle.isclose(score, paddle.tensor(1.0)), (
                f"F1 score is not 1 for top_k={k} when num_classes={num_classes}"
            )


def test_multiclass_f1_score_top_k_equivalence():
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/1653.

    Test that top-k F1 score is equivalent to corrected top-1 F1 score.
    """
    num_classes = 5
    preds = paddle.randn(200, num_classes).softmax(dim=-1)
    target = paddle.randint(low=0, high=num_classes, shape=(200,))
    f1_val_top3 = MulticlassF1Score(
        num_classes=num_classes, top_k=3, average="macro"
    )
    f1_val_top1 = MulticlassF1Score(
        num_classes=num_classes, top_k=1, average="macro"
    )
    pred_top_3 = paddle.argsort(preds, axis=1, descending=True)[:, :3]
    pred_top_1 = pred_top_3[:, 0]
    target_in_top3 = (target.unsqueeze(1) == pred_top_3).any(dim=1)
    pred_corrected_top3 = paddle.where(target_in_top3, target, pred_top_1)
    score_top3 = f1_val_top3(preds, target)
    score_corrected = f1_val_top1(pred_corrected_top3, target)
    assert paddle.isclose(score_top3, score_corrected), (
        f"Top-3 F1 score ({score_top3}) does not match corrected top-1 F1 score ({score_corrected})"
    )


def _reference_sklearn_fbeta_score_multilabel_global(
    preds, target, sk_fn, ignore_index, average, zero_division
):
    if average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        target, preds = remove_ignore_index(
            target=target, preds=preds, ignore_index=ignore_index
        )
        return sk_fn(target, preds, zero_division=zero_division)
    fbeta_score, weights = [], []
    for i in range(preds.shape[1]):
        pred, true = preds[:, i].flatten(), target[:, i].flatten()
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        fbeta_score.append(sk_fn(true, pred, zero_division=zero_division))
        confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
        weights.append(confmat[1, 1] + confmat[1, 0])
    res = np.stack(fbeta_score, axis=0)
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


def _reference_sklearn_fbeta_score_multilabel_local(
    preds, target, sk_fn, ignore_index, average, zero_division
):
    fbeta_score, weights = [], []
    for i in range(preds.shape[0]):
        if average == "micro":
            pred, true = preds[i].flatten(), target[i].flatten()
            true, pred = remove_ignore_index(
                target=true, preds=pred, ignore_index=ignore_index
            )
            fbeta_score.append(sk_fn(true, pred, zero_division=zero_division))
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            weights.append(confmat[1, 1] + confmat[1, 0])
        else:
            scores, w = [], []
            for j in range(preds.shape[1]):
                pred, true = preds[i, j], target[i, j]
                true, pred = remove_ignore_index(
                    target=true, preds=pred, ignore_index=ignore_index
                )
                scores.append(sk_fn(true, pred, zero_division=zero_division))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                w.append(confmat[1, 1] + confmat[1, 0])
            fbeta_score.append(np.stack(scores))
            weights.append(np.stack(w))
    if average == "micro":
        return np.array(fbeta_score)
    res = np.stack(fbeta_score, 0)
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


def _reference_sklearn_fbeta_score_multilabel(
    preds,
    target,
    sk_fn,
    ignore_index,
    multidim_average,
    average,
    zero_division=0,
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if ignore_index is None and multidim_average == "global":
        return sk_fn(
            target.transpose(0, 2, 1).reshape(-1, NUM_CLASSES),
            preds.transpose(0, 2, 1).reshape(-1, NUM_CLASSES),
            average=average,
            zero_division=zero_division,
        )
    if multidim_average == "global":
        return _reference_sklearn_fbeta_score_multilabel_global(
            preds, target, sk_fn, ignore_index, average, zero_division
        )
    return _reference_sklearn_fbeta_score_multilabel_local(
        preds, target, sk_fn, ignore_index, average, zero_division
    )


@pytest.mark.parametrize("inputs", _multilabel_cases)
@pytest.mark.parametrize(
    ("module", "functional", "compare"),
    [
        (MultilabelF1Score, multilabel_f1_score, sk_f1_score),
        (
            partial(MultilabelFBetaScore, beta=2.0),
            partial(multilabel_fbeta_score, beta=2.0),
            partial(sk_fbeta_score, beta=2.0),
        ),
    ],
    ids=["f1", "fbeta"],
)
class TestMultilabelFBetaScore(MetricTester):
    """Test class for `MultilabelFBetaScore` metric."""

    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_fbeta_score(
        self,
        ddp,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        average,
        zero_division,
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
            metric_class=module,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "zero_division": zero_division,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("zero_division", [0, 1])
    def test_multilabel_fbeta_score_functional(
        self,
        inputs,
        module,
        functional,
        compare,
        ignore_index,
        multidim_average,
        average,
        zero_division,
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
            metric_functional=functional,
            reference_metric=partial(
                _reference_sklearn_fbeta_score_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
                zero_division=zero_division,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "zero_division": zero_division,
            },
        )

    def test_multilabel_fbeta_score_differentiability(
        self, inputs, module, functional, compare
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_fbeta_score_half_cpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_fbeta_score_half_gpu(
        self, inputs, module, functional, compare, dtype
    ):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


def test_corner_case():
    """Issue: https://github.com/Lightning-AI/paddlemetrics/issues/1664."""
    target = paddle.tensor([2, 1, 0, 0])
    preds = paddle.tensor([2, 1, 0, 1])
    for i in range(3, 9):
        f1_score = MulticlassF1Score(num_classes=i, average="macro")
        res = f1_score(preds, target)
        assert res == paddle.tensor([0.77777779])


@pytest.mark.parametrize(
    ("metric", "kwargs", "base_metric"),
    [
        (BinaryF1Score, {"task": "binary"}, F1Score),
        (MulticlassF1Score, {"task": "multiclass", "num_classes": 3}, F1Score),
        (MultilabelF1Score, {"task": "multilabel", "num_labels": 3}, F1Score),
        (None, {"task": "not_valid_task"}, F1Score),
        (BinaryFBetaScore, {"task": "binary", "beta": 2.0}, FBetaScore),
        (
            MulticlassFBetaScore,
            {"task": "multiclass", "num_classes": 3, "beta": 2.0},
            FBetaScore,
        ),
        (
            MultilabelFBetaScore,
            {"task": "multilabel", "num_labels": 3, "beta": 2.0},
            FBetaScore,
        ),
        (None, {"task": "not_valid_task"}, FBetaScore),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
