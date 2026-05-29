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
from paddle.metric.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    ConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from paddle.metric.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from paddle.metric.metric import Metric

seed_all(42)


def _reference_sklearn_confusion_matrix_binary(
    preds, target, normalize=None, ignore_index=None
):
    preds = preds.view(-1).numpy()
    target = target.view(-1).numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_confusion_matrix(
        y_true=target, y_pred=preds, labels=[0, 1], normalize=normalize
    )


@pytest.mark.parametrize("inputs", _binary_cases)
class TestBinaryConfusionMatrix(MetricTester):
    """Test class for `BinaryConfusionMatrix` metric."""

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_binary_confusion_matrix(
        self, inputs, ddp, normalize, ignore_index
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryConfusionMatrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_binary,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_binary_confusion_matrix_functional(
        self, inputs, normalize, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_confusion_matrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_binary,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "threshold": THRESHOLD,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_binary_confusion_matrix_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryConfusionMatrix,
            metric_functional=binary_confusion_matrix,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_confusion_matrix_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryConfusionMatrix,
            metric_functional=binary_confusion_matrix,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_binary_confusion_matrix_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryConfusionMatrix,
            metric_functional=binary_confusion_matrix,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _reference_sklearn_confusion_matrix_multiclass(
    preds, target, normalize=None, ignore_index=None
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        preds = np.argmax(preds, axis=1)
    preds = preds.flatten()
    target = target.flatten()
    target, preds = remove_ignore_index(
        target=target, preds=preds, ignore_index=ignore_index
    )
    return sk_confusion_matrix(
        y_true=target,
        y_pred=preds,
        normalize=normalize,
        labels=list(range(NUM_CLASSES)),
    )


@pytest.mark.parametrize("inputs", _multiclass_cases)
class TestMulticlassConfusionMatrix(MetricTester):
    """Test class for `MultiClassConfusionMatrix` metric."""

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multiclass_confusion_matrix(
        self, inputs, ddp, normalize, ignore_index
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MulticlassConfusionMatrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_multiclass,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multiclass_confusion_matrix_functional(
        self, inputs, normalize, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multiclass_confusion_matrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_multiclass,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_multiclass_confusion_matrix_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MulticlassConfusionMatrix,
            metric_functional=multiclass_confusion_matrix,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_confusion_matrix_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MulticlassConfusionMatrix,
            metric_functional=multiclass_confusion_matrix,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multiclass_confusion_matrix_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MulticlassConfusionMatrix,
            metric_functional=multiclass_confusion_matrix,
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
        multiclass_confusion_matrix(
            preds, target, num_classes=NUM_CLASSES, ignore_index=ignore_index
        )


def test_multiclass_overflow():
    """Test that multiclass computations does not overflow even on byte inputs."""
    preds = paddle.randint(low=0, high=20, shape=(100,)).byte()
    target = paddle.randint(low=0, high=20, shape=(100,)).byte()
    m = MulticlassConfusionMatrix(num_classes=20)
    res = m(preds, target)
    compare = sk_confusion_matrix(target, preds)
    assert paddle.allclose(x=res, y=paddle.tensor(compare)).item()


def _reference_sklearn_confusion_matrix_multilabel(
    preds, target, normalize=None, ignore_index=None
):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = np.moveaxis(preds, 1, -1).reshape((-1, preds.shape[1]))
    target = np.moveaxis(target, 1, -1).reshape((-1, target.shape[1]))
    confmat = []
    for i in range(preds.shape[1]):
        pred, true = preds[:, i], target[:, i]
        true, pred = remove_ignore_index(
            target=true, preds=pred, ignore_index=ignore_index
        )
        confmat.append(
            sk_confusion_matrix(true, pred, normalize=normalize, labels=[0, 1])
        )
    return np.stack(confmat, axis=0)


@pytest.mark.parametrize("inputs", _multilabel_cases)
class TestMultilabelConfusionMatrix(MetricTester):
    """Test class for `MultilabelConfusionMatrix` metric."""

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    @pytest.mark.parametrize(
        "ddp", [pytest.param(True, marks=pytest.mark.DDP), False]
    )
    def test_multilabel_confusion_matrix(
        self, inputs, ddp, normalize, ignore_index
    ):
        """Test class implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MultilabelConfusionMatrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_multilabel,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    @pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
    @pytest.mark.parametrize("ignore_index", [None, -1, 0])
    def test_multilabel_confusion_matrix_functional(
        self, inputs, normalize, ignore_index
    ):
        """Test functional implementation of metric."""
        preds, target = inputs
        if ignore_index is not None:
            target = inject_ignore_index(target, ignore_index)
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=multilabel_confusion_matrix,
            reference_metric=partial(
                _reference_sklearn_confusion_matrix_multilabel,
                normalize=normalize,
                ignore_index=ignore_index,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "normalize": normalize,
                "ignore_index": ignore_index,
            },
        )

    def test_multilabel_confusion_matrix_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MultilabelConfusionMatrix,
            metric_functional=multilabel_confusion_matrix,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_confusion_matrix_dtype_cpu(self, inputs, dtype):
        """Test dtype support of the metric on CPU."""
        preds, target = inputs
        if not True and (preds < 0).any() and dtype == paddle.float16:
            pytest.xfail(
                reason="paddle.sigmoid in metric does not support cpu + half precision for torch<2.1"
            )
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=MultilabelConfusionMatrix,
            metric_functional=multilabel_confusion_matrix,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(
        not paddle.cuda.is_available(), reason="test requires cuda"
    )
    @pytest.mark.parametrize("dtype", [paddle.float16, paddle.float64])
    def test_multilabel_confusion_matrix_dtype_gpu(self, inputs, dtype):
        """Test dtype support of the metric on GPU."""
        preds, target = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=MultilabelConfusionMatrix,
            metric_functional=multilabel_confusion_matrix,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.parametrize("num_labels", [2, NUM_CLASSES])
    def test_multilabel_confusion_matrix_plot(self, num_labels, inputs):
        """Test multilabel cm plots."""
        multi_label_confusion_matrix = MultilabelConfusionMatrix(
            num_labels=num_labels
        )
        preds = target = paddle.ones(1, num_labels).int()
        multi_label_confusion_matrix.update(preds, target)
        fig, ax = multi_label_confusion_matrix.plot()
        assert fig is not None
        assert ax is not None


def test_warning_on_nan():
    """Test that a warning is given if division by zero happens during normalization of confusion matrix."""
    preds = paddle.randint(low=0, high=3, shape=(20,))
    target = paddle.randint(low=0, high=3, shape=(20,))
    with pytest.warns(
        UserWarning,
        match=".* NaN values found in confusion matrix have been replaced with zeros.",
    ):
        multiclass_confusion_matrix(
            preds, target, num_classes=5, normalize="true"
        )


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (BinaryConfusionMatrix, {"task": "binary"}),
        (MulticlassConfusionMatrix, {"task": "multiclass", "num_classes": 3}),
        (MultilabelConfusionMatrix, {"task": "multilabel", "num_labels": 3}),
        (None, {"task": "not_valid_task"}),
    ],
)
def test_wrapper_class(metric, kwargs, base_metric=ConfusionMatrix):
    """Test the wrapper class."""
    assert issubclass(base_metric, Metric)
    if metric is None:
        with pytest.raises(ValueError, match="Invalid *"):
            base_metric(**kwargs)
    else:
        instance = base_metric(**kwargs)
        assert isinstance(instance, metric)
        assert isinstance(instance, Metric)
