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

from typing_extensions import Literal

import paddle
from paddle.metric.functional.classification.roc import (
    binary_roc,
    multiclass_roc,
    multilabel_roc,
)
from paddle.metric.utils.enums import ClassificationTask


def _binary_eer_compute(
    fpr: paddle.Tensor, tpr: paddle.Tensor
) -> paddle.Tensor:
    """Compute Equal Error Rate (EER) for binary classification task."""
    diff = fpr - (1 - tpr)
    idx = paddle.argmin(paddle.abs(diff))
    return (fpr[idx] + (1 - tpr[idx])) / 2


def _eer_compute(
    fpr: paddle.Tensor | list[paddle.Tensor],
    tpr: paddle.Tensor | list[paddle.Tensor],
) -> paddle.Tensor:
    """Compute Equal Error Rate (EER)."""
    if (
        isinstance(fpr, paddle.Tensor)
        and isinstance(tpr, paddle.Tensor)
        and fpr.ndim == 1
    ):
        return _binary_eer_compute(fpr, tpr)
    return paddle.stack([_binary_eer_compute(f, t) for f, t in zip(fpr, tpr)])


def binary_eer(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Equal Error Rate (EER) for binary classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + \\text{FRR}}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations

    Returns:
        A single scalar with the eer score

    Example:
        >>> from paddle.metric.functional.classification import binary_eer
        >>> preds = paddle.to_tensor([0, 0.5, 0.7, 0.8])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> binary_eer(preds, target, thresholds=None)
        tensor(0.5000)
        >>> binary_eer(preds, target, thresholds=5)
        tensor(0.7500)

    """
    fpr, tpr, _ = binary_roc(
        preds, target, thresholds, ignore_index, validate_args
    )
    return _eer_compute(fpr, tpr)


def multiclass_eer(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    average: Literal["micro", "macro"] | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Equal Error Rate (EER) for multiclass classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + (1 - \\text{FRR})}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.
        average:
            If aggregation of should be applied. The aggregation is applied to underlying ROC curves.
            By default, eer is not aggregated and a score for each class is returned. If `average` is set to ``"micro"``
            , the metric will aggregate the curves by one hot encoding the targets and flattening the predictions,
            considering all classes jointly as a binary problem. If `average` is set to ``"macro"``, the metric will
            aggregate the curves by first interpolating the curves from each class at a combined set of thresholds and
            then average over the classwise interpolated curves. See `averaging curve objects`_ for more info on the
            different averaging methods.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with eer score per class.
        If `average="macro"|"micro"` then a single scalar is returned.


    Example:
        >>> from paddle.metric.functional.classification import multiclass_eer
        >>> preds = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> multiclass_eer(preds, target, num_classes=5, average="macro", thresholds=None)
        tensor(0.4667)
        >>> multiclass_eer(preds, target, num_classes=5, average=None, thresholds=None)
        tensor([0.0000, 0.0000, 0.6667, 0.6667, 1.0000])
        >>> multiclass_eer(preds, target, num_classes=5, average="macro", thresholds=5)
        tensor(0.4667)
        >>> multiclass_eer(preds, target, num_classes=5, average=None, thresholds=5)
        tensor([0.0000, 0.0000, 0.6667, 0.6667, 1.0000])

    """
    fpr, tpr, _ = multiclass_roc(
        preds,
        target,
        num_classes,
        thresholds,
        average,
        ignore_index,
        validate_args,
    )
    return _eer_compute(fpr, tpr)


def multilabel_eer(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Equal Error Rate (EER) for multilabel classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + (1 - \\text{FRR})}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        A 1d tensor of shape (n_classes, ) will be returned with eer score per label.

    Example:
        >>> from paddle.metric.functional.classification import multilabel_eer
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = paddle.to_tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_eer(preds, target, num_labels=3, thresholds=None)
        tensor([0.5000, 0.5000, 0.1667])
        >>> multilabel_eer(preds, target, num_labels=3, thresholds=5)
        tensor([0.5000, 0.7500, 0.1667])

    """
    fpr, tpr, _ = multilabel_roc(
        preds, target, num_labels, thresholds, ignore_index, validate_args
    )
    return _eer_compute(fpr, tpr)


def eer(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: int | list[float] | paddle.Tensor | None = None,
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["micro", "macro"] | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor | list[paddle.Tensor]:
    """Compute Equal Error Rate (EER) metric.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_eer`,
    :func:`~paddle.metric.functional.classification.multiclass_eer` and
    :func:`~paddle.metric.functional.classification.multilabel_eer` for the specific details of
    each argument influence and examples.

    Args:
        preds: Predictions from model (logits or probabilities)
        target: Ground truth labels
        task: Type of task, either 'binary', 'multiclass' or 'multilabel'
        thresholds: Thresholds used for computing the ROC curve
        num_classes: Number of classes (for multiclass task)
        num_labels: Number of labels (for multilabel task)
        average: Method to average EER over multiple classes/labels
        ignore_index: Specify a target value that is ignored
        validate_args: Bool indicating whether to validate input arguments

    Legacy Example:
        >>> from paddle.metric.functional.classification import eer
        >>> preds = paddle.to_tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = paddle.to_tensor([0, 0, 1, 1, 1])
        >>> eer(preds, target, task='binary')
        tensor(0.5833)

        >>> preds = paddle.to_tensor(
        ...     [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.85, 0.05, 0.10], [0.10, 0.10, 0.80]]
        ... )
        >>> target = paddle.to_tensor([0, 1, 1, 2, 2])
        >>> eer(
        ...     preds,
        ...     target,
        ...     task='multiclass',
        ...     num_classes=3,
        ... )
        tensor([0.0000, 0.4167, 0.4167])

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_eer(
            preds, target, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_eer(
            preds,
            target,
            num_classes,
            thresholds,
            average,
            ignore_index,
            validate_args,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_eer(
            preds, target, num_labels, thresholds, ignore_index, validate_args
        )
    raise ValueError(f"Task {task} not supported.")
