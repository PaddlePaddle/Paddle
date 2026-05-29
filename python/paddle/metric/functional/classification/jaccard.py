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
from paddle.metric.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from paddle.metric.utils.compute import _safe_divide
from paddle.metric.utils.enums import ClassificationTask


def _jaccard_index_reduce(
    confmat: paddle.Tensor,
    average: Literal["micro", "macro", "weighted", "none", "binary"] | None,
    ignore_index: int | None = None,
    zero_division: float = 0.0,
) -> paddle.Tensor:
    """Perform reduction of an un-normalized confusion matrix into jaccard score.

    Args:
        confmat: tensor with un-normalized confusionmatrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        zero_division:
            Value to replace when there is a division by zero. Should be `0` or `1`.

    """
    allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(
            f"The `average` has to be one of {allowed_average}, got {average}."
        )
    confmat = confmat.float()
    if average == "binary":
        return _safe_divide(
            confmat[1, 1],
            confmat[0, 1] + confmat[1, 0] + confmat[1, 1],
            zero_division=zero_division,
        )
    ignore_index_cond = (
        ignore_index is not None and 0 <= ignore_index < confmat.shape[0]
    )
    multilabel = confmat.ndim == 3
    if multilabel:
        num = confmat[:, 1, 1]
        denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
    else:
        num = paddle.diag(confmat)
        denom = confmat.sum(0) + confmat.sum(1) - num
    if average == "micro":
        num = num.sum()
        denom = denom.sum() - (
            denom[ignore_index] if ignore_index_cond else 0.0
        )
    jaccard = _safe_divide(num, denom, zero_division=zero_division)
    if average is None or average == "none" or average == "micro":
        return jaccard
    if average == "weighted":
        weights = (
            confmat[:, 1, 1] + confmat[:, 1, 0]
            if confmat.ndim == 3
            else confmat.sum(1)
        )
    else:
        weights = paddle.ones_like(jaccard)
        if ignore_index_cond:
            weights[ignore_index] = 0.0
        if not multilabel:
            weights[confmat.sum(1) + confmat.sum(0) == 0] = 0.0
    return (weights * jaccard / weights.sum()).sum()


def binary_jaccard_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> paddle.Tensor:
    """Calculate the Jaccard index for binary tasks.

    The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value to replace when there is a division by zero. Should be `0` or `1`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import binary_jaccard_index
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import binary_jaccard_index
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_jaccard_index(preds, target)
        tensor(0.5000)

    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(threshold, ignore_index)
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold, ignore_index
    )
    confmat = _binary_confusion_matrix_update(preds, target)
    return _jaccard_index_reduce(
        confmat, average="binary", zero_division=zero_division
    )


def _multiclass_jaccard_index_arg_validation(
    num_classes: int,
    ignore_index: int | None = None,
    average: Literal["micro", "macro", "weighted", "none"] | None = None,
) -> None:
    _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index)
    allowed_average = "micro", "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, but got {average}."
        )


def multiclass_jaccard_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> paddle.Tensor:
    """Calculate the Jaccard index for multiclass tasks.

    The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value to replace when there is a division by zero. Should be `0` or `1`.

    Example (pred is integer tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multiclass_jaccard_index
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from paddle.metric.functional.classification import multiclass_jaccard_index
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> multiclass_jaccard_index(preds, target, num_classes=3)
        tensor(0.6667)

    """
    if validate_args:
        _multiclass_jaccard_index_arg_validation(
            num_classes, ignore_index, average
        )
        _multiclass_confusion_matrix_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target = _multiclass_confusion_matrix_format(
        preds, target, ignore_index
    )
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _jaccard_index_reduce(
        confmat,
        average=average,
        ignore_index=ignore_index,
        zero_division=zero_division,
    )


def _multilabel_jaccard_index_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
) -> None:
    _multilabel_confusion_matrix_arg_validation(
        num_labels, threshold, ignore_index
    )
    allowed_average = "micro", "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, but got {average}."
        )


def multilabel_jaccard_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> paddle.Tensor:
    """Calculate the Jaccard index for multilabel tasks.

    The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division:
            Value to replace when there is a division by zero. Should be `0` or `1`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multilabel_jaccard_index
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multilabel_jaccard_index
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_jaccard_index(preds, target, num_labels=3)
        tensor(0.5000)

    """
    if validate_args:
        _multilabel_jaccard_index_arg_validation(
            num_labels, threshold, ignore_index
        )
        _multilabel_confusion_matrix_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds, target, num_labels, threshold, ignore_index
    )
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _jaccard_index_reduce(
        confmat,
        average=average,
        ignore_index=ignore_index,
        zero_division=zero_division,
    )


def jaccard_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0.0,
) -> paddle.Tensor:
    """Calculate the Jaccard index.

    The `Jaccard index`_ (also known as the intersection over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_jaccard_index`,
    :func:`~paddle.metric.functional.classification.multiclass_jaccard_index` and
    :func:`~paddle.metric.functional.classification.multilabel_jaccard_index` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from paddle import randint, tensor
        >>> target = randint(0, 2, (10, 25, 25))
        >>> pred = tensor(target)
        >>> pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
        >>> jaccard_index(pred, target, task="multiclass", num_classes=2)
        tensor(0.9660)

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_jaccard_index(
            preds, target, threshold, ignore_index, validate_args, zero_division
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_jaccard_index(
            preds,
            target,
            num_classes,
            average,
            ignore_index,
            validate_args,
            zero_division,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_jaccard_index(
            preds,
            target,
            num_labels,
            threshold,
            average,
            ignore_index,
            validate_args,
            zero_division,
        )
    raise ValueError(f"Not handled value: {task}")
