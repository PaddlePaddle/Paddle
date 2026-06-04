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
from paddle.metric.utils.enums import ClassificationTask


def _matthews_corrcoef_reduce(confmat: paddle.Tensor) -> paddle.Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the matthews corrcoef score.

    See: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7 for more info.

    """
    confmat = confmat.sum(0) if confmat.ndim == 3 else confmat
    if confmat.size == 4:
        tn, fp, fn, tp = confmat.reshape(-1)
        if tp + tn != 0 and fp + fn == 0:
            return paddle.tensor(1.0, dtype=confmat.dtype, device=confmat.place)
        if tp + tn == 0 and fp + fn != 0:
            return paddle.tensor(
                -1.0, dtype=confmat.dtype, device=confmat.place
            )
    confmat = confmat.float()
    tk = confmat.sum(dim=-1)
    pk = confmat.sum(dim=-2)
    c = paddle.trace(x=confmat)
    s = confmat.sum()
    cov_ytyp = c * s - sum(tk * pk)
    cov_ypyp = s**2 - sum(pk * pk)
    cov_ytyt = s**2 - sum(tk * tk)
    numerator = cov_ytyp
    denom = cov_ypyp * cov_ytyt
    if denom == 0 and confmat.size == 4:
        eps = paddle.tensor(
            paddle.finfo(paddle.float32).eps,
            dtype=paddle.float32,
            device=confmat.device,
        )
        if fn == 0 and tn == 0:
            numerator = paddle.sqrt(eps) * (tp - fp)
        elif fp == 0 and tn == 0:
            numerator = paddle.sqrt(eps) * (tp - fn)
        elif tp == 0 and fn == 0:
            numerator = paddle.sqrt(eps) * (tn - fp)
        elif tp == 0 and fp == 0:
            numerator = paddle.sqrt(eps) * (tn - fn)
        elif tp == 0:
            numerator = tn - fp * fn
        elif tn == 0:
            numerator = tp - fp * fn
        elif fp == 0 or fn == 0:
            numerator = tp * tn
        else:
            return paddle.tensor(0, dtype=confmat.dtype, device=confmat.place)
        denom = (
            (tp + fp + eps)
            * (tp + fn + eps)
            * (tn + fp + eps)
            * (tn + fn + eps)
        )
    elif denom == 0:
        return paddle.tensor(0, dtype=confmat.dtype, device=confmat.place)
    return numerator / paddle.sqrt(denom)


def binary_matthews_corrcoef(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Calculate `Matthews correlation coefficient`_ for binary tasks.

    This metric measures the general correlation or quality of a classification.

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
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import binary_matthews_corrcoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import binary_matthews_corrcoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> binary_matthews_corrcoef(preds, target)
        tensor(0.5774)

    """
    if validate_args:
        _binary_confusion_matrix_arg_validation(
            threshold, ignore_index, normalize=None
        )
        _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold, ignore_index
    )
    confmat = _binary_confusion_matrix_update(preds, target)
    return _matthews_corrcoef_reduce(confmat)


def multiclass_matthews_corrcoef(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Calculate `Matthews correlation coefficient`_ for multiclass tasks.

    This metric measures the general correlation or quality of a classification.

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
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multiclass_matthews_corrcoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)

    Example (pred is float tensor):
        >>> from paddle.metric.functional.classification import multiclass_matthews_corrcoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> multiclass_matthews_corrcoef(preds, target, num_classes=3)
        tensor(0.7000)

    """
    if validate_args:
        _multiclass_confusion_matrix_arg_validation(
            num_classes, ignore_index, normalize=None
        )
        _multiclass_confusion_matrix_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target = _multiclass_confusion_matrix_format(
        preds, target, ignore_index
    )
    confmat = _multiclass_confusion_matrix_update(preds, target, num_classes)
    return _matthews_corrcoef_reduce(confmat)


def multilabel_matthews_corrcoef(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Calculate `Matthews correlation coefficient`_ for multilabel tasks.

    This metric measures the general correlation or quality of a classification.

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
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multilabel_matthews_corrcoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_matthews_corrcoef(preds, target, num_labels=3)
        tensor(0.3333)

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(
            num_labels, threshold, ignore_index, normalize=None
        )
        _multilabel_confusion_matrix_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds, target, num_labels, threshold, ignore_index
    )
    confmat = _multilabel_confusion_matrix_update(preds, target, num_labels)
    return _matthews_corrcoef_reduce(confmat)


def matthews_corrcoef(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: int | None = None,
    num_labels: int | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Calculate `Matthews correlation coefficient`_ .

    This metric measures the general correlation or quality of a classification.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_matthews_corrcoef`,
    :func:`~paddle.metric.functional.classification.multiclass_matthews_corrcoef` and
    :func:`~paddle.metric.functional.classification.multilabel_matthews_corrcoef` for
    the specific details of each argument influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> matthews_corrcoef(preds, target, task="multiclass", num_classes=2)
        tensor(0.5774)

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_matthews_corrcoef(
            preds, target, threshold, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_matthews_corrcoef(
            preds, target, num_classes, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_matthews_corrcoef(
            preds, target, num_labels, threshold, ignore_index, validate_args
        )
    raise ValueError(f"Not handled value: {task}")
