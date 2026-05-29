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
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
)
from paddle.metric.utils.compute import normalize_logits_if_needed
from paddle.metric.utils.data import to_onehot
from paddle.metric.utils.enums import ClassificationTaskNoMultilabel


def _hinge_loss_compute(
    measure: paddle.Tensor, total: paddle.Tensor
) -> paddle.Tensor:
    return measure / total


def _binary_hinge_loss_arg_validation(
    squared: bool, ignore_index: int | None = None
) -> None:
    if not isinstance(squared, bool):
        raise ValueError(
            f"Expected argument `squared` to be an bool but got {squared}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )


def _binary_hinge_loss_tensor_validation(
    preds: paddle.Tensor, target: paddle.Tensor, ignore_index: int | None = None
) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected argument `preds` to be floating tensor with probabilities/logits but got tensor with dtype {preds.dtype}"
        )


def _binary_hinge_loss_update(
    preds: paddle.Tensor, target: paddle.Tensor, squared: bool
) -> tuple[paddle.Tensor, paddle.Tensor]:
    target = target.bool()
    margin = paddle.zeros_like(preds)
    margin[target] = preds[target]
    margin[~target] = -preds[~target]
    measures = 1 - margin
    measures = paddle.clamp(measures, 0)
    if squared:
        measures = measures.pow(2)
    total = paddle.tensor(target.shape[0], device=target.place)
    return measures.sum(dim=0), total


def binary_hinge_loss(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    squared: bool = False,
    ignore_index: int | None = None,
    validate_args: bool = False,
) -> paddle.Tensor:
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks.

    .. math::
        \\text{Hinge loss} = \\max(0, 1 - y \\times \\hat{y})

    Where :math:`y \\in {-1, 1}` is the target, and :math:`\\hat{y} \\in \\mathbb{R}` is the prediction.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import binary_hinge_loss
        >>> preds = tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> binary_hinge_loss(preds, target)
        tensor(0.6900)
        >>> binary_hinge_loss(preds, target, squared=True)
        tensor(0.6905)

    """
    if validate_args:
        _binary_hinge_loss_arg_validation(squared, ignore_index)
        _binary_hinge_loss_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds,
        target,
        threshold=0.0,
        ignore_index=ignore_index,
        convert_to_labels=False,
    )
    measures, total = _binary_hinge_loss_update(preds, target, squared)
    return _hinge_loss_compute(measures, total)


def _multiclass_hinge_loss_arg_validation(
    num_classes: int,
    squared: bool = False,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
    ignore_index: int | None = None,
) -> None:
    _binary_hinge_loss_arg_validation(squared, ignore_index)
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
        )
    allowed_mm = "crammer-singer", "one-vs-all"
    if multiclass_mode not in allowed_mm:
        raise ValueError(
            f"Expected argument `multiclass_mode` to be one of {allowed_mm}, but got {multiclass_mode}."
        )


def _multiclass_hinge_loss_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> None:
    _multiclass_confusion_matrix_tensor_validation(
        preds, target, num_classes, ignore_index
    )
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected argument `preds` to be floating tensor with probabilities/logits but got tensor with dtype {preds.dtype}"
        )


def _multiclass_hinge_loss_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    squared: bool,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    preds = normalize_logits_if_needed(preds, "softmax")
    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= paddle.max(preds[~target].view(preds.shape[0], -1), axis=1)[0]
    else:
        target = target.bool()
        margin = paddle.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]
    measures = 1 - margin
    measures = paddle.clamp(measures, 0)
    if squared:
        measures = measures.pow(2)
    total = paddle.tensor(target.shape[0], device=target.place)
    return measures.sum(dim=0), total


def multiclass_hinge_loss(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    squared: bool = False,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
    ignore_index: int | None = None,
    validate_args: bool = False,
) -> paddle.Tensor:
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \\text{Hinge loss} = \\max\\left(0, 1 - \\hat{y}_y + \\max_{i \\ne y} (\\hat{y}_i)\\right)

    Where :math:`y \\in {0, ..., \\mathrm{C}}` is the target class (where :math:`\\mathrm{C}` is the number of classes),
    and :math:`\\hat{y} \\in \\mathbb{R}^\\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multiclass_hinge_loss
        >>> preds = tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
        >>> target = tensor([0, 1, 2, 0])
        >>> multiclass_hinge_loss(preds, target, num_classes=3)
        tensor(0.9125)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, squared=True)
        tensor(1.1131)
        >>> multiclass_hinge_loss(preds, target, num_classes=3, multiclass_mode='one-vs-all')
        tensor([0.8750, 1.1250, 1.1000])

    """
    if validate_args:
        _multiclass_hinge_loss_arg_validation(
            num_classes, squared, multiclass_mode, ignore_index
        )
        _multiclass_hinge_loss_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target = _multiclass_confusion_matrix_format(
        preds, target, ignore_index, convert_to_labels=False
    )
    measures, total = _multiclass_hinge_loss_update(
        preds, target, squared, multiclass_mode
    )
    return _hinge_loss_compute(measures, total)


def hinge_loss(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: int | None = None,
    squared: bool = False,
    multiclass_mode: Literal["crammer-singer", "one-vs-all"] = "crammer-singer",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_hinge_loss` and
    :func:`~paddle.metric.functional.classification.multiclass_hinge_loss` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([0, 1, 1])
        >>> preds = tensor([0.5, 0.7, 0.1])
        >>> hinge_loss(preds, target, task="binary")
        tensor(0.9000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target, task="multiclass", num_classes=3)
        tensor(1.5551)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge_loss(preds, target, task="multiclass", num_classes=3, multiclass_mode="one-vs-all")
        tensor([1.3743, 1.1945, 1.2359])

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_hinge_loss(
            preds, target, squared, ignore_index, validate_args
        )
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_hinge_loss(
            preds,
            target,
            num_classes,
            squared,
            multiclass_mode,
            ignore_index,
            validate_args,
        )
    raise ValueError(f"Not handled value: {task}")
