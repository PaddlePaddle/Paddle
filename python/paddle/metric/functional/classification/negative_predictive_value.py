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

from typing import TYPE_CHECKING

from typing_extensions import Literal

from paddle.metric.functional.classification.stat_scores import (
    _binary_stat_scores_arg_validation,
    _binary_stat_scores_format,
    _binary_stat_scores_tensor_validation,
    _binary_stat_scores_update,
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)
from paddle.metric.utils.compute import (
    _adjust_weights_safe_divide,
    _safe_divide,
)
from paddle.metric.utils.enums import ClassificationTask

if TYPE_CHECKING:
    import paddle


def _negative_predictive_value_reduce(
    tp: paddle.Tensor,
    fp: paddle.Tensor,
    tn: paddle.Tensor,
    fn: paddle.Tensor,
    average: Literal["binary", "micro", "macro", "weighted", "none"] | None,
    multidim_average: Literal["global", "samplewise"] = "global",
    multilabel: bool = False,
    top_k: int = 1,
    zero_division: float = 0,
) -> paddle.Tensor:
    """Reduction logic for negative predictive value."""
    if average == "binary":
        return _safe_divide(tn, tn + fn, zero_division)
    if average == "micro":
        tn = tn.sum(dim=0 if multidim_average == "global" else 1)
        fn = fn.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide(tn, tn + fn, zero_division)
    score = _safe_divide(tn, tn + fn, zero_division)
    return _adjust_weights_safe_divide(
        score, average, multilabel, tp, fp, fn, top_k=top_k
    )


def binary_negative_predictive_value(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> paddle.Tensor:
    """Compute `Negative Predictive Value`_ for binary tasks.

    .. math:: \\text{Negative Predictive Value} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and false positives
    respectively. The metric is only proper defined when :math:`\\text{TN} + \\text{FP} \\neq 0`. If this case is
    encountered a score of 0 is returned.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Should be `0` or `1`. The value returned when :math:`\\text{TP} + \\text{FP} = 0`.

    Returns:
        If ``multidim_average`` is set to ``global``, the metric returns a scalar value. If ``multidim_average``
        is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar value per sample.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import binary_negative_predictive_value
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> binary_negative_predictive_value(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import binary_negative_predictive_value
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_negative_predictive_value(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import binary_negative_predictive_value
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> binary_negative_predictive_value(preds, target, multidim_average='samplewise')
        tensor([0.0000, 0.2500])

    """
    if validate_args:
        _binary_stat_scores_arg_validation(
            threshold, multidim_average, ignore_index
        )
        _binary_stat_scores_tensor_validation(
            preds, target, multidim_average, ignore_index
        )
    preds, target = _binary_stat_scores_format(
        preds, target, threshold, ignore_index
    )
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _negative_predictive_value_reduce(
        tp,
        fp,
        tn,
        fn,
        average="binary",
        multidim_average=multidim_average,
        zero_division=zero_division,
    )


def multiclass_negative_predictive_value(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> paddle.Tensor:
    """Compute `Negative Predictive Value`_ for multiclass tasks.

    .. math:: \\text{Negative Predictive Value} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and false positives
    respectively. The metric is only proper defined when :math:`\\text{TN} + \\text{FP} \\neq 0`. If this case is
    encountered a score of 0 is returned.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: Calculate statistics for each label and apply no reduction

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: Bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Should be `0` or `1`. The value returned when :math:`\\text{TP} + \\text{FP} = 0`.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multiclass_negative_predictive_value
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3)
        tensor(0.8889)
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3, average=None)
        tensor([0.6667, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multiclass_negative_predictive_value
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3)
        tensor(0.8889)
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3, average=None)
        tensor([0.6667, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import multiclass_negative_predictive_value
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3, multidim_average='samplewise')
        tensor([0.7833, 0.6556])
        >>> multiclass_negative_predictive_value(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[1.0000, 0.6000, 0.7500],
                [0.8000, 0.5000, 0.6667]])

    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(
            num_classes, top_k, average, multidim_average, ignore_index
        )
        _multiclass_stat_scores_tensor_validation(
            preds, target, num_classes, multidim_average, ignore_index
        )
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds,
        target,
        num_classes,
        top_k,
        average,
        multidim_average,
        ignore_index,
    )
    return _negative_predictive_value_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        multidim_average=multidim_average,
        top_k=top_k,
        zero_division=zero_division,
    )


def multilabel_negative_predictive_value(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> paddle.Tensor:
    """Compute `Negative Predictive Value`_ for multilabel tasks.

    .. math:: \\text{Negative Predictive Value} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and false positives
    respectively. The metric is only proper defined when :math:`\\text{TN} + \\text{FP} \\neq 0`. If this case is
    encountered a score of 0 is returned.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculate statistics for each label and compute a weighted average using their support
            - ``"none"`` or ``None``: Calculate statistics for each label and apply no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: Bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        zero_division: Should be `0` or `1`. The value returned when :math:`\\text{TP} + \\text{FP} = 0`.

    Returns:
        The returned shape depends on the ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multilabel_negative_predictive_value
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3)
        tensor(0.5000)
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multilabel_negative_predictive_value
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3)
        tensor(0.5000)
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3, average=None)
        tensor([1.0000, 0.5000, 0.0000])

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import multilabel_negative_predictive_value
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3, multidim_average='samplewise')
        tensor([0.0000, 0.1667])
        >>> multilabel_negative_predictive_value(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000]])

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(
            num_labels, threshold, average, multidim_average, ignore_index
        )
        _multilabel_stat_scores_tensor_validation(
            preds, target, num_labels, multidim_average, ignore_index
        )
    preds, target = _multilabel_stat_scores_format(
        preds, target, num_labels, threshold, ignore_index
    )
    tp, fp, tn, fn = _multilabel_stat_scores_update(
        preds, target, multidim_average
    )
    return _negative_predictive_value_reduce(
        tp,
        fp,
        tn,
        fn,
        average=average,
        multidim_average=multidim_average,
        multilabel=True,
        zero_division=zero_division,
    )


def negative_predictive_value(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
    multidim_average: Literal["global", "samplewise"] | None = "global",
    top_k: int | None = 1,
    ignore_index: int | None = None,
    validate_args: bool = True,
    zero_division: float = 0,
) -> paddle.Tensor:
    """Compute `Negative Predictive Value`_.

    .. math:: \\text{Negative Predictive Value} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and false positives
    respectively. The metric is only proper defined when :math:`\\text{TN} + \\text{FP} \\neq 0`. If this case is
    encountered a score of 0 is returned.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_negative_predictive_value`,
    :func:`~paddle.metric.functional.classification.multiclass_negative_predictive_value` and
    :func:`~paddle.metric.functional.classification.multilabel_negative_predictive_value` for the specific
    details of each argument influence and examples.

    LegacyExample:
        >>> from paddle import tensor
        >>> preds = tensor([2, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> negative_predictive_value(preds, target, task="multiclass", average='macro', num_classes=3)
        tensor(0.6667)
        >>> negative_predictive_value(preds, target, task="multiclass", average='micro', num_classes=3)
        tensor(0.6250)

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None
    if task == ClassificationTask.BINARY:
        return binary_negative_predictive_value(
            preds,
            target,
            threshold,
            multidim_average,
            ignore_index,
            validate_args,
            zero_division,
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        if not isinstance(top_k, int):
            raise ValueError(
                f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`"
            )
        return multiclass_negative_predictive_value(
            preds,
            target,
            num_classes,
            average,
            top_k,
            multidim_average,
            ignore_index,
            validate_args,
            zero_division,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_negative_predictive_value(
            preds,
            target,
            num_labels,
            threshold,
            average,
            multidim_average,
            ignore_index,
            validate_args,
            zero_division,
        )
    raise ValueError(f"Not handled value: {task}")
