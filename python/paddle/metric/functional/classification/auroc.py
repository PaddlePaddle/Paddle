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
from paddle import Tensor
from paddle.metric.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from paddle.metric.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from paddle.metric.utils.compute import _auc_compute_without_check, _safe_divide
from paddle.metric.utils.data import _bincount
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.prints import rank_zero_warn


def _reduce_auroc(
    fpr: paddle.Tensor | list[paddle.Tensor],
    tpr: paddle.Tensor | list[paddle.Tensor],
    average: Literal["macro", "weighted", "none"] | None = "macro",
    weights: paddle.Tensor | None = None,
    direction: float = 1.0,
) -> paddle.Tensor:
    """Reduce multiple average precision score into one number."""
    if isinstance(fpr, paddle.Tensor) and isinstance(tpr, paddle.Tensor):
        res = _auc_compute_without_check(fpr, tpr, direction=direction, axis=1)
    else:
        res = paddle.stack(
            [
                _auc_compute_without_check(x, y, direction=direction)
                for x, y in zip(fpr, tpr)
            ]
        )
    if average is None or average == "none":
        return res
    if paddle.isnan(res).any():
        rank_zero_warn(
            f"Average precision score for one or more classes was `nan`. Ignoring these classes in {average}-average",
            UserWarning,
        )
    idx = ~paddle.isnan(res)
    if average == "macro":
        return res[idx].mean()
    if average == "weighted" and weights is not None:
        weights = _safe_divide(weights[idx], weights[idx].sum())
        return (res[idx] * weights).sum()
    raise ValueError(
        "Received an incompatible combinations of inputs to make reduction."
    )


def _binary_auroc_arg_validation(
    max_fpr: float | None = None,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if (
        max_fpr is not None
        and not isinstance(max_fpr, float)
        and 0 < max_fpr <= 1
    ):
        raise ValueError(
            f"Arguments `max_fpr` should be a float in range (0, 1], but got: {max_fpr}"
        )


def _binary_auroc_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    thresholds: paddle.Tensor | None,
    max_fpr: float | None = None,
    pos_label: int = 1,
) -> paddle.Tensor:
    fpr, tpr, _ = _binary_roc_compute(state, thresholds, pos_label)
    if max_fpr is None or max_fpr == 1 or fpr.sum() == 0 or tpr.sum() == 0:
        return _auc_compute_without_check(fpr, tpr, 1.0)
    _device = fpr.device if isinstance(fpr, paddle.Tensor) else fpr[0].device
    max_area: Tensor = paddle.tensor(max_fpr, device=_device)
    stop = paddle.bucketize(max_area, fpr, out_int32=True, right=True)
    weight = (max_area - fpr[stop - 1]) / (fpr[stop] - fpr[stop - 1])
    interp_tpr: Tensor = paddle.lerp(
        x=tpr[stop - 1], y=tpr[stop], weight=weight
    )
    tpr = paddle.concat([tpr[:stop], interp_tpr.view(1)])
    fpr = paddle.concat([fpr[:stop], max_area.view(1)])
    partial_auc = _auc_compute_without_check(fpr, tpr, 1.0)
    min_area: Tensor = 0.5 * max_area**2
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))


def binary_auroc(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    max_fpr: float | None = None,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for binary tasks.

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        max_fpr: If not ``None``, calculates standardized partial AUC over the range ``[0, max_fpr]``.
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
        A single scalar with the auroc score

    Example:
        >>> from paddle.metric.functional.classification import binary_auroc
        >>> preds = paddle.to_tensor([0, 0.5, 0.7, 0.8])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> binary_auroc(preds, target, thresholds=None)
        tensor(0.5000)
        >>> binary_auroc(preds, target, thresholds=5)
        tensor(0.5000)

    """
    if validate_args:
        _binary_auroc_arg_validation(max_fpr, thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(
            preds, target, ignore_index
        )
    preds, target, thresholds = _binary_precision_recall_curve_format(
        preds, target, thresholds, ignore_index
    )
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_auroc_compute(state, thresholds, max_fpr)


def _multiclass_auroc_arg_validation(
    num_classes: int,
    average: Literal["macro", "weighted", "none"] | None = "macro",
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(
        num_classes, thresholds, ignore_index
    )
    allowed_average = "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average} but got {average}"
        )


def _multiclass_auroc_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_classes: int,
    average: Literal["macro", "weighted", "none"] | None = "macro",
    thresholds: paddle.Tensor | None = None,
) -> paddle.Tensor:
    fpr, tpr, _ = _multiclass_roc_compute(state, num_classes, thresholds)
    return _reduce_auroc(
        fpr,
        tpr,
        average,
        weights=_bincount(state[1], minlength=num_classes).float()
        if thresholds is None
        else state[0][:, 1, :].sum(-1),
    )


def multiclass_auroc(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    average: Literal["macro", "weighted", "none"] | None = "macro",
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for multiclass tasks.

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``weighted``: calculates score for each class and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each class and applies no reduction
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
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with auroc score per class.
        If `average="macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from paddle.metric.functional.classification import multiclass_auroc
        >>> preds = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> multiclass_auroc(preds, target, num_classes=5, average="macro", thresholds=None)
        tensor(0.5333)
        >>> multiclass_auroc(preds, target, num_classes=5, average=None, thresholds=None)
        tensor([1.0000, 1.0000, 0.3333, 0.3333, 0.0000])
        >>> multiclass_auroc(preds, target, num_classes=5, average="macro", thresholds=5)
        tensor(0.5333)
        >>> multiclass_auroc(preds, target, num_classes=5, average=None, thresholds=5)
        tensor([1.0000, 1.0000, 0.3333, 0.3333, 0.0000])

    """
    if validate_args:
        _multiclass_auroc_arg_validation(
            num_classes, average, thresholds, ignore_index
        )
        _multiclass_precision_recall_curve_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(
        preds, target, num_classes, thresholds
    )
    return _multiclass_auroc_compute(state, num_classes, average, thresholds)


def _multilabel_auroc_arg_validation(
    num_labels: int,
    average: Literal["micro", "macro", "weighted", "none"] | None,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(
        num_labels, thresholds, ignore_index
    )
    allowed_average = "micro", "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average} but got {average}"
        )


def _multilabel_auroc_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_labels: int,
    average: Literal["micro", "macro", "weighted", "none"] | None,
    thresholds: paddle.Tensor | None,
    ignore_index: int | None = None,
) -> paddle.Tensor:
    if average == "micro":
        if isinstance(state, paddle.Tensor) and thresholds is not None:
            return _binary_auroc_compute(state.sum(1), thresholds, max_fpr=None)
        preds = state[0].flatten()
        target = state[1].flatten()
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        return _binary_auroc_compute((preds, target), thresholds, max_fpr=None)
    fpr, tpr, _ = _multilabel_roc_compute(
        state, num_labels, thresholds, ignore_index
    )
    return _reduce_auroc(
        fpr,
        tpr,
        average,
        weights=(state[1] == 1).sum(dim=0).float()
        if thresholds is None
        else state[0][:, 1, :].sum(-1),
    )


def multilabel_auroc(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_) for multilabel tasks.

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction
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
        If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be returned with auroc score per class.
        If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Example:
        >>> from paddle.metric.functional.classification import multilabel_auroc
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = paddle.to_tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_auroc(preds, target, num_labels=3, average="macro", thresholds=None)
        tensor(0.6528)
        >>> multilabel_auroc(preds, target, num_labels=3, average=None, thresholds=None)
        tensor([0.6250, 0.5000, 0.8333])
        >>> multilabel_auroc(preds, target, num_labels=3, average="macro", thresholds=5)
        tensor(0.6528)
        >>> multilabel_auroc(preds, target, num_labels=3, average=None, thresholds=5)
        tensor([0.6250, 0.5000, 0.8333])

    """
    if validate_args:
        _multilabel_auroc_arg_validation(
            num_labels, average, thresholds, ignore_index
        )
        _multilabel_precision_recall_curve_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(
        preds, target, num_labels, thresholds
    )
    return _multilabel_auroc_compute(
        state, num_labels, average, thresholds, ignore_index
    )


def auroc(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: int | list[float] | paddle.Tensor | None = None,
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["macro", "weighted", "none"] | None = "macro",
    max_fpr: float | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor | None:
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_auroc`,
    :func:`~paddle.metric.functional.classification.multiclass_auroc` and
    :func:`~paddle.metric.functional.classification.multilabel_auroc` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> preds = paddle.to_tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = paddle.to_tensor([0, 0, 1, 1, 1])
        >>> auroc(preds, target, task='binary')
        tensor(0.5000)

        >>> preds = paddle.to_tensor(
        ...     [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.85, 0.05, 0.10], [0.10, 0.10, 0.80]]
        ... )
        >>> target = paddle.to_tensor([0, 1, 1, 2, 2])
        >>> auroc(preds, target, task='multiclass', num_classes=3)
        tensor(0.7778)

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_auroc(
            preds, target, max_fpr, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_auroc(
            preds,
            target,
            num_classes,
            average,
            thresholds,
            ignore_index,
            validate_args,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_auroc(
            preds,
            target,
            num_labels,
            average,
            thresholds,
            ignore_index,
            validate_args,
        )
    return None
