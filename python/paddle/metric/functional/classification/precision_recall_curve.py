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

import paddle
from paddle import Tensor
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.compute import (
    _safe_divide,
    interp,
    normalize_logits_if_needed,
)
from paddle.metric.utils.data import _bincount, _cumsum
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.prints import rank_zero_warn

if TYPE_CHECKING:
    from collections.abc import Sequence


def _binary_clf_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    sample_weights: Sequence | paddle.Tensor | None = None,
    pos_label: int = 1,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Calculate the TPs and false positives for all unique thresholds in the preds tensor.

    Adapted from
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py.

    Args:
        preds: 1d tensor with predictions
        target: 1d tensor with true values
        sample_weights: a 1d tensor with a weight per sample
        pos_label: integer determining what the positive class in target tensor is

    Returns:
        fps: 1d tensor with false positives for different thresholds
        tps: 1d tensor with true positives for different thresholds
        thresholds: the unique thresholds use for calculating fps and tps

    """
    with paddle.no_grad():
        if sample_weights is not None and not isinstance(
            sample_weights, paddle.Tensor
        ):
            sample_weights = paddle.tensor(
                sample_weights, device=preds.device, dtype=paddle.float32
            )
        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = paddle.argsort(preds, descending=True)
        preds = preds[desc_score_indices]
        target = target[desc_score_indices]
        weight = (
            sample_weights[desc_score_indices]
            if sample_weights is not None
            else 1.0
        )
        distinct_value_indices = paddle.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = paddle.nn.functional.pad(
            distinct_value_indices, [0, 1], value=target.size(0) - 1
        )
        target = (target == pos_label).to(paddle.long)
        tps = _cumsum(target * weight, axis=0)[threshold_idxs]
        if sample_weights is not None:
            fps = _cumsum((1 - target) * weight, axis=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps.to(paddle.long)
        return fps, tps, preds[threshold_idxs]


def _adjust_threshold_arg(
    thresholds: int | list[float] | paddle.Tensor | None = None,
    device: paddle.device | None = None,
) -> paddle.Tensor | None:
    """Convert threshold arg for list and int to tensor format."""
    if isinstance(thresholds, int):
        return paddle.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        return paddle.tensor(thresholds, device=device)
    return thresholds


def _binary_precision_recall_curve_arg_validation(
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int

    """
    if thresholds is not None and not isinstance(
        thresholds, (list, int, Tensor)
    ):
        raise ValueError(
            f"Expected argument `thresholds` to either be an integer, list of floats or tensor of floats, but got {thresholds}"
        )
    if isinstance(thresholds, int) and thresholds < 2:
        raise ValueError(
            f"If argument `thresholds` is an integer, expected it to be larger than 1, but got {thresholds}"
        )
    if isinstance(thresholds, list) and not all(
        isinstance(t, float) and 0 <= t <= 1 for t in thresholds
    ):
        raise ValueError(
            f"If argument `thresholds` is a list, expected all elements to be floats in the [0,1] range, but got {thresholds}"
        )
    if isinstance(thresholds, paddle.Tensor) and not thresholds.ndim == 1:
        raise ValueError(
            "If argument `thresholds` is an tensor, expected the tensor to be 1d"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )


def _binary_precision_recall_curve_tensor_validation(
    preds: paddle.Tensor, target: paddle.Tensor, ignore_index: int | None = None
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point

    """
    _check_same_shape(preds, target)
    if target.is_floating_point():
        raise ValueError(
            f"Expected argument `target` to be an int or long tensor with ground truth labels but got tensor with dtype {target.dtype}"
        )
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected argument `preds` to be an floating tensor with probability/logit scores, but got tensor with dtype {preds.dtype}"
        )
    unique_values = paddle.unique(target, axis=None)
    if ignore_index is None:
        check = paddle.any((unique_values != 0) & (unique_values != 1))
    else:
        check = paddle.any(
            (unique_values != 0)
            & (unique_values != 1)
            & (unique_values != ignore_index)
        )
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )


def _binary_precision_recall_curve_format(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    normalization: Literal["sigmoid", "softmax"] | None = "sigmoid",
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor | None]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]
    preds = normalize_logits_if_needed(preds, normalization)
    thresholds = _adjust_threshold_arg(thresholds, preds.place)
    return preds, target, thresholds


def _binary_precision_recall_curve_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    thresholds: paddle.Tensor | None,
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.

    """
    if thresholds is None:
        return preds, target
    if preds.size <= 50000:
        update_fn = _binary_precision_recall_curve_update_vectorized
    else:
        update_fn = _binary_precision_recall_curve_update_loop
    return update_fn(preds, target, thresholds)


def _binary_precision_recall_curve_update_vectorized(
    preds: paddle.Tensor, target: paddle.Tensor, thresholds: paddle.Tensor
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation is vectorized and faster than `_binary_precision_recall_curve_update_loop` for small
    numbers of samples (up to 50k) but less memory- and time-efficient for more samples.

    """
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0)).long()
    unique_mapping = (
        preds_t
        + 2 * target.long().unsqueeze(-1)
        + 4 * paddle.arange(len_t, device=target.place)
    )
    bins = _bincount(unique_mapping.flatten(), minlength=4 * len_t)
    return bins.reshape(len_t, 2, 2)


def _binary_precision_recall_curve_update_loop(
    preds: paddle.Tensor, target: paddle.Tensor, thresholds: paddle.Tensor
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples (up to 50k).

    """
    len_t = len(thresholds)
    target = target == 1
    confmat = thresholds.new_empty((len_t, 2, 2), dtype=paddle.int64)
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, 1, 1] = (target & preds_t).sum()
        confmat[i, 0, 1] = (~target & preds_t).sum()
        confmat[i, 1, 0] = (target & ~preds_t).sum()
    confmat[:, 0, 0] = (
        len(preds_t) - confmat[:, 0, 1] - confmat[:, 1, 0] - confmat[:, 1, 1]
    )
    return confmat


def _binary_precision_recall_curve_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    thresholds: paddle.Tensor | None,
    pos_label: int = 1,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve.

    """
    if isinstance(state, paddle.Tensor) and thresholds is not None:
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        precision = _safe_divide(tps, tps + fps, zero_division="nan")
        recall = _safe_divide(tps, tps + fns, zero_division="nan")
        precision = paddle.concat(
            [
                precision,
                paddle.ones(1, dtype=precision.dtype, device=precision.place),
            ]
        )
        recall = paddle.concat(
            [recall, paddle.zeros(1, dtype=recall.dtype, device=recall.place)]
        )
        return precision, recall, thresholds
    fps, tps, thresholds = _binary_clf_curve(
        state[0], state[1], pos_label=pos_label
    )
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    if (state[1] == 0).all():
        rank_zero_warn(
            "No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.",
            UserWarning,
        )
        recall = paddle.ones_like(recall)
    precision = paddle.concat(
        [
            precision.flip(axis=0),
            paddle.ones(1, dtype=precision.dtype, device=precision.place),
        ]
    )
    recall = paddle.concat(
        [
            recall.flip(axis=0),
            paddle.zeros(1, dtype=recall.dtype, device=recall.place),
        ]
    )
    thresholds = thresholds.flip(axis=0).detach().clone()
    return precision, recall, thresholds


def binary_precision_recall_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute the precision-recall curve for binary tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

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
        (tuple): a tuple of 3 tensors containing:

        - precision: an 1d tensor of size (n_thresholds+1, ) with precision values
        - recall: an 1d tensor of size (n_thresholds+1, ) with recall values
        - thresholds: an 1d tensor of size (n_thresholds, ) with increasing threshold values

    Example:
        >>> from paddle.metric.functional.classification import binary_precision_recall_curve
        >>> preds = paddle.to_tensor([0, 0.5, 0.7, 0.8])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> binary_precision_recall_curve(preds, target, thresholds=None)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([0.0000, 0.5000, 0.7000, 0.8000]))
        >>> binary_precision_recall_curve(preds, target, thresholds=5)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.6667, 0.0000, nan, 1.0000]),
         tensor([1., 1., 1., 0., 0., 0.]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """
    if validate_args:
        _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(
            preds, target, ignore_index
        )
    preds, target, thresholds = _binary_precision_recall_curve_format(
        preds, target, thresholds, ignore_index
    )
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_precision_recall_curve_compute(state, thresholds)


def _multiclass_precision_recall_curve_arg_validation(
    num_classes: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    average: Literal["micro", "macro"] | None = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be an int larger than 1
    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int

    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
        )
    if average not in (None, "micro", "macro"):
        raise ValueError(
            f"Expected argument `average` to be one of None, 'micro' or 'macro', but got {average}"
        )
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)


def _multiclass_precision_recall_curve_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> None:
    """Validate tensor input.

    - target should have one more dimension than preds and all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - all values in target tensor that are not ignored have to be in {0, 1}

    """
    if not preds.ndim == target.ndim + 1:
        raise ValueError(
            f"Expected `preds` to have one more dimension than `target` but got {preds.ndim} and {target.ndim}"
        )
    if target.is_floating_point():
        raise ValueError(
            f"Expected argument `target` to be an int or long tensor, but got tensor with dtype {target.dtype}"
        )
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected `preds` to be a float tensor, but got {preds.dtype}"
        )
    if preds.shape[1] != num_classes:
        raise ValueError(
            f"Expected `preds.shape[1]` to be equal to the number of classes but got {preds.shape[1]} and {num_classes}."
        )
    if preds.shape[0] != target.shape[0] or preds.shape[2:] != target.shape[1:]:
        raise ValueError(
            f"Expected the shape of `preds` should be (N, C, ...) and the shape of `target` should be (N, ...) but got {preds.shape} and {target.shape}"
        )
    num_unique_values = len(paddle.unique(target, axis=None))
    check = (
        num_unique_values > num_classes
        if ignore_index is None
        else num_unique_values > num_classes + 1
    )
    if check:
        raise RuntimeError(
            f"Detected more unique values in `target` than `num_classes`. Expected only {num_classes if ignore_index is None else num_classes + 1} but found {num_unique_values} in `target`."
        )


def _multiclass_precision_recall_curve_format(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    average: Literal["micro", "macro"] | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor | None]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies softmax if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.transpose(0, 1).reshape(num_classes, -1).T
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]
    preds = normalize_logits_if_needed(preds, "softmax")
    if average == "micro":
        preds = preds.flatten()
        target = paddle.nn.functional.one_hot(
            target, num_classes=num_classes
        ).flatten()
    thresholds = _adjust_threshold_arg(thresholds, preds.place)
    return preds, target, thresholds


def _multiclass_precision_recall_curve_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: paddle.Tensor | None,
    average: Literal["micro", "macro"] | None = None,
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.

    """
    if thresholds is None:
        return preds, target
    if average == "micro":
        return _binary_precision_recall_curve_update(preds, target, thresholds)
    if preds.size * num_classes <= 1000000:
        update_fn = _multiclass_precision_recall_curve_update_vectorized
    else:
        update_fn = _multiclass_precision_recall_curve_update_loop
    return update_fn(preds, target, num_classes, thresholds)


def _multiclass_precision_recall_curve_update_vectorized(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: paddle.Tensor,
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation is vectorized and faster than `_binary_precision_recall_curve_update_loop` for small
    numbers of samples but less memory- and time-efficient for more samples.

    """
    len_t = len(thresholds)
    preds_t = (
        preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)
    ).long()
    target_t = paddle.nn.functional.one_hot(target, num_classes=num_classes)
    unique_mapping = preds_t + 2 * target_t.long().unsqueeze(-1)
    unique_mapping += 4 * paddle.arange(
        num_classes, device=preds.place
    ).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_classes * paddle.arange(len_t, device=preds.place)
    bins = _bincount(
        unique_mapping.flatten(), minlength=4 * num_classes * len_t
    )
    return bins.reshape(len_t, num_classes, 2, 2)


def _multiclass_precision_recall_curve_update_loop(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: paddle.Tensor,
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the state to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples.

    """
    len_t = len(thresholds)
    target_t = paddle.nn.functional.one_hot(target, num_classes=num_classes)
    confmat = thresholds.new_empty(
        (len_t, num_classes, 2, 2), dtype=paddle.int64
    )
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, :, 1, 1] = (target_t & preds_t).sum(dim=0)
        confmat[i, :, 0, 1] = (~target_t & preds_t).sum(dim=0)
        confmat[i, :, 1, 0] = (target_t & ~preds_t).sum(dim=0)
    confmat[:, :, 0, 0] = (
        len(preds_t)
        - confmat[:, :, 0, 1]
        - confmat[:, :, 1, 0]
        - confmat[:, :, 1, 1]
    )
    return confmat


def _multiclass_precision_recall_curve_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_classes: int,
    thresholds: paddle.Tensor | None,
    average: Literal["micro", "macro"] | None = None,
) -> (
    tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
):
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.

    """
    if average == "micro":
        return _binary_precision_recall_curve_compute(state, thresholds)
    if isinstance(state, paddle.Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps, zero_division="nan")
        recall = _safe_divide(tps, tps + fns, zero_division="nan")
        precision = paddle.concat(
            [
                precision,
                paddle.ones(
                    1,
                    num_classes,
                    dtype=precision.dtype,
                    device=precision.device,
                ),
            ]
        )
        recall = paddle.concat(
            [
                recall,
                paddle.zeros(
                    1, num_classes, dtype=recall.dtype, device=recall.place
                ),
            ]
        )
        precision = precision.T
        recall = recall.T
        thres = thresholds
        tensor_state = True
    else:
        precision_list, recall_list, thres_list = [], [], []
        for i in range(num_classes):
            res = _binary_precision_recall_curve_compute(
                (state[0][:, i], state[1]), thresholds=None, pos_label=i
            )
            precision_list.append(res[0])
            recall_list.append(res[1])
            thres_list.append(res[2])
        tensor_state = False
    if average == "macro":
        thres = (
            thres.repeat(num_classes)
            if tensor_state
            else paddle.concat(thres_list, 0)
        )
        thres = paddle.sort(x=thres)
        mean_precision = (
            precision.flatten()
            if tensor_state
            else paddle.concat(precision_list, 0)
        )
        mean_precision = paddle.sort(x=mean_precision)
        mean_recall = paddle.zeros_like(mean_precision)
        for i in range(num_classes):
            mean_recall += interp(
                mean_precision,
                precision[i] if tensor_state else precision_list[i],
                recall[i] if tensor_state else recall_list[i],
            )
        mean_recall /= num_classes
        return mean_precision, mean_recall, thres
    if tensor_state:
        return precision, recall, thres
    return precision_list, recall_list, thres_list


def multiclass_precision_recall_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    average: Literal["micro", "macro"] | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> (
    tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
):
    """Compute the precision-recall curve for multiclass tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

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
            If aggregation of curves should be applied. By default, the curves are not aggregated and a curve for
            each class is returned. If `average` is set to ``"micro"``, the metric will aggregate the curves by one hot
            encoding the targets and flattening the predictions, considering all classes jointly as a binary problem.
            If `average` is set to ``"macro"``, the metric will aggregate the curves by first interpolating the curves
            from each class at a combined set of thresholds and then average over the classwise interpolated curves.
            See `averaging curve objects`_ for more info on the different averaging methods.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from paddle.metric.functional.classification import multiclass_precision_recall_curve
        >>> preds = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> precision, recall, thresholds = multiclass_precision_recall_curve(preds, target, num_classes=5, thresholds=None)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 1., 0.]), tensor([1., 1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]),
         tensor([0.0500])]
        >>> multiclass_precision_recall_curve(preds, target, num_classes=5, thresholds=5)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.2500, 1.0000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.2500, 1.0000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000,    nan, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000,    nan, 1.0000],
                 [0.0000,    nan,    nan,    nan,    nan, 1.0000]]),
         tensor([[1., 1., 1., 1., 0., 0.],
                 [1., 1., 1., 1., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [nan, nan, nan, nan, nan, 0.]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """
    if validate_args:
        _multiclass_precision_recall_curve_arg_validation(
            num_classes, thresholds, ignore_index, average
        )
        _multiclass_precision_recall_curve_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index, average
    )
    state = _multiclass_precision_recall_curve_update(
        preds, target, num_classes, thresholds, average
    )
    return _multiclass_precision_recall_curve_compute(
        state, num_classes, thresholds, average
    )


def _multilabel_precision_recall_curve_arg_validation(
    num_labels: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` has to be an int larger than 1
    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int

    """
    _multiclass_precision_recall_curve_arg_validation(
        num_labels, thresholds, ignore_index
    )


def _multilabel_precision_recall_curve_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    ignore_index: int | None = None,
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - preds.shape[1] is equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point

    """
    _binary_precision_recall_curve_tensor_validation(
        preds, target, ignore_index
    )
    if preds.shape[1] != num_labels:
        raise ValueError(
            f"Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels but got {preds.shape[1]} and expected {num_labels}"
        )


def _multilabel_precision_recall_curve_format(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor | None]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Mask all datapoints that should be ignored with negative values
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor

    """
    preds = preds.transpose(0, 1).reshape(num_labels, -1).T
    target = target.transpose(0, 1).reshape(num_labels, -1).T
    preds = normalize_logits_if_needed(preds, "sigmoid")
    thresholds = _adjust_threshold_arg(thresholds, preds.place)
    if ignore_index is not None and thresholds is not None:
        preds = preds.clone()
        target = target.clone()
        idx = target == ignore_index
        preds[idx] = (
            -4 * num_labels * (len(thresholds) if thresholds is not None else 1)
        )
        target[idx] = (
            -4 * num_labels * (len(thresholds) if thresholds is not None else 1)
        )
    return preds, target, thresholds


def _multilabel_precision_recall_curve_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    thresholds: paddle.Tensor | None,
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Return the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.

    """
    if thresholds is None:
        return preds, target
    len_t = len(thresholds)
    preds_t = (
        preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)
    ).long()
    unique_mapping = preds_t + 2 * target.long().unsqueeze(-1)
    unique_mapping += 4 * paddle.arange(
        num_labels, device=preds.place
    ).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_labels * paddle.arange(len_t, device=preds.place)
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels * len_t)
    return bins.reshape(len_t, num_labels, 2, 2)


def _multilabel_precision_recall_curve_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_labels: int,
    thresholds: paddle.Tensor | None,
    ignore_index: int | None = None,
) -> (
    tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
):
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.

    """
    if isinstance(state, paddle.Tensor) and thresholds is not None:
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps, zero_division="nan")
        recall = _safe_divide(tps, tps + fns, zero_division="nan")
        precision = paddle.concat(
            [
                precision,
                paddle.ones(
                    1,
                    num_labels,
                    dtype=precision.dtype,
                    device=precision.device,
                ),
            ]
        )
        recall = paddle.concat(
            [
                recall,
                paddle.zeros(
                    1, num_labels, dtype=recall.dtype, device=recall.place
                ),
            ]
        )
        return precision.T, recall.T, thresholds
    precision_list, recall_list, thres_list = [], [], []
    for i in range(num_labels):
        preds = state[0][:, i]
        target = state[1][:, i]
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        res = _binary_precision_recall_curve_compute(
            (preds, target), thresholds=None, pos_label=1
        )
        precision_list.append(res[0])
        recall_list.append(res[1])
        thres_list.append(res[2])
    return precision_list, recall_list, thres_list


def multilabel_precision_recall_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> (
    tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
):
    """Compute the precision-recall curve for multilabel tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

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
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between labels). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all labels.

    Example:
        >>> from paddle.metric.functional.classification import multilabel_precision_recall_curve
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = paddle.to_tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> precision, recall, thresholds = multilabel_precision_recall_curve(preds, target, num_labels=3, thresholds=None)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.5000, 0.5000, 1.0000, 1.0000]), tensor([0.5000, 0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([0.7500, 1.0000, 1.0000, 1.0000])]
        >>> recall  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([1.0000, 0.6667, 0.3333, 0.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0500, 0.4500, 0.7500]), tensor([0.0500, 0.5500, 0.6500, 0.7500]), tensor([0.0500, 0.3500, 0.7500])]
        >>> multilabel_precision_recall_curve(preds, target, num_labels=3, thresholds=5)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000,    nan, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000,    nan, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """
    if validate_args:
        _multilabel_precision_recall_curve_arg_validation(
            num_labels, thresholds, ignore_index
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
    return _multilabel_precision_recall_curve_compute(
        state, num_labels, thresholds, ignore_index
    )


def precision_recall_curve(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: int | list[float] | paddle.Tensor | None = None,
    num_classes: int | None = None,
    num_labels: int | None = None,
    average: Literal["micro", "macro"] | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> (
    tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
):
    """Compute the precision-recall curve.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_precision_recall_curve`,
    :func:`~paddle.metric.functional.classification.multiclass_precision_recall_curve` and
    :func:`~paddle.metric.functional.classification.multilabel_precision_recall_curve` for the specific details of each
    argument influence and examples.

    Legacy Example:
        >>> pred = paddle.to_tensor([0, 0.1, 0.8, 0.4])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, task='binary')
        >>> precision
        tensor([0.5000, 0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.0000, 0.1000, 0.4000, 0.8000])

        >>> pred = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, task='multiclass', num_classes=5)
        >>> precision
        [tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 1., 0.]), tensor([1., 1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]),
         tensor([0.0500])]

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_precision_recall_curve(
            preds, target, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_precision_recall_curve(
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
        return multilabel_precision_recall_curve(
            preds, target, num_labels, thresholds, ignore_index, validate_args
        )
    raise ValueError(f"Task {task} not supported.")
