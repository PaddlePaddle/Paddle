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
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.compute import normalize_logits_if_needed
from paddle.metric.utils.data import _bincount, select_topk
from paddle.metric.utils.enums import ClassificationTask


def _binary_stat_scores_arg_validation(
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if not (isinstance(threshold, float) and 0 <= threshold <= 1):
        raise ValueError(
            f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}."
        )
    allowed_multidim_average = "global", "samplewise"
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )
    if zero_division not in [0, 1]:
        raise ValueError(
            f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}."
        )


def _binary_stat_scores_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 2 dimensional

    """
    _check_same_shape(preds, target)
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
    if not preds.is_floating_point():
        unique_values = paddle.unique(preds, axis=None)
        if paddle.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only the following values [0,1] since `preds` is a label tensor."
            )
    if multidim_average != "global" and preds.ndim < 2:
        raise ValueError(
            "Expected input to be at least 2D when multidim_average is set to `samplewise`"
        )


def _binary_stat_scores_format(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all datapoints that should be ignored with negative values

    """
    if preds.is_floating_point():
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold
    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1
    return preds, target


def _binary_stat_scores_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute the statistics."""
    # Ensure same dtype for comparison (Paddle requires same type)
    if preds.dtype != target.dtype:
        preds = preds.cast(target.dtype)
    sum_dim = [0, 1] if multidim_average == "global" else [1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _binary_stat_scores_compute(
    tp: paddle.Tensor,
    fp: paddle.Tensor,
    tn: paddle.Tensor,
    fn: paddle.Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
) -> paddle.Tensor:
    """Stack statistics and compute support also."""
    return paddle.stack(
        [tp, fp, tn, fn, tp + fn], axis=0 if multidim_average == "global" else 1
    ).squeeze()


def binary_stat_scores(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute the true positives, false positives, true negatives, false negatives, support for binary tasks.

    Related to `Type I and Type II errors`_.

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

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on the ``multidim_average`` parameter:

        - If ``multidim_average`` is set to ``global``, the shape will be ``(5,)``
        - If ``multidim_average`` is set to ``samplewise``, the shape will be ``(N, 5)``

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import binary_stat_scores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import binary_stat_scores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import binary_stat_scores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> binary_stat_scores(preds, target, multidim_average='samplewise')
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])

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
    return _binary_stat_scores_compute(tp, fp, tn, fn, multidim_average)


def _multiclass_stat_scores_arg_validation(
    num_classes: int | None,
    top_k: int = 1,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``top_k`` has to be an int larger than 0 but no larger than number of classes
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if num_classes is None and average != "micro":
        raise ValueError(
            f"Argument `num_classes` can only be `None` for `average='micro'`, but got `average={average}`."
        )
    if num_classes is not None and (
        not isinstance(num_classes, int) or num_classes < 2
    ):
        raise ValueError(
            f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
        )
    if not isinstance(top_k, int) and top_k < 1:
        raise ValueError(
            f"Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}"
        )
    if top_k > (num_classes if num_classes is not None else 1):
        raise ValueError(
            f"Expected argument `top_k` to be smaller or equal to `num_classes` but got {top_k} and {num_classes}"
        )
    allowed_average = "micro", "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, but got {average}"
        )
    allowed_multidim_average = "global", "samplewise"
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )
    if zero_division not in [0, 1]:
        raise ValueError(
            f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}."
        )


def _multiclass_stat_scores_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int | None,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
) -> None:
    """Validate tensor input.

    - if preds has one more dimension than target, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 2 dimensional in the
    int case and 3 dimensional in the float case
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}

    """
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds` should be a float tensor."
            )
        if num_classes is not None and preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds.shape[1]` should be equal to number of classes."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be (N, C, ...), and the shape of `target` should be (N, ...)."
            )
        if multidim_average != "global" and preds.ndim < 3:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be at least 3D when multidim_average is set to `samplewise`"
            )
    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                f"The `preds` and `target` should have the same shape, got `preds` with shape={preds.shape} and `target` with shape={target.shape}."
            )
        if multidim_average != "global" and preds.ndim < 2:
            raise ValueError(
                "When `preds` and `target` have the same shape, the shape of `preds` should be at least 2D when multidim_average is set to `samplewise`"
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...) and `preds` should be (N, C, ...)."
        )
    if num_classes is not None:
        check_value = num_classes if ignore_index is None else num_classes + 1
        for t, name in (
            ((target, "target"), (preds, "preds"))
            if not preds.is_floating_point()
            else ()
        ):
            num_unique_values = len(paddle.unique(t, axis=None))
            if num_unique_values > check_value:
                raise RuntimeError(
                    f"Detected more unique values in `{name}` than expected. Expected only {check_value} but found {num_unique_values} in `{name}`. Found values: {paddle.unique(t, axis=None)}."
                )


def _multiclass_stat_scores_format(
    preds: paddle.Tensor, target: paddle.Tensor, top_k: int = 1
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions

    """
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = preds.argmax(dim=1)
    preds = (
        preds.reshape(*preds.shape[:2], -1)
        if top_k != 1
        else preds.reshape(preds.shape[0], -1)
    )
    target = target.reshape(target.shape[0], -1)
    return preds, target


def _refine_preds_oh(
    preds: paddle.Tensor,
    preds_oh: paddle.Tensor,
    target: paddle.Tensor,
    top_k: int,
) -> paddle.Tensor:
    """Refines prediction one-hot encodings by replacing entries with target one-hot when there's an intersection.

    When no intersection is found between the top-k predictions and target, uses the top-1 prediction.

    Args:
        preds: Original prediction tensor with probabilities/logits
        preds_oh: Current one-hot encoded predictions from top-k selection
        target: Target tensor with class indices
        top_k: Number of top predictions to consider

    Returns:
        Refined one-hot encoded predictions tensor

    """
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
        target = target.unsqueeze(0) if target.dim() == 0 else target
    top_k_indices = paddle.topk(preds, k=top_k, axis=1).indices
    top_1_indices = top_k_indices[:, 0]
    target_in_topk = paddle.any(top_k_indices == target.unsqueeze(1), axis=1)
    result = paddle.where(target_in_topk, target, top_1_indices)
    return paddle.zeros_like(preds_oh, dtype=paddle.int32).scatter_(
        -1, result.unsqueeze(-1), 1
    )


def _multiclass_stat_scores_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    top_k: int = 1,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute the statistics.

    - If ``multidim_average`` is equal to samplewise or ``top_k`` is not 1, we transform both preds and
    target into one hot format.
    - Else we calculate statistics by first calculating the confusion matrix and afterwards deriving the
    statistics from that
    - Remove all datapoints that should be ignored. Depending on if ``ignore_index`` is in the set of labels
    or outside we have do use different augmentation strategies when one hot encoding.

    """
    if multidim_average == "samplewise" or top_k != 1:
        ignore_in = (
            0 <= ignore_index <= num_classes - 1
            if ignore_index is not None
            else None
        )
        if ignore_index is not None and not ignore_in:
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            target[idx] = num_classes
            idx = (
                idx.unsqueeze(1).repeat(1, num_classes, 1)
                if preds.ndim > target.ndim
                else idx
            )
            preds[idx] = num_classes
        if top_k > 1:
            preds_oh = paddle.moveaxis(
                x=select_topk(preds, topk=top_k, axis=1),
                source=1,
                destination=-1,
            )
            preds_oh = _refine_preds_oh(preds, preds_oh, target, top_k)
        else:
            preds_oh = paddle.nn.functional.one_hot(
                preds.long(),
                num_classes + 1
                if ignore_index is not None and not ignore_in
                else num_classes,
            )
        target_oh = paddle.nn.functional.one_hot(
            target.long(),
            num_classes + 1
            if ignore_index is not None and not ignore_in
            else num_classes,
        )
        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        preds_oh = preds_oh.astype(target_oh.dtype)
        sum_dim = [0, 1] if multidim_average == "global" else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
    elif average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        tp = (preds == target).sum()
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        tn = num_classes * preds.size - (fp + fn + tp)
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = target.to(paddle.long) * num_classes + preds.to(
            paddle.long
        )
        bins = _bincount(unique_mapping, minlength=num_classes**2)
        confmat = bins.reshape(num_classes, num_classes)
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
    return tp, fp, tn, fn


def _multiclass_stat_scores_compute(
    tp: paddle.Tensor,
    fp: paddle.Tensor,
    tn: paddle.Tensor,
    fn: paddle.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> paddle.Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.

    """
    res = paddle.stack([tp, fp, tn, fn, tp + fn], axis=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim) if res.ndim > 1 else res
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        weight = tp + fn
        if multidim_average == "global":
            return (
                res * (weight / weight.sum()).reshape(*weight.shape, 1)
            ).sum(sum_dim)
        return (
            res
            * (weight / weight.sum(-1, keepdim=True)).reshape(*weight.shape, 1)
        ).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multiclass_stat_scores(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multiclass tasks.

    Related to `Type I and Type II errors`_.

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
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
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
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import multiclass_stat_scores
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average='micro')
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])

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
    return _multiclass_stat_scores_compute(
        tp, fp, tn, fn, average, multidim_average
    )


def _multilabel_stat_scores_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` should be an int larger than 1
    - ``threshold`` has to be a float in the [0,1] range
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(
            f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}"
        )
    if not (isinstance(threshold, float) and 0 <= threshold <= 1):
        raise ValueError(
            f"Expected argument `threshold` to be a float, but got {threshold}."
        )
    allowed_average = "micro", "macro", "weighted", "none", None
    if average not in allowed_average:
        raise ValueError(
            f"Expected argument `average` to be one of {allowed_average}, but got {average}"
        )
    allowed_multidim_average = "global", "samplewise"
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )
    if zero_division not in [0, 1]:
        raise ValueError(
            f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}."
        )


def _multilabel_stat_scores_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    multidim_average: str,
    ignore_index: int | None = None,
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - the second dimension of both tensors need to be equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 3 dimensional

    """
    _check_same_shape(preds, target)
    if preds.shape[1] != num_labels:
        raise ValueError(
            f"Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels but got {preds.shape[1]} and expected {num_labels}"
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
    if not preds.is_floating_point():
        unique_values = paddle.unique(preds, axis=None)
        if paddle.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only the following values [0,1] since preds is a label tensor."
            )
    if multidim_average != "global" and preds.ndim < 3:
        raise ValueError(
            "Expected input to be at least 3D when multidim_average is set to `samplewise`"
        )


def _multilabel_stat_scores_format(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    threshold: float = 0.5,
    ignore_index: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all elements that should be ignored with negative numbers for later filtration

    """
    if preds.is_floating_point():
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1
    return preds, target


def _multilabel_stat_scores_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute the statistics."""
    sum_dim = [0, -1] if multidim_average == "global" else [-1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _multilabel_stat_scores_compute(
    tp: paddle.Tensor,
    fp: paddle.Tensor,
    tn: paddle.Tensor,
    fn: paddle.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> paddle.Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.

    """
    res = paddle.stack([tp, fp, tn, fn, tp + fn], axis=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim)
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(*w.shape, 1)).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multilabel_stat_scores(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute the true positives, false positives, true negatives, false negatives and support for multilabel tasks.

    Related to `Type I and Type II errors`_.

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
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (preds is float tensor):
        >>> from paddle.metric.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (multidim tensors):
        >>> from paddle.metric.functional.classification import multilabel_stat_scores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average='micro')
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[[1, 1, 0, 0, 1],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1]],
                [[0, 0, 0, 2, 2],
                 [0, 2, 0, 0, 0],
                 [0, 0, 1, 1, 1]]])

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
    return _multilabel_stat_scores_compute(
        tp, fp, tn, fn, average, multidim_average
    )


def stat_scores(
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
) -> paddle.Tensor:
    """Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_stat_scores`,
    :func:`~paddle.metric.functional.classification.multiclass_stat_scores` and
    :func:`~paddle.metric.functional.classification.multilabel_stat_scores` for the specific
    details of each argument influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> preds = tensor([1, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average='micro')
        tensor([2, 2, 6, 2, 4])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average=None)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None
    if task == ClassificationTask.BINARY:
        return binary_stat_scores(
            preds,
            target,
            threshold,
            multidim_average,
            ignore_index,
            validate_args,
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
        return multiclass_stat_scores(
            preds,
            target,
            num_classes,
            average,
            top_k,
            multidim_average,
            ignore_index,
            validate_args,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_stat_scores(
            preds,
            target,
            num_labels,
            threshold,
            average,
            multidim_average,
            ignore_index,
            validate_args,
        )
    raise ValueError(f"Unsupported task `{task}`")
