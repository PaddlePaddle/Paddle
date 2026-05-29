from __future__ import annotations

from typing import Callable

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from paddle.metric.utils.enums import ClassificationTask


def _lexargmax(x: paddle.Tensor) -> paddle.Tensor:
    """Returns the index of the maximum value in a list of tuples according to lexicographic ordering.

    Based on https://stackoverflow.com/a/65615160

    """
    idx: paddle.Tensor | None = None
    for k in range(x.shape[1]):
        col: Tensor = x[idx, k] if idx is not None else x[:, k]
        z = paddle.where(col == col._max())[0]
        idx = z if idx is None else idx[z]
        if len(idx) < 2:
            break
    if idx is None:
        raise ValueError("Failed to extract index")
    return idx


def _recall_at_precision(
    precision: paddle.Tensor,
    recall: paddle.Tensor,
    thresholds: paddle.Tensor,
    min_precision: float,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    max_recall = paddle.tensor(0.0, device=recall.device, dtype=recall.dtype)
    best_threshold = paddle.tensor(0)
    zipped_len = min(t.shape[0] for t in (recall, precision, thresholds))
    zipped = paddle.vstack(
        x=(recall[:zipped_len], precision[:zipped_len], thresholds[:zipped_len])
    ).T
    zipped_masked = zipped[zipped[:, 1] >= min_precision]
    if zipped_masked.shape[0] > 0:
        idx = _lexargmax(zipped_masked)[0]
        max_recall, _, best_threshold = zipped_masked[idx]
    if max_recall == 0.0:
        best_threshold = paddle.tensor(
            float("nan"), device=thresholds.device, dtype=thresholds.dtype
        )
    return max_recall, best_threshold


def _binary_recall_at_fixed_precision_arg_validation(
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if not isinstance(min_precision, float) and not 0 <= min_precision <= 1:
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _binary_recall_at_fixed_precision_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    thresholds: paddle.Tensor | None,
    min_precision: float,
    pos_label: int = 1,
    reduce_fn: Callable = _recall_at_precision,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    precision, recall, thresholds = _binary_precision_recall_curve_compute(
        state, thresholds, pos_label
    )
    return reduce_fn(precision, recall, thresholds, min_precision)


def binary_recall_at_fixed_precision(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute the highest possible recall value given the minimum precision thresholds provided for binary tasks.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall
    for a given precision level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of 2 tensors containing:

        - recall: an scalar tensor with the maximum recall for the given precision level
        - threshold: an scalar tensor with the corresponding threshold level

    Example:
        >>> from paddle.metric.functional.classification import binary_recall_at_fixed_precision
        >>> preds = paddle.to_tensor([0, 0.5, 0.7, 0.8])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> binary_recall_at_fixed_precision(preds, target, min_precision=0.5, thresholds=None)
        (tensor(1.), tensor(0.5000))
        >>> binary_recall_at_fixed_precision(preds, target, min_precision=0.5, thresholds=5)
        (tensor(1.), tensor(0.5000))

    """
    if validate_args:
        _binary_recall_at_fixed_precision_arg_validation(
            min_precision, thresholds, ignore_index
        )
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(
        preds, target, thresholds, ignore_index
    )
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_recall_at_fixed_precision_compute(state, thresholds, min_precision)


def _multiclass_recall_at_fixed_precision_arg_validation(
    num_classes: int,
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _multiclass_precision_recall_curve_arg_validation(
        num_classes, thresholds, ignore_index
    )
    if not isinstance(min_precision, float) and not 0 <= min_precision <= 1:
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _multiclass_recall_at_fixed_precision_arg_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_classes: int,
    thresholds: paddle.Tensor | None,
    min_precision: float,
    reduce_fn: Callable = _recall_at_precision,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    precision, recall, thresholds = _multiclass_precision_recall_curve_compute(
        state, num_classes, thresholds
    )
    if isinstance(state, paddle.Tensor):
        res = [
            reduce_fn(p, r, thresholds, min_precision)
            for p, r in zip(precision, recall)
        ]
    else:
        res = [
            reduce_fn(p, r, t, min_precision)
            for p, r, t in zip(precision, recall, thresholds)
        ]
    recall = paddle.stack([r[0] for r in res])
    thresholds = paddle.stack([r[1] for r in res])
    return recall, thresholds


def multiclass_recall_at_fixed_precision(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute the highest possible recall value given the minimum precision thresholds provided for multiclass tasks.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - recall: an 1d tensor of size (n_classes, ) with the maximum recall for the given precision level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class

    Example:
        >>> from paddle.metric.functional.classification import multiclass_recall_at_fixed_precision
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> multiclass_recall_at_fixed_precision(preds, target, num_classes=5, min_precision=0.5, thresholds=None)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, nan, nan, nan]))
        >>> multiclass_recall_at_fixed_precision(preds, target, num_classes=5, min_precision=0.5, thresholds=5)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, nan, nan, nan]))

    """
    if validate_args:
        _multiclass_recall_at_fixed_precision_arg_validation(
            num_classes, min_precision, thresholds, ignore_index
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
    return _multiclass_recall_at_fixed_precision_arg_compute(
        state, num_classes, thresholds, min_precision
    )


def _multilabel_recall_at_fixed_precision_arg_validation(
    num_labels: int,
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
) -> None:
    _multilabel_precision_recall_curve_arg_validation(
        num_labels, thresholds, ignore_index
    )
    if not isinstance(min_precision, float) and not 0 <= min_precision <= 1:
        raise ValueError(
            f"Expected argument `min_precision` to be an float in the [0,1] range, but got {min_precision}"
        )


def _multilabel_recall_at_fixed_precision_arg_compute(
    state: paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor],
    num_labels: int,
    thresholds: paddle.Tensor | None,
    ignore_index: int | None,
    min_precision: float,
    reduce_fn: Callable = _recall_at_precision,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    precision, recall, thresholds = _multilabel_precision_recall_curve_compute(
        state, num_labels, thresholds, ignore_index
    )
    if isinstance(state, paddle.Tensor):
        res = [
            reduce_fn(p, r, thresholds, min_precision)
            for p, r in zip(precision, recall)
        ]
    else:
        res = [
            reduce_fn(p, r, t, min_precision)
            for p, r, t in zip(precision, recall, thresholds)
        ]
    recall = paddle.stack([r[0] for r in res])
    thresholds = paddle.stack([r[1] for r in res])
    return recall, thresholds


def multilabel_recall_at_fixed_precision(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute the highest possible recall value given the minimum precision thresholds provided for multilabel tasks.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - recall: an 1d tensor of size (n_classes, ) with the maximum recall for the given precision level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class

    Example:
        >>> from paddle.metric.functional.classification import multilabel_recall_at_fixed_precision
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = paddle.to_tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> multilabel_recall_at_fixed_precision(preds, target, num_labels=3, min_precision=0.5, thresholds=None)
        (tensor([1., 1., 1.]), tensor([0.0500, 0.5500, 0.0500]))
        >>> multilabel_recall_at_fixed_precision(preds, target, num_labels=3, min_precision=0.5, thresholds=5)
        (tensor([1., 1., 1.]), tensor([0.0000, 0.5000, 0.0000]))

    """
    if validate_args:
        _multilabel_recall_at_fixed_precision_arg_validation(
            num_labels, min_precision, thresholds, ignore_index
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
    return _multilabel_recall_at_fixed_precision_arg_compute(
        state, num_labels, thresholds, ignore_index, min_precision
    )


def recall_at_fixed_precision(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    min_precision: float,
    thresholds: int | list[float] | paddle.Tensor | None = None,
    num_classes: int | None = None,
    num_labels: int | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor] | None:
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for a
    given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~paddle.metric.functional.classification.binary_recall_at_fixed_precision`,
    :func:`~paddle.metric.functional.classification.multiclass_recall_at_fixed_precision` and
    :func:`~paddle.metric.functional.classification.multilabel_recall_at_fixed_precision` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_recall_at_fixed_precision(
            preds, target, min_precision, thresholds, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_recall_at_fixed_precision(
            preds,
            target,
            num_classes,
            min_precision,
            thresholds,
            ignore_index,
            validate_args,
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(
                f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
            )
        return multilabel_recall_at_fixed_precision(
            preds,
            target,
            num_labels,
            min_precision,
            thresholds,
            ignore_index,
            validate_args,
        )
    return None
