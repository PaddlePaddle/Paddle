from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.compute import _safe_divide


def _critical_success_index_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float,
    keep_sequence_dim: int | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Update and return variables required to compute Critical Success Index. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    """
    _check_same_shape(preds, target)
    if keep_sequence_dim is None:
        sum_dims = None
    elif not 0 <= keep_sequence_dim < preds.ndim:
        raise ValueError(
            f"Expected keep_sequence dim to be in range [0, {preds.ndim}] but got {keep_sequence_dim}"
        )
    else:
        sum_dims = tuple(i for i in range(preds.ndim) if i != keep_sequence_dim)
    preds_bin = (preds >= threshold).bool()
    target_bin = (target >= threshold).bool()
    if keep_sequence_dim is None:
        hits = paddle.sum(preds_bin & target_bin).int()
        misses = paddle.sum((preds_bin ^ target_bin) & target_bin).int()
        false_alarms = paddle.sum((preds_bin ^ target_bin) & preds_bin).int()
    else:
        hits = paddle.sum(preds_bin & target_bin, axis=sum_dims).int()
        misses = paddle.sum((preds_bin ^ target_bin) & target_bin, axis=sum_dims).int()
        false_alarms = paddle.sum(
            (preds_bin ^ target_bin) & preds_bin, axis=sum_dims
        ).int()
    return hits, misses, false_alarms


def _critical_success_index_compute(
    hits: paddle.Tensor, misses: paddle.Tensor, false_alarms: paddle.Tensor
) -> paddle.Tensor:
    """Compute critical success index.

    Args:
        hits: Number of true positives after binarization
        misses: Number of false negatives after binarization
        false_alarms: Number of false positives after binarization

    Returns:
        If input tensors are 5-dimensional and ``keep_sequence_dim=True``, the metric returns a ``(S,)`` vector
        with CSI scores for each image in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    """
    return _safe_divide(hits, hits + misses + false_alarms)


def critical_success_index(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    threshold: float,
    keep_sequence_dim: int | None = None,
) -> paddle.Tensor:
    """Compute critical success index.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    Returns:
        If ``keep_sequence_dim`` is specified, the metric returns a vector of  with CSI scores for each image
        in the sequence. Otherwise, it returns a scalar tensor with the CSI score.

    Example:
        >>> import paddle
        >>> from paddle.metric.functional.regression import critical_success_index
        >>> x = paddle.Tensor([[0.2, 0.7], [0.9, 0.3]])
        >>> y = paddle.Tensor([[0.4, 0.2], [0.8, 0.6]])
        >>> critical_success_index(x, y, 0.5)
        tensor(0.3333)

    Example:
        >>> import paddle
        >>> from paddle.metric.functional.regression import critical_success_index
        >>> x = paddle.Tensor([[[0.2, 0.7], [0.9, 0.3]], [[0.2, 0.7], [0.9, 0.3]]])
        >>> y = paddle.Tensor([[[0.4, 0.2], [0.8, 0.6]], [[0.4, 0.2], [0.8, 0.6]]])
        >>> critical_success_index(x, y, 0.5, keep_sequence_dim=0)
        tensor([0.3333, 0.3333])

    """
    hits, misses, false_alarms = _critical_success_index_update(
        preds, target, threshold, keep_sequence_dim
    )
    return _critical_success_index_compute(hits, misses, false_alarms)
