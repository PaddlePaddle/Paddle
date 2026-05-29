from __future__ import annotations

import paddle
from paddle.metric.functional.regression.utils import (
    _check_data_shape_to_num_outputs,
)
from paddle.metric.utils.checks import _check_same_shape


def _rank_data(data: paddle.Tensor) -> paddle.Tensor:
    """Calculate the rank for each element of a tensor.

    The rank refers to the indices of an element in the corresponding sorted tensor (starting from 1). Duplicates of the
    same value will be assigned the mean of their rank.

    Adopted from `Rank of element tensor`_

    """
    n = data.size
    rank = paddle.empty_like(data, dtype=paddle.int32)
    idx = data.argsort()
    rank[idx[:n]] = paddle.arange(1, n + 1, dtype=paddle.int32, device=data.place)
    uniq, inv, counts = paddle.unique(
        data, sorted=True, return_inverse=True, return_counts=True
    )
    sum_ranks = paddle.zeros_like(uniq, dtype=paddle.int32)
    sum_ranks.scatter_add_(0, inv, rank.to(paddle.int32))
    mean_ranks = sum_ranks / counts
    return mean_ranks[inv]


def _spearman_corrcoef_update(
    preds: paddle.Tensor, target: paddle.Tensor, num_outputs: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Spearman Correlation Coefficient.

    Check for same shape and type of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    if not (preds.is_floating_point() and target.is_floating_point()):
        raise TypeError(
            "Expected `preds` and `target` both to be floating point tensors, but got {pred.dtype} and {target.dtype}"
        )
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    return preds, target


def _spearman_corrcoef_compute(
    preds: paddle.Tensor, target: paddle.Tensor, eps: float = 1e-06
) -> paddle.Tensor:
    """Compute Spearman Correlation Coefficient.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        eps: Avoids ``ZeroDivisionError``.

    Example:
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> preds, target = _spearman_corrcoef_update(preds, target, num_outputs=1)
        >>> _spearman_corrcoef_compute(preds, target)
        tensor(1.0000)

    """
    if preds.ndim == 1:
        preds = _rank_data(preds)
        target = _rank_data(target)
    else:
        preds = paddle.stack([_rank_data(p) for p in preds.T]).T
        target = paddle.stack([_rank_data(t) for t in target.T]).T
    preds_diff = preds - preds.mean(0)
    target_diff = target - target.mean(0)
    cov = (preds_diff * target_diff).mean(0)
    preds_std = paddle.sqrt((preds_diff * preds_diff).mean(0))
    target_std = paddle.sqrt((target_diff * target_diff).mean(0))
    corrcoef = cov / (preds_std * target_std + eps)
    return paddle.clamp(corrcoef, -1.0, 1.0)


def spearman_corrcoef(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute `spearmans rank correlation coefficient`_.

    .. math:
        r_s = = \\frac{cov(rg_x, rg_y)}{\\sigma_{rg_x} * \\sigma_{rg_y}}

    where :math:`rg_x` and :math:`rg_y` are the rank associated to the variables x and y. Spearmans correlations
    coefficient corresponds to the standard pearsons correlation coefficient calculated on the rank variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from paddle.metric.functional.regression import spearman_corrcoef
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> spearman_corrcoef(preds, target)
        tensor(1.0000)

    Example (multi output regression):
        >>> from paddle.metric.functional.regression import spearman_corrcoef
        >>> target = paddle.to_tensor([[3, -0.5], [2, 7]])
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> spearman_corrcoef(preds, target)
        tensor([1.0000, 1.0000])

    """
    preds, target = _spearman_corrcoef_update(
        preds, target, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _spearman_corrcoef_compute(preds, target)
