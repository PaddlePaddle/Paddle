from __future__ import annotations

import paddle
from paddle.metric.functional.regression.r2 import _r2_score_update


def _relative_squared_error_compute(
    sum_squared_obs: paddle.Tensor,
    sum_obs: paddle.Tensor,
    sum_squared_error: paddle.Tensor,
    num_obs: int | paddle.Tensor,
    squared: bool = True,
) -> paddle.Tensor:
    """Computes Relative Squared Error.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        sum_squared_error: Residual sum of squares
        num_obs: Number of predictions or observations
        squared: Returns RRSE value if set to False.

    Example:
        >>> target = paddle.to_tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = paddle.to_tensor([[0, 2], [-1, 2], [8, -5]])
        >>> # RSE uses the same update function as R2 score.
        >>> sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
        >>> _relative_squared_error_compute(sum_squared_obs, sum_obs, rss, num_obs, squared=True)
        tensor(0.0632)

    """
    epsilon = paddle.finfo(sum_squared_error.dtype).eps
    rse = sum_squared_error / paddle.clamp(
        sum_squared_obs - sum_obs * sum_obs / num_obs, min=epsilon
    )
    if not squared:
        rse = paddle.sqrt(rse)
    return paddle.mean(rse)


def relative_squared_error(
    preds: paddle.Tensor, target: paddle.Tensor, squared: bool = True
) -> paddle.Tensor:
    """Computes the relative squared error (RSE).

    .. math:: \\text{RSE} = \\frac{\\sum_i^N(y_i - \\hat{y_i})^2}{\\sum_i^N(y_i - \\overline{y})^2}

    Where :math:`y` is a tensor of target values with mean :math:`\\overline{y}`, and
    :math:`\\hat{y}` is a tensor of predictions.

    If `preds` and `targets` are 2D tensors, the RSE is averaged over the second dim.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RRSE value if set to False
    Return:
        Tensor with RSE

    Example:
        >>> from paddle.metric.functional.regression import relative_squared_error
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> relative_squared_error(preds, target)
        tensor(0.0514)

    """
    sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
    return _relative_squared_error_compute(
        sum_squared_obs, sum_obs, rss, num_obs, squared=squared
    )
