from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape


def _weighted_mean_absolute_percentage_error_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Weighted Absolute Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    sum_abs_error = (preds - target).abs().sum()
    sum_scale = target.abs().sum()
    return sum_abs_error, sum_scale


def _weighted_mean_absolute_percentage_error_compute(
    sum_abs_error: paddle.Tensor, sum_scale: paddle.Tensor, epsilon: float = 1.17e-06
) -> paddle.Tensor:
    """Compute Weighted Absolute Percentage Error.

    Args:
        sum_abs_error: scalar with sum of absolute errors
        sum_scale: scalar with sum of target values
        epsilon: small float to prevent division by zero

    """
    return sum_abs_error / paddle.clamp(sum_scale, min=epsilon)


def weighted_mean_absolute_percentage_error(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Compute weighted mean absolute percentage error (`WMAPE`_).

    The output of WMAPE metric is a non-negative floating point, where the optimal value is 0. It is computes as:

    .. math::
        \\text{WMAPE} = \\frac{\\sum_{t=1}^n | y_t - \\hat{y}_t | }{\\sum_{t=1}^n |y_t| }

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with WMAPE.

    Example:
        >>> from paddle import randn
        >>> preds = randn(20,)
        >>> target = randn(20,)
        >>> weighted_mean_absolute_percentage_error(preds, target)
        tensor(1.3967)

    """
    sum_abs_error, sum_scale = _weighted_mean_absolute_percentage_error_update(
        preds, target
    )
    return _weighted_mean_absolute_percentage_error_compute(sum_abs_error, sum_scale)
