from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape


def _mean_squared_error_update(
    preds: paddle.Tensor, target: paddle.Tensor, num_outputs: int
) -> tuple[paddle.Tensor, int]:
    """Update and returns variables required to compute Mean Squared Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds, target)
    if num_outputs == 1:
        preds = preds.view(-1)
        target = target.view(-1)
    diff = preds - target
    sum_squared_error = paddle.sum(diff * diff, axis=0)
    return sum_squared_error, target.shape[0]


def _mean_squared_error_compute(
    sum_squared_error: paddle.Tensor,
    num_obs: int | paddle.Tensor,
    squared: bool = True,
) -> paddle.Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_squared_error: Sum of square of errors over all observations
        num_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.

    Example:
        >>> preds = paddle.to_tensor([0., 1, 2, 3])
        >>> target = paddle.to_tensor([0., 1, 2, 2])
        >>> sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs=1)
        >>> _mean_squared_error_compute(sum_squared_error, num_obs)
        tensor(0.2500)

    """
    return (
        sum_squared_error / num_obs
        if squared
        else paddle.sqrt(sum_squared_error / num_obs)
    )


def mean_squared_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    squared: bool = True,
    num_outputs: int = 1,
) -> paddle.Tensor:
    """Compute mean squared error.

    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with MSE

    Example:
        >>> from paddle.metric.functional.regression import mean_squared_error
        >>> x = paddle.to_tensor([0., 1, 2, 3])
        >>> y = paddle.to_tensor([0., 1, 2, 2])
        >>> mean_squared_error(x, y)
        tensor(0.2500)

    """
    sum_squared_error, num_obs = _mean_squared_error_update(
        preds, target, num_outputs=num_outputs
    )
    return _mean_squared_error_compute(sum_squared_error, num_obs, squared=squared)
