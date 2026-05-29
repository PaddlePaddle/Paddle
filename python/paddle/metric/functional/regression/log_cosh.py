from __future__ import annotations

import paddle
from paddle.metric.functional.regression.utils import (
    _check_data_shape_to_num_outputs,
)
from paddle.metric.utils.checks import _check_same_shape


def _unsqueeze_tensors(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    if preds.ndim == 2:
        return preds, target
    return preds.unsqueeze(1), target.unsqueeze(1)


def _log_cosh_error_update(
    preds: paddle.Tensor, target: paddle.Tensor, num_outputs: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute LogCosh error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    Return:
        Sum of LogCosh error over examples, and total number of examples

    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    preds, target = _unsqueeze_tensors(preds, target)
    diff = preds - target
    sum_log_cosh_error = (
        paddle.log((paddle.exp(diff) + paddle.exp(-diff)) / 2).sum(0).squeeze()
    )
    num_obs = paddle.tensor(target.shape[0], device=preds.place)
    return sum_log_cosh_error, num_obs


def _log_cosh_error_compute(
    sum_log_cosh_error: paddle.Tensor, num_obs: paddle.Tensor
) -> paddle.Tensor:
    """Compute Mean Squared Error.

    Args:
        sum_log_cosh_error: Sum of LogCosh errors over all observations
        num_obs: Number of predictions or observations

    """
    return (sum_log_cosh_error / num_obs).squeeze()


def log_cosh_error(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute the `LogCosh Error`_.

    .. math:: \\text{LogCoshError} = \\log\\left(\\frac{\\exp(\\hat{y} - y) + \\exp(\\hat{y - y})}{2}\\right)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    Args:
        preds: estimated labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``
        target: ground truth labels with shape ``(batch_size,)`` or `(batch_size, num_outputs)``

    Return:
        Tensor with LogCosh error

    Example (single output regression)::
        >>> from paddle.metric.functional.regression import log_cosh_error
        >>> preds = paddle.to_tensor([3.0, 5.0, 2.5, 7.0])
        >>> target = paddle.to_tensor([2.5, 5.0, 4.0, 8.0])
        >>> log_cosh_error(preds, target)
        tensor(0.3523)

    Example (multi output regression)::
        >>> from paddle.metric.functional.regression import log_cosh_error
        >>> preds = paddle.to_tensor([[3.0, 5.0, 1.2], [-2.1, 2.5, 7.0]])
        >>> target = paddle.to_tensor([[2.5, 5.0, 1.3], [0.3, 4.0, 8.0]])
        >>> log_cosh_error(preds, target)
        tensor([0.9176, 0.4277, 0.2194])

    """
    sum_log_cosh_error, num_obs = _log_cosh_error_update(
        preds, target, num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    return _log_cosh_error_compute(sum_log_cosh_error, num_obs)
