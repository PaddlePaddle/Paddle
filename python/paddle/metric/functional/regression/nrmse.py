from __future__ import annotations

from typing_extensions import Literal

import paddle
from paddle.metric.functional.regression.mse import _mean_squared_error_update


def _normalized_root_mean_squared_error_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_outputs: int,
    normalization: Literal["mean", "range", "std", "l2"] = "mean",
) -> tuple[paddle.Tensor, int, paddle.Tensor]:
    """Updates and returns the sum of squared errors and the number of observations for NRMSE computation.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting
        normalization: type of normalization to be applied. Choose from "mean", "range", "std", "l2"

    """
    sum_squared_error, num_obs = _mean_squared_error_update(preds, target, num_outputs)
    target = target.view(-1) if num_outputs == 1 else target
    if normalization == "mean":
        denom = paddle.mean(target, axis=0)
    elif normalization == "range":
        denom = paddle.max(target, axis=0) - paddle.min(target, axis=0)
    elif normalization == "std":
        denom = paddle.std(x=target, unbiased=0, axis=0)
    elif normalization == "l2":
        denom = paddle.norm(target, p=2, axis=0)
    else:
        raise ValueError(
            f"Argument `normalization` should be either 'mean', 'range', 'std' or 'l2' but got {normalization}"
        )
    return sum_squared_error, num_obs, denom


def _normalized_root_mean_squared_error_compute(
    sum_squared_error: paddle.Tensor,
    num_obs: int | paddle.Tensor,
    denom: paddle.Tensor,
) -> paddle.Tensor:
    """Calculates RMSE and normalizes it."""
    rmse = paddle.sqrt(sum_squared_error / num_obs)
    return rmse / denom


def normalized_root_mean_squared_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    normalization: Literal["mean", "range", "std", "l2"] = "mean",
    num_outputs: int = 1,
) -> paddle.Tensor:
    """Calculates the `Normalized Root Mean Squared Error`_ (NRMSE) also know as scatter index.

    Args:
        preds: estimated labels
        target: ground truth labels
        normalization: type of normalization to be applied. Choose from "mean", "range", "std", "l2" which corresponds
          to normalizing the RMSE by the mean of the target, the range of the target, the standard deviation of the
          target or the L2 norm of the target.
        num_outputs: Number of outputs in multioutput setting

    Return:
        Tensor with the NRMSE score

    Example:
        >>> import paddle
        >>> from paddle.metric.functional.regression import normalized_root_mean_squared_error
        >>> preds = paddle.to_tensor([0., 1, 2, 3])
        >>> target = paddle.to_tensor([0., 1, 2, 2])
        >>> normalized_root_mean_squared_error(preds, target, normalization="mean")
        tensor(0.4000)
        >>> normalized_root_mean_squared_error(preds, target, normalization="range")
        tensor(0.2500)
        >>> normalized_root_mean_squared_error(preds, target, normalization="std")
        tensor(0.6030)
        >>> normalized_root_mean_squared_error(preds, target, normalization="l2")
        tensor(0.1667)

    Example (multioutput):
        >>> import paddle
        >>> from paddle.metric.functional.regression import normalized_root_mean_squared_error
        >>> preds = paddle.to_tensor([[0., 1], [2, 3], [4, 5], [6, 7]])
        >>> target = paddle.to_tensor([[0., 1], [3, 3], [4, 5], [8, 9]])
        >>> normalized_root_mean_squared_error(preds, target, normalization="mean", num_outputs=2)
        tensor([0.2981, 0.2222])

    """
    sum_squared_error, num_obs, denom = _normalized_root_mean_squared_error_update(
        preds, target, num_outputs=num_outputs, normalization=normalization
    )
    return _normalized_root_mean_squared_error_compute(
        sum_squared_error, num_obs, denom
    )
