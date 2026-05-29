from __future__ import annotations

import paddle
from paddle.metric.utils import rank_zero_warn
from paddle.metric.utils.checks import _check_same_shape


def _r2_score_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, int]:
    """Update and returns variables required to compute R2 score.

    Check for same shape and 1D/2D input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    if preds.ndim > 2:
        raise ValueError(
            f"Expected both prediction and target to be 1D or 2D tensors, but received tensors with dimension {preds.shape}"
        )
    sum_obs = paddle.sum(target, axis=0)
    sum_squared_obs = paddle.sum(target * target, axis=0)
    residual = target - preds
    rss = paddle.sum(residual * residual, axis=0)
    return sum_squared_obs, sum_obs, rss, target.size(0)


def _r2_score_compute(
    sum_squared_obs: paddle.Tensor,
    sum_obs: paddle.Tensor,
    rss: paddle.Tensor,
    num_obs: int | paddle.Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> paddle.Tensor:
    """Compute R2 score.

    Args:
        sum_squared_obs: Sum of square of all observations
        sum_obs: Sum of all observations
        rss: Residual sum of squares
        num_obs: Number of predictions or observations
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * `'raw_values'` returns full set of scores
            * `'uniform_average'` scores are uniformly averaged
            * `'variance_weighted'` scores are weighted by their individual variances

    Example:
        >>> target = paddle.to_tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = paddle.to_tensor([[0, 2], [-1, 2], [8, -5]])
        >>> sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
        >>> _r2_score_compute(sum_squared_obs, sum_obs, rss, num_obs, multioutput="raw_values")
        tensor([0.9654, 0.9082])

    """
    if num_obs < 2:
        raise ValueError("Needs at least two samples to calculate r2 score.")
    mean_obs = sum_obs / num_obs
    tss = sum_squared_obs - sum_obs * mean_obs
    cond_rss = ~paddle.isclose(rss, paddle.zeros_like(rss), atol=0.0001)
    cond_tss = ~paddle.isclose(tss, paddle.zeros_like(tss), atol=0.0001)
    cond = cond_rss & cond_tss
    raw_scores = paddle.ones_like(rss)
    raw_scores[cond] = 1 - rss[cond] / tss[cond]
    raw_scores[cond_rss & ~cond_tss] = 0.0
    if multioutput == "raw_values":
        r2 = raw_scores
    elif multioutput == "uniform_average":
        r2 = paddle.mean(raw_scores)
    elif multioutput == "variance_weighted":
        tss_sum = paddle.sum(tss)
        r2 = paddle.sum(tss / tss_sum * raw_scores)
    else:
        raise ValueError(
            f"Argument `multioutput` must be either `raw_values`, `uniform_average` or `variance_weighted`. Received {multioutput}."
        )
    if adjusted < 0 or not isinstance(adjusted, int):
        raise ValueError(
            "`adjusted` parameter should be an integer larger or equal to 0."
        )
    if adjusted != 0:
        if adjusted > num_obs - 1:
            rank_zero_warn(
                "More independent regressions than data points in adjusted r2 score. Falls back to standard r2 score.",
                UserWarning,
            )
        elif adjusted == num_obs - 1:
            rank_zero_warn(
                "Division by zero in adjusted r2 score. Falls back to standard r2 score.",
                UserWarning,
            )
        else:
            return 1 - (1 - r2) * (num_obs - 1) / (num_obs - adjusted - 1)
    return r2


def r2_score(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    adjusted: int = 0,
    multioutput: str = "uniform_average",
) -> paddle.Tensor:
    """Compute r2 score also known as `R2 Score_Coefficient Determination`_.

    .. math:: R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}=\\sum_i (y_i - f(x_i))^2` is the sum of residual squares, and
    :math:`SS_{tot}=\\sum_i (y_i - \\bar{y})^2` is total sum of squares. Can also calculate
    adjusted r2 score given by

    .. math:: R^2_{adj} = 1 - \\frac{(1-R^2)(n-1)}{n-k-1}

    where the parameter :math:`k` (the number of independent regressors) should
    be provided as the ``adjusted`` argument.

    Args:
        preds: estimated labels
        target: ground truth labels
        adjusted: number of independent regressors for calculating adjusted r2 score.
        multioutput: Defines aggregation in the case of multiple output scores. Can be one of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Raises:
        ValueError:
            If both ``preds`` and ``targets`` are not ``1D`` or ``2D`` tensors.
        ValueError:
            If ``len(preds)`` is less than ``2`` since at least ``2`` samples are needed to calculate r2 score.
        ValueError:
            If ``multioutput`` is not one of ``raw_values``, ``uniform_average`` or ``variance_weighted``.
        ValueError:
            If ``adjusted`` is not an ``integer`` greater than ``0``.

    Example:
        >>> from paddle.metric.functional.regression import r2_score
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> r2_score(preds, target)
        tensor(0.9486)

        >>> target = paddle.to_tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = paddle.to_tensor([[0, 2], [-1, 2], [8, -5]])
        >>> r2_score(preds, target, multioutput='raw_values')
        tensor([0.9654, 0.9082])

    """
    sum_squared_obs, sum_obs, rss, num_obs = _r2_score_update(preds, target)
    return _r2_score_compute(
        sum_squared_obs, sum_obs, rss, num_obs, adjusted, multioutput
    )
