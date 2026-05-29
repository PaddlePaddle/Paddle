from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape


def _crps_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[int, paddle.Tensor, paddle.Tensor]:
    """Compute intermediate CRPS values before aggregation.

    Args:
        preds: Tensor of shape (batch_size, ensemble_members)
        target: Tensor of shape (batch_size,)

    Returns:
        batch_size: int
        diff: Tensor (batch-wise absolute error term)
        ensemble_sum: Tensor (pairwise ensemble term)

    """
    _check_same_shape(preds[:, 0], target)
    batch_size, n_ensemble_members = preds.shape
    if n_ensemble_members < 2:
        raise ValueError(
            f"CRPS requires at least 2 ensemble members, but you provided {preds.shape}."
        )
    preds = paddle.sort(preds, axis=1)[0]
    observation_inflated = target.unsqueeze(1).expand_as(preds)
    diff = (
        paddle.sum(paddle.abs(preds - observation_inflated), axis=1) / n_ensemble_members
    )
    ensemble_diffs = paddle.abs(preds.unsqueeze(2) - preds.unsqueeze(1))
    ensemble_sum = paddle.sum(ensemble_diffs, axis=(1, 2)) / (
        2 * n_ensemble_members * n_ensemble_members
    )
    return batch_size, diff, ensemble_sum


def _crps_compute(
    batch_size: int, diff: paddle.Tensor, ensemble_sum: paddle.Tensor
) -> paddle.Tensor:
    """Final CRPS computation."""
    return paddle.mean(diff - ensemble_sum)


def continuous_ranked_probability_score(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Computes continuous ranked probability score.

    .. math::
        CRPS(F, y) = \\int_{-\\infty}^{\\infty} (F(x) - 1_{x \\geq y})^2 dx

    where :math:`F` is the predicted cumulative distribution function and :math:`y` is the true target. The metric is
    usually used to evaluate probabilistic regression models, such as forecasting models. A lower CRPS indicates a
    better forecast, meaning that forecasted probabilities are closer to the true observed values. CRPS can also be
    seen as a generalization of the brier score for non binary classification problems.

    Args:
        preds: a 2d tensor of shape (batch_size, ensemble_members) with predictions. The second dimension represents
            the ensemble members.
        target: a 1d tensor of shape (batch_size) with the target values.

    Return:
        Tensor with CRPS

    Raises:
        ValueError:
            If the number of ensemble members is less than 2.
        ValueError:
            If the first dimension of preds and target do not match.

    Example::
        >>> from paddle.metric.functional.regression import continuous_ranked_probability_score
        >>> from paddle import randn
        >>> preds = randn(10, 5)
        >>> target = randn(10)
        >>> continuous_ranked_probability_score(preds, target)
        tensor(0.7731)

    """
    batch_size, diff, ensemble_sum = _crps_update(preds, target)
    return _crps_compute(batch_size, diff, ensemble_sum)
