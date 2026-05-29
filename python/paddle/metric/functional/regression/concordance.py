from __future__ import annotations

import paddle
from paddle.metric.functional.regression.pearson import (
    _pearson_corrcoef_compute,
    _pearson_corrcoef_update,
)


def _concordance_corrcoef_compute(
    max_abs_dev_x: paddle.Tensor,
    max_abs_dev_y: paddle.Tensor,
    mean_x: paddle.Tensor,
    mean_y: paddle.Tensor,
    var_x: paddle.Tensor,
    var_y: paddle.Tensor,
    corr_xy: paddle.Tensor,
    nb: paddle.Tensor,
) -> paddle.Tensor:
    """Compute the final concordance correlation coefficient based on accumulated statistics."""
    pearson = _pearson_corrcoef_compute(
        max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, nb
    )
    var_x = var_x / (nb - 1)
    var_y = var_y / (nb - 1)
    return (
        2.0
        * pearson
        * var_x.sqrt()
        * var_y.sqrt()
        / (var_x + var_y + (mean_x - mean_y) ** 2)
    )


def concordance_corrcoef(preds: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
    """Compute concordance correlation coefficient that measures the agreement between two variables.

    .. math::
        \\rho_c = \\frac{2 \\rho \\sigma_x \\sigma_y}{\\sigma_x^2 + \\sigma_y^2 + (\\mu_x - \\mu_y)^2}

    where :math:`\\mu_x, \\mu_y` is the means for the two variables, :math:`\\sigma_x^2, \\sigma_y^2` are the corresponding
    variances and \\rho is the pearson correlation coefficient between the two variables.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from paddle.metric.functional.regression import concordance_corrcoef
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> concordance_corrcoef(preds, target)
        tensor([0.9777])

    Example (multi output regression):
        >>> from paddle.metric.functional.regression import concordance_corrcoef
        >>> target = paddle.to_tensor([[3, -0.5], [2, 7]])
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> concordance_corrcoef(preds, target)
        tensor([0.7273, 0.9887])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = paddle.zeros(d, dtype=preds.dtype, device=preds.place)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    max_abs_dev_x, max_abs_dev_y = _temp.clone(), _temp.clone()
    (
        mean_x,
        mean_y,
        max_abs_dev_x,
        max_abs_dev_y,
        var_x,
        var_y,
        corr_xy,
        nb,
    ) = _pearson_corrcoef_update(
        preds=preds,
        target=target,
        mean_x=mean_x,
        mean_y=mean_y,
        max_abs_dev_x=max_abs_dev_x,
        max_abs_dev_y=max_abs_dev_y,
        var_x=var_x,
        var_y=var_y,
        corr_xy=corr_xy,
        num_prior=nb,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
    )
    return _concordance_corrcoef_compute(
        max_abs_dev_x, max_abs_dev_y, mean_x, mean_y, var_x, var_y, corr_xy, nb
    )
