from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.exceptions import MetricUserError


def _minkowski_distance_update(
    preds: paddle.Tensor, targets: paddle.Tensor, p: float
) -> paddle.Tensor:
    """Update and return variables required to compute Minkowski distance.

    Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        targets: Ground truth tensor
        p: Non-negative number acting as the p to the errors

    """
    _check_same_shape(preds, targets)
    if not (isinstance(p, (float, int)) and p >= 1):
        raise MetricUserError(
            f"Argument ``p`` must be a float or int greater than 1, but got {p}"
        )
    difference = paddle.abs(preds - targets)
    return paddle.sum(paddle.pow(difference, p))


def _minkowski_distance_compute(distance: paddle.Tensor, p: float) -> paddle.Tensor:
    """Compute Minkowski Distance.

    Args:
        distance: Sum of the p-th powers of errors over all observations
        p: The non-negative numeric power the errors are to be raised to

    Example:
        >>> preds = paddle.to_tensor([0., 1, 2, 3])
        >>> target = paddle.to_tensor([0., 2, 3, 1])
        >>> distance_p_sum = _minkowski_distance_update(preds, target, 5)
        >>> _minkowski_distance_compute(distance_p_sum, 5)
        tensor(2.0244)

    """
    return paddle.pow(distance, 1.0 / p)


def minkowski_distance(
    preds: paddle.Tensor, targets: paddle.Tensor, p: float
) -> paddle.Tensor:
    """Compute the `Minkowski distance`_.

    .. math:: d_{\\text{Minkowski}} = \\\\sum_{i}^N (| y_i - \\\\hat{y_i} |^p)^\\frac{1}{p}

    This metric can be seen as generalized version of the standard euclidean distance which corresponds to minkowski
    distance with p=2.

    Args:
        preds: estimated labels of type Tensor
        targets: ground truth labels of type Tensor
        p: int or float larger than 1, exponent to which the difference between preds and target is to be raised

    Return:
        Tensor with the Minkowski distance

    Example:
        >>> from paddle.metric.functional.regression import minkowski_distance
        >>> x = paddle.to_tensor([1.0, 2.8, 3.5, 4.5])
        >>> y = paddle.to_tensor([6.1, 2.11, 3.1, 5.6])
        >>> minkowski_distance(x, y, p=3)
        tensor(5.1220)

    """
    minkowski_dist_sum = _minkowski_distance_update(preds, targets, p)
    return _minkowski_distance_compute(minkowski_dist_sum, p)
