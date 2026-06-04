# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape


def _mean_absolute_percentage_error_update(
    preds: paddle.Tensor, target: paddle.Tensor, epsilon: float = 1.17e-06
) -> tuple[paddle.Tensor, int]:
    """Update and returns variables required to compute Mean Percentage Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        epsilon: Specifies the lower bound for target values. Any target value below epsilon
            is set to epsilon (avoids ``ZeroDivisionError``).

    """
    _check_same_shape(preds, target)
    abs_diff = paddle.abs(preds - target)
    abs_per_error = abs_diff / paddle.clamp(paddle.abs(target), min=epsilon)
    sum_abs_per_error = paddle.sum(abs_per_error)
    num_obs = target.size
    return sum_abs_per_error, num_obs


def _mean_absolute_percentage_error_compute(
    sum_abs_per_error: paddle.Tensor, num_obs: int | paddle.Tensor
) -> paddle.Tensor:
    """Compute Mean Absolute Percentage Error.

    Args:
        sum_abs_per_error: Sum of absolute value of percentage errors over all observations
            ``(percentage error = (target - prediction) / target)``
        num_obs: Number of predictions or observations

    Example:
        >>> target = paddle.to_tensor([1, 10, 1e6])
        >>> preds = paddle.to_tensor([0.9, 15, 1.2e6])
        >>> sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)
        >>> _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
        tensor(0.2667)

    """
    return sum_abs_per_error / num_obs


def mean_absolute_percentage_error(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Compute mean absolute percentage error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with MAPE

    Note:
        The epsilon value is taken from `scikit-learn's implementation of MAPE`_.

    Example:
        >>> from paddle.metric.functional.regression import mean_absolute_percentage_error
        >>> target = paddle.to_tensor([1, 10, 1e6])
        >>> preds = paddle.to_tensor([0.9, 15, 1.2e6])
        >>> mean_absolute_percentage_error(preds, target)
        tensor(0.2667)

    """
    sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(
        preds, target
    )
    return _mean_absolute_percentage_error_compute(sum_abs_per_error, num_obs)
