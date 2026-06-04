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


def _mean_squared_log_error_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, int]:
    """Return variables required to compute Mean Squared Log Error. Checks for same shape of tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    sum_squared_log_error = paddle.sum(
        paddle.pow(paddle.log1p(x=preds) - paddle.log1p(x=target), 2)
    )
    return sum_squared_log_error, target.size


def _mean_squared_log_error_compute(
    sum_squared_log_error: paddle.Tensor, num_obs: int | paddle.Tensor
) -> paddle.Tensor:
    """Compute Mean Squared Log Error.

    Args:
        sum_squared_log_error:
            Sum of square of log errors over all observations ``(log error = log(target) - log(prediction))``
        num_obs: Number of predictions or observations

    Example:
        >>> preds = paddle.to_tensor([0.0, 1, 2, 3])
        >>> target = paddle.to_tensor([0.0, 1, 2, 2])
        >>> sum_squared_log_error, num_obs = _mean_squared_log_error_update(preds, target)
        >>> _mean_squared_log_error_compute(sum_squared_log_error, num_obs)
        tensor(0.0207)

    """
    return sum_squared_log_error / num_obs


def mean_squared_log_error(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Compute mean squared log error.

    Args:
        preds: estimated labels
        target: ground truth labels

    Return:
        Tensor with RMSLE

    Example:
        >>> from paddle.metric.functional.regression import mean_squared_log_error
        >>> x = paddle.to_tensor([0.0, 1, 2, 3])
        >>> y = paddle.to_tensor([0.0, 1, 2, 2])
        >>> mean_squared_log_error(x, y)
        tensor(0.0207)

    .. attention::
        Half precision is only support on GPU for this metric.

    """
    sum_squared_log_error, num_obs = _mean_squared_log_error_update(
        preds, target
    )
    return _mean_squared_log_error_compute(sum_squared_log_error, num_obs)
