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

from typing import TYPE_CHECKING

from typing_extensions import Literal

import paddle
from paddle.metric.utils.checks import _check_same_shape

if TYPE_CHECKING:
    from collections.abc import Sequence

ALLOWED_MULTIOUTPUT = "raw_values", "uniform_average", "variance_weighted"


def _explained_variance_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Explained Variance. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    num_obs = preds.size(0)
    sum_error = paddle.sum(target - preds, axis=0)
    diff = target - preds
    sum_squared_error = paddle.sum(diff * diff, axis=0)
    sum_target = paddle.sum(target, axis=0)
    sum_squared_target = paddle.sum(target * target, axis=0)
    return (
        num_obs,
        sum_error,
        sum_squared_error,
        sum_target,
        sum_squared_target,
    )


def _explained_variance_compute(
    num_obs: int | paddle.Tensor,
    sum_error: paddle.Tensor,
    sum_squared_error: paddle.Tensor,
    sum_target: paddle.Tensor,
    sum_squared_target: paddle.Tensor,
    multioutput: Literal[
        "raw_values", "uniform_average", "variance_weighted"
    ] = "uniform_average",
) -> paddle.Tensor:
    """Compute Explained Variance.

    Args:
        num_obs: Number of predictions or observations
        sum_error: Sum of errors over all observations
        sum_squared_error: Sum of square of errors over all observations
        sum_target: Sum of target values
        sum_squared_target: Sum of squares of target values
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings:

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Example:
        >>> target = paddle.to_tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = paddle.to_tensor([[0, 2], [-1, 2], [8, -5]])
        >>> num_obs, sum_error, ss_error, sum_target, ss_target = _explained_variance_update(preds, target)
        >>> _explained_variance_compute(num_obs, sum_error, ss_error, sum_target, ss_target, multioutput='raw_values')
        tensor([0.9677, 1.0000])

    """
    diff_avg = sum_error / num_obs
    numerator = sum_squared_error / num_obs - diff_avg * diff_avg
    target_avg = sum_target / num_obs
    denominator = sum_squared_target / num_obs - target_avg * target_avg
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = paddle.ones_like(diff_avg)
    output_scores[valid_score] = (
        1.0 - numerator[valid_score] / denominator[valid_score]
    )
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    if multioutput == "raw_values":
        return output_scores
    if multioutput == "uniform_average":
        return paddle.mean(output_scores)
    denom_sum = paddle.sum(denominator)
    return paddle.sum(denominator / denom_sum * output_scores)


def explained_variance(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    multioutput: Literal[
        "raw_values", "uniform_average", "variance_weighted"
    ] = "uniform_average",
) -> paddle.Tensor | Sequence[paddle.Tensor]:
    """Compute explained variance.

    Args:
        preds: estimated labels
        target: ground truth labels
        multioutput: Defines aggregation in the case of multiple output scores. Can be one
            of the following strings):

            * ``'raw_values'`` returns full set of scores
            * ``'uniform_average'`` scores are uniformly averaged
            * ``'variance_weighted'`` scores are weighted by their individual variances

    Example:
        >>> from paddle.metric.functional.regression import explained_variance
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> explained_variance(preds, target)
        tensor(0.9572)

        >>> target = paddle.to_tensor([[0.5, 1], [-1, 1], [7, -6]])
        >>> preds = paddle.to_tensor([[0, 2], [-1, 2], [8, -5]])
        >>> explained_variance(preds, target, multioutput='raw_values')
        tensor([0.9677, 1.0000])

    """
    if multioutput not in ALLOWED_MULTIOUTPUT:
        raise ValueError(
            f"Invalid input to argument `multioutput`. Choose one of the following: {ALLOWED_MULTIOUTPUT}"
        )
    (
        num_obs,
        sum_error,
        sum_squared_error,
        sum_target,
        sum_squared_target,
    ) = _explained_variance_update(preds, target)
    return _explained_variance_compute(
        num_obs,
        sum_error,
        sum_squared_error,
        sum_target,
        sum_squared_target,
        multioutput,
    )
