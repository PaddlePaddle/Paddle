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

from typing import Any

import paddle
from paddle import Tensor
from paddle.metric.functional.regression.csi import (
    _critical_success_index_compute,
    _critical_success_index_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils import dim_zero_cat


class CriticalSuccessIndex(Metric):
    """Calculate critical success index (CSI).

    Critical success index (also known as the threat score) is a statistic used weather forecasting that measures
    forecast performance over inputs binarized at a specified threshold. It is defined as:

    .. math:: \\text{CSI} = \\frac{\\text{TP}}{\\text{TP}+\\text{FN}+\\text{FP}}

    Where :math:`\\text{TP}`, :math:`\\text{FN}` and :math:`\\text{FP}` represent the number of true positives, false
    negatives and false positives respectively after binarizing the input tensors.

    Args:
        threshold: Values above or equal to threshold are replaced with 1, below by 0
        keep_sequence_dim: Index of the sequence dimension if the inputs are sequences of images. If specified,
            the score will be calculated separately for each image in the sequence. If ``None``, the score will be
            calculated across all dimensions.

    Example:
        >>> import paddle
        >>> from paddle.metric.regression import CriticalSuccessIndex
        >>> x = paddle.Tensor([[0.2, 0.7], [0.9, 0.3]])
        >>> y = paddle.Tensor([[0.4, 0.2], [0.8, 0.6]])
        >>> csi = CriticalSuccessIndex(0.5)
        >>> csi(x, y)
        tensor(0.3333)

    Example:
        >>> import paddle
        >>> from paddle.metric.regression import CriticalSuccessIndex
        >>> x = paddle.Tensor([[[0.2, 0.7], [0.9, 0.3]], [[0.2, 0.7], [0.9, 0.3]]])
        >>> y = paddle.Tensor([[[0.4, 0.2], [0.8, 0.6]], [[0.4, 0.2], [0.8, 0.6]]])
        >>> csi = CriticalSuccessIndex(0.5, keep_sequence_dim=0)
        >>> csi(x, y)
        tensor([0.3333, 0.3333])

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    hits: Tensor
    misses: Tensor
    false_alarms: Tensor
    hits_list: list[paddle.Tensor]
    misses_list: list[paddle.Tensor]
    false_alarms_list: list[paddle.Tensor]

    def __init__(
        self,
        threshold: float,
        keep_sequence_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = float(threshold)
        if keep_sequence_dim and (
            not isinstance(keep_sequence_dim, int) or keep_sequence_dim < 0
        ):
            raise ValueError(
                f"Expected keep_sequence_dim to be a non-negative integer but got {keep_sequence_dim}"
            )
        self.keep_sequence_dim = keep_sequence_dim
        if keep_sequence_dim is None:
            self.declare("hits", default=paddle.tensor(0), dist_reduce_fn="sum")
            self.declare(
                "misses", default=paddle.tensor(0), dist_reduce_fn="sum"
            )
            self.declare(
                "false_alarms", default=paddle.tensor(0), dist_reduce_fn="sum"
            )
        else:
            self.declare("hits_list", default=[], dist_reduce_fn="cat")
            self.declare("misses_list", default=[], dist_reduce_fn="cat")
            self.declare("false_alarms_list", default=[], dist_reduce_fn="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        hits, misses, false_alarms = _critical_success_index_update(
            preds, target, self.threshold, self.keep_sequence_dim
        )
        if self.keep_sequence_dim is None:
            self.hits += hits
            self.misses += misses
            self.false_alarms += false_alarms
        else:
            self.hits_list.append(hits)
            self.misses_list.append(misses)
            self.false_alarms_list.append(false_alarms)

    def compute(self) -> paddle.Tensor:
        """Compute critical success index over state."""
        if self.keep_sequence_dim is None:
            hits = self.hits
            misses = self.misses
            false_alarms = self.false_alarms
        else:
            hits = dim_zero_cat(self.hits_list)
            misses = dim_zero_cat(self.misses_list)
            false_alarms = dim_zero_cat(self.false_alarms_list)
        return _critical_success_index_compute(hits, misses, false_alarms)
