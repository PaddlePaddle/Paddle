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
from paddle.metric.functional.classification.confusion_matrix import (
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
)
from paddle.metric.utils.data import _cumsum


def _rank_data(x: paddle.Tensor) -> paddle.Tensor:
    """Rank data based on values."""
    with paddle.no_grad():
        _, inverse, counts = paddle.unique(
            x, sorted=True, return_inverse=True, return_counts=True
        )
    ranks = _cumsum(counts, axis=0)
    return ranks[inverse]


def _ranking_reduce(score: paddle.Tensor, num_elements: int) -> paddle.Tensor:
    return score / num_elements


def _multilabel_ranking_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    ignore_index: int | None = None,
) -> None:
    _multilabel_confusion_matrix_tensor_validation(
        preds, target, num_labels, ignore_index
    )
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected preds tensor to be floating point, but received input with dtype {preds.dtype}"
        )


def _multilabel_coverage_error_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, int]:
    """Accumulate state for coverage error."""
    offset = paddle.zeros_like(preds)
    offset[target == 0] = preds._min().abs() + 10
    preds_mod = preds + offset
    preds_min = (preds_mod.min(axis=1), preds_mod.argmin(axis=1))[0]
    coverage = (preds >= preds_min[:, None]).sum(dim=1).to(paddle.float32)
    return coverage.sum(), coverage.size


def multilabel_coverage_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute multilabel coverage error [1].

    The score measure how far we need to go through the ranked scores to cover all true labels. The best value is equal
    to the average number of labels in the target tensor per sample.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import rand, randint
        >>> from paddle.metric.functional.classification import multilabel_coverage_error
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> multilabel_coverage_error(preds, target, num_labels=5)
        tensor(3.9000)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(
            num_labels, threshold=0.0, ignore_index=ignore_index
        )
        _multilabel_ranking_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds,
        target,
        num_labels,
        threshold=0.0,
        ignore_index=ignore_index,
        should_threshold=False,
    )
    coverage, total = _multilabel_coverage_error_update(preds, target)
    return _ranking_reduce(coverage, total)


def _multilabel_ranking_average_precision_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, int]:
    """Accumulate state for label ranking average precision."""
    neg_preds = -preds
    score = paddle.tensor(0.0, device=neg_preds.place)
    num_preds, num_labels = neg_preds.shape
    for i in range(num_preds):
        relevant = target[i] == 1
        ranking = _rank_data(neg_preds[i][relevant]).float()
        if len(ranking) > 0 and len(ranking) < num_labels:
            rank = _rank_data(neg_preds[i])[relevant].float()
            score_idx = (ranking / rank).mean()
        else:
            score_idx = paddle.ones_like(score)
        score += score_idx
    return score, num_preds


def multilabel_ranking_average_precision(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute label ranking average precision score for multilabel data [1].

    The score is the average over each ground truth label assigned to each sample of the ratio of true vs. total labels
    with lower score. Best score is 1.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import rand, randint
        >>> from paddle.metric.functional.classification import multilabel_ranking_average_precision
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> multilabel_ranking_average_precision(preds, target, num_labels=5)
        tensor(0.7744)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(
            num_labels, threshold=0.0, ignore_index=ignore_index
        )
        _multilabel_ranking_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds,
        target,
        num_labels,
        threshold=0.0,
        ignore_index=ignore_index,
        should_threshold=False,
    )
    score, num_elements = _multilabel_ranking_average_precision_update(
        preds, target
    )
    return _ranking_reduce(score, num_elements)


def _multilabel_ranking_loss_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, int]:
    """Accumulate state for label ranking loss.

    Args:
        preds: tensor with predictions
        target: tensor with ground truth labels
        sample_weight: optional tensor with weight for each sample

    """
    num_preds, num_labels = preds.shape
    relevant = target == 1
    num_relevant = relevant.sum(dim=1)
    mask = (num_relevant > 0) & (num_relevant < num_labels)
    preds = preds[mask]
    relevant = relevant[mask]
    num_relevant = num_relevant[mask]
    if len(preds) == 0:
        return paddle.tensor(0.0, device=preds.place), 1
    inverse = preds.argsort(dim=1).argsort(dim=1)
    per_label_loss = ((num_labels - inverse) * relevant).to(paddle.float32)
    correction = 0.5 * num_relevant * (num_relevant + 1)
    denom = num_relevant * (num_labels - num_relevant)
    loss = (per_label_loss.sum(dim=1) - correction) / denom
    return loss.sum(), num_preds


def multilabel_ranking_loss(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_labels: int,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
    """Compute the label ranking loss for multilabel data [1].

    The score is corresponds to the average number of label pairs that are incorrectly ordered given some predictions
    weighted by the size of the label set and the number of labels not in the label set. The best score is 0.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import rand, randint
        >>> from paddle.metric.functional.classification import multilabel_ranking_loss
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> multilabel_ranking_loss(preds, target, num_labels=5)
        tensor(0.4167)

    References:
        [1] Tsoumakas, G., Katakis, I., & Vlahavas, I. (2010). Mining multi-label data. In Data mining and
        knowledge discovery handbook (pp. 667-685). Springer US.

    """
    if validate_args:
        _multilabel_confusion_matrix_arg_validation(
            num_labels, threshold=0.0, ignore_index=ignore_index
        )
        _multilabel_ranking_tensor_validation(
            preds, target, num_labels, ignore_index
        )
    preds, target = _multilabel_confusion_matrix_format(
        preds,
        target,
        num_labels,
        threshold=0.0,
        ignore_index=ignore_index,
        should_threshold=False,
    )
    loss, num_elements = _multilabel_ranking_loss_update(preds, target)
    return _ranking_reduce(loss, num_elements)
