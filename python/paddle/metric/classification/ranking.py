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

from typing import TYPE_CHECKING, Any

import paddle
from paddle.metric.functional.classification.ranking import (
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_coverage_error_update,
    _multilabel_ranking_average_precision_update,
    _multilabel_ranking_loss_update,
    _multilabel_ranking_tensor_validation,
    _ranking_reduce,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "MultilabelCoverageError.plot",
        "MultilabelRankingAveragePrecision.plot",
        "MultilabelRankingLoss.plot",
    ]


class MultilabelCoverageError(Metric):
    """Compute `Multilabel coverage error`_.

    The score measure how far we need to go through the ranked scores to cover all true labels. The best value is equal
    to the average number of labels in the target tensor per sample.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlce`` (:class:`~paddle.Tensor`): A tensor containing the multilabel coverage error.

    Args:
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import rand, randint
        >>> from paddle.metric.classification import MultilabelCoverageError
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> mlce = MultilabelCoverageError(num_labels=5)
        >>> mlce(preds, target)
        tensor(3.9000)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold=0.0, ignore_index=ignore_index
            )
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.declare("measure", paddle.tensor(0.0), dist_reduce_fn="sum")
        self.declare("total", paddle.tensor(0.0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_ranking_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target = _multilabel_confusion_matrix_format(
            preds,
            target,
            self.num_labels,
            threshold=0.0,
            ignore_index=self.ignore_index,
            should_threshold=False,
        )
        measure, num_elements = _multilabel_coverage_error_update(preds, target)
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        self.measure += measure
        self.total += num_elements

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        return _ranking_reduce(self.measure, int(self.total.item()))

    def plot(
        self,
        val: paddle.Tensor | Sequence[paddle.Tensor] | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import MultilabelCoverageError
            >>> metric = MultilabelCoverageError(num_labels=3)
            >>> metric.update(rand(20, 3), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelCoverageError
            >>> metric = MultilabelCoverageError(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(20, 3), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelRankingAveragePrecision(Metric):
    """Compute label ranking average precision score for multilabel data [1].

    The score is the average over each ground truth label assigned to each sample of the ratio of true vs. total labels
    with lower score. Best score is 1.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlrap`` (:class:`~paddle.Tensor`): A tensor containing the multilabel ranking average precision.

    Args:
        num_labels: Integer specifying the number of labels
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle import rand, randint
        >>> from paddle.metric.classification import MultilabelRankingAveragePrecision
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> mlrap = MultilabelRankingAveragePrecision(num_labels=5)
        >>> mlrap(preds, target)
        tensor(0.7744)

    """

    higher_is_better: bool = True
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold=0.0, ignore_index=ignore_index
            )
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.declare("measure", paddle.tensor(0.0), dist_reduce_fn="sum")
        self.declare("total", paddle.tensor(0.0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_ranking_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target = _multilabel_confusion_matrix_format(
            preds,
            target,
            self.num_labels,
            threshold=0.0,
            ignore_index=self.ignore_index,
            should_threshold=False,
        )
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        measure, num_elements = _multilabel_ranking_average_precision_update(
            preds, target
        )
        self.measure += measure
        self.total += num_elements

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        return _ranking_reduce(self.measure, int(self.total.item()))

    def plot(
        self,
        val: paddle.Tensor | Sequence[paddle.Tensor] | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import MultilabelRankingAveragePrecision
            >>> metric = MultilabelRankingAveragePrecision(num_labels=3)
            >>> metric.update(rand(20, 3), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelRankingAveragePrecision
            >>> metric = MultilabelRankingAveragePrecision(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(20, 3), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelRankingLoss(Metric):
    """Compute the label ranking loss for multilabel data [1].

    The score is corresponds to the average number of label pairs that are incorrectly ordered given some predictions
    weighted by the size of the label set and the number of labels not in the label set. The best score is 0.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlrl`` (:class:`~paddle.Tensor`): A tensor containing the multilabel ranking loss.

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
        >>> from paddle.metric.classification import MultilabelRankingLoss
        >>> preds = rand(10, 5)
        >>> target = randint(2, (10, 5))
        >>> mlrl = MultilabelRankingLoss(num_labels=5)
        >>> mlrl(preds, target)
        tensor(0.4167)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold=0.0, ignore_index=ignore_index
            )
        self.validate_args = validate_args
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.declare("measure", paddle.tensor(0.0), dist_reduce_fn="sum")
        self.declare("total", paddle.tensor(0.0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_ranking_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target = _multilabel_confusion_matrix_format(
            preds,
            target,
            self.num_labels,
            threshold=0.0,
            ignore_index=self.ignore_index,
            should_threshold=False,
        )
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        measure, num_elements = _multilabel_ranking_loss_update(preds, target)
        self.measure += measure
        self.total += num_elements

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        if not isinstance(self.measure, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.measure' to be of type Tensor, but got {type(self.measure)}."
            )
        if not isinstance(self.total, paddle.Tensor):
            raise TypeError(
                f"Expected 'self.total' to be of type Tensor, but got {type(self.total)}."
            )
        return _ranking_reduce(self.measure, int(self.total.item()))

    def plot(
        self,
        val: paddle.Tensor | Sequence[paddle.Tensor] | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import MultilabelRankingLoss
            >>> metric = MultilabelRankingLoss(num_labels=3)
            >>> metric.update(rand(20, 3), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelRankingLoss
            >>> metric = MultilabelRankingLoss(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(20, 3), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
