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

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.classification.base import _ClassificationTaskWrapper
from paddle.metric.functional.classification.exact_match import (
    _exact_match_reduce,
    _multiclass_exact_match_update,
    _multilabel_exact_match_update,
)
from paddle.metric.functional.classification.stat_scores import (
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.enums import ClassificationTaskNoBinary
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "MulticlassExactMatch.plot",
        "MultilabelExactMatch.plot",
    ]


class MulticlassExactMatch(Metric):
    """Compute Exact match (also known as subset accuracy) for multiclass tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcem`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of labels
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (multidim tensors):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassExactMatch
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassExactMatch(num_classes=3, multidim_average='global')
        >>> metric(preds, target)
        tensor(0.5000)

    Example (multidim tensors):
        >>> from paddle.metric.classification import MulticlassExactMatch
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassExactMatch(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([1., 0.])

    """

    total: Tensor
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        top_k, average = 1, None
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes, top_k, average, multidim_average, ignore_index
            )
        self.num_classes = num_classes
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.declare(
            "correct",
            paddle.zeros(1, dtype=paddle.long)
            if self.multidim_average == "global"
            else [],
            dist_reduce_fn="sum"
            if self.multidim_average == "global"
            else "cat",
        )
        self.declare(
            "total",
            paddle.zeros(1, dtype=paddle.long),
            dist_reduce_fn="sum"
            if self.multidim_average == "global"
            else "mean",
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )
        preds, target = _multiclass_stat_scores_format(preds, target, 1)
        correct, total = _multiclass_exact_match_update(
            preds, target, self.multidim_average, self.ignore_index
        )
        if self.multidim_average == "samplewise":
            if not isinstance(self.correct, list):
                raise TypeError(
                    "Expected `self.correct` to be a list in samplewise mode."
                )
            self.correct.append(correct)
            if not isinstance(self.total, paddle.Tensor):
                raise TypeError(
                    "Expected `self.total` to be a Tensor in samplewise mode."
                )
            self.total = total
        else:
            if not isinstance(self.correct, paddle.Tensor):
                raise TypeError(
                    "Expected `self.correct` to be a tensor in global mode."
                )
            self.correct += correct
            if not isinstance(self.total, paddle.Tensor):
                raise TypeError(
                    "Expected `self.total` to be a Tensor in samplewise mode."
                )
            self.total += total

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        correct = (
            dim_zero_cat(self.correct)
            if isinstance(self.correct, list)
            else self.correct
        )
        if not isinstance(correct, paddle.Tensor) or not isinstance(
            self.total, paddle.Tensor
        ):
            raise TypeError(
                "Expected `correct` and `total` to be tensors after processing."
            )
        return _exact_match_reduce(correct, self.total)

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

            >>> # Example plotting a single value per class
            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassExactMatch
            >>> metric = MulticlassExactMatch(num_classes=3)
            >>> metric.update(randint(3, (20, 5)), randint(3, (20, 5)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassExactMatch
            >>> metric = MulticlassExactMatch(num_classes=3)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20, 5)), randint(3, (20, 5))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelExactMatch(Metric):
    """Compute Exact match (also known as subset accuracy) for multilabel tasks.

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor or float tensor of shape ``(N, C, ..)``. If preds is a
      floating point tensor with values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlem`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelExactMatch
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelExactMatch(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelExactMatch
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelExactMatch(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    Example (multidim tensors):
        >>> from paddle.metric.classification import MultilabelExactMatch
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelExactMatch(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0., 0.])

    """

    total: Tensor
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average=None,
                multidim_average=multidim_average,
                ignore_index=ignore_index,
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.declare(
            "correct",
            paddle.zeros(1, dtype=paddle.long)
            if self.multidim_average == "global"
            else [],
            dist_reduce_fn="sum"
            if self.multidim_average == "global"
            else "cat",
        )
        self.declare(
            "total",
            paddle.zeros(1, dtype=paddle.long),
            dist_reduce_fn="sum"
            if self.multidim_average == "global"
            else "mean",
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds,
                target,
                self.num_labels,
                self.multidim_average,
                self.ignore_index,
            )
        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        correct, total = _multilabel_exact_match_update(
            preds=preds,
            target=target,
            num_labels=self.num_labels,
            multidim_average=self.multidim_average,
            ignore_index=self.ignore_index,
        )
        if self.multidim_average == "samplewise":
            if not isinstance(self.correct, list):
                raise TypeError(
                    "Expected `self.correct` to be a list in samplewise mode."
                )
            self.correct.append(correct)
            if not isinstance(self.total, paddle.Tensor):
                raise TypeError(
                    "Expected `self.total` to be a Tensor in samplewise mode."
                )
            self.total = total
        else:
            if not isinstance(self.correct, paddle.Tensor):
                raise TypeError(
                    "Expected `self.correct` to be a tensor in global mode."
                )
            self.correct += correct
            if not isinstance(self.total, paddle.Tensor):
                raise TypeError(
                    "Expected `self.total` to be a Tensor in samplewise mode."
                )
            self.total += total

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        correct = (
            dim_zero_cat(self.correct)
            if isinstance(self.correct, list)
            else self.correct
        )
        if not isinstance(correct, paddle.Tensor) or not isinstance(
            self.total, paddle.Tensor
        ):
            raise TypeError(
                "Expected `correct` and `total` to be tensors after processing."
            )
        return _exact_match_reduce(correct, self.total)

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
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelExactMatch
            >>> metric = MultilabelExactMatch(num_labels=3)
            >>> metric.update(randint(2, (20, 3, 5)), randint(2, (20, 3, 5)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelExactMatch
            >>> metric = MultilabelExactMatch(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3, 5)), randint(2, (20, 3, 5))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class ExactMatch(_ClassificationTaskWrapper):
    """Compute Exact match (also known as subset accuracy).

    Exact Match is a stricter version of accuracy where all labels have to match exactly for the sample to be
    correctly classified.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.MulticlassExactMatch` and
    :class:`~paddle.metric.classification.MultilabelExactMatch` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = ExactMatch(task="multiclass", num_classes=3, multidim_average='global')
        >>> metric(preds, target)
        tensor(0.5000)

        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = ExactMatch(task="multiclass", num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([1., 0.])

    """

    def __new__(
        cls: type[ExactMatch],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoBinary.from_str(task)
        kwargs.update(
            {
                "multidim_average": multidim_average,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoBinary.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassExactMatch(num_classes, **kwargs)
        if task == ClassificationTaskNoBinary.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelExactMatch(num_labels, threshold, **kwargs)
        raise ValueError(f"Task {task} not supported!")
