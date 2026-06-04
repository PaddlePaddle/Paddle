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

from paddle.metric.classification.base import _ClassificationTaskWrapper
from paddle.metric.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    MultilabelConfusionMatrix,
)
from paddle.metric.functional.classification.matthews_corrcoef import (
    _matthews_corrcoef_reduce,
)
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.metric import Metric
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryMatthewsCorrCoef.plot",
        "MulticlassMatthewsCorrCoef.plot",
        "MultilabelMatthewsCorrCoef.plot",
    ]


class BinaryMatthewsCorrCoef(BinaryConfusionMatrix):
    """Calculate `Matthews correlation coefficient`_ for binary tasks.

    This metric measures the general correlation or quality of a classification.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A int tensor or float tensor of shape ``(N, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bmcc`` (:class:`~paddle.Tensor`): A tensor containing the Binary Matthews Correlation Coefficient.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryMatthewsCorrCoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> metric = BinaryMatthewsCorrCoef()
        >>> metric(preds, target)
        tensor(0.5774)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryMatthewsCorrCoef
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0.35, 0.85, 0.48, 0.01])
        >>> metric = BinaryMatthewsCorrCoef()
        >>> metric(preds, target)
        tensor(0.5774)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold,
            ignore_index,
            normalize=None,
            validate_args=validate_args,
            **kwargs,
        )

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _matthews_corrcoef_reduce(self.confmat)

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
            >>> from paddle.metric.classification import BinaryMatthewsCorrCoef
            >>> metric = BinaryMatthewsCorrCoef()
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryMatthewsCorrCoef
            >>> metric = BinaryMatthewsCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassMatthewsCorrCoef(MulticlassConfusionMatrix):
    """Calculate `Matthews correlation coefficient`_ for multiclass tasks.

    This metric measures the general correlation or quality of a classification.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcmcc`` (:class:`~paddle.Tensor`): A tensor containing the Multi-class Matthews Correlation Coefficient.

    Args:
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassMatthewsCorrCoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7000)

    Example (pred is float tensor):
        >>> from paddle.metric.classification import MulticlassMatthewsCorrCoef
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7000)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes,
            ignore_index,
            normalize=None,
            validate_args=validate_args,
            **kwargs,
        )

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _matthews_corrcoef_reduce(self.confmat)

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

            >>> from paddle import randint
            >>> # Example plotting a single value per class
            >>> from paddle.metric.classification import MulticlassMatthewsCorrCoef
            >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassMatthewsCorrCoef
            >>> metric = MulticlassMatthewsCorrCoef(num_classes=3)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelMatthewsCorrCoef(MultilabelConfusionMatrix):
    """Calculate `Matthews correlation coefficient`_ for multilabel tasks.

    This metric measures the general correlation or quality of a classification.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, C, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlmcc`` (:class:`~paddle.Tensor`): A tensor containing the Multi-label Matthews Correlation Coefficient.

    Args:
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelMatthewsCorrCoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelMatthewsCorrCoef
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)

    """

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
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels,
            threshold,
            ignore_index,
            normalize=None,
            validate_args=validate_args,
            **kwargs,
        )

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _matthews_corrcoef_reduce(self.confmat)

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
            >>> from paddle.metric.classification import MultilabelMatthewsCorrCoef
            >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelMatthewsCorrCoef
            >>> metric = MultilabelMatthewsCorrCoef(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MatthewsCorrCoef(_ClassificationTaskWrapper):
    """Calculate `Matthews correlation coefficient`_ .

    This metric measures the general correlation or quality of a classification.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryMatthewsCorrCoef`,
    :class:`~paddle.metric.classification.MulticlassMatthewsCorrCoef` and
    :class:`~paddle.metric.classification.MultilabelMatthewsCorrCoef` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> matthews_corrcoef = MatthewsCorrCoef(task='binary')
        >>> matthews_corrcoef(preds, target)
        tensor(0.5774)

    """

    def __new__(
        cls: type[MatthewsCorrCoef],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {"ignore_index": ignore_index, "validate_args": validate_args}
        )
        if task == ClassificationTask.BINARY:
            return BinaryMatthewsCorrCoef(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassMatthewsCorrCoef(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelMatthewsCorrCoef(num_labels, threshold, **kwargs)
        raise ValueError(f"Not handled value: {task}")
