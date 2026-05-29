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
from paddle.metric.functional.classification.hinge import (
    _binary_confusion_matrix_format,
    _binary_hinge_loss_arg_validation,
    _binary_hinge_loss_tensor_validation,
    _binary_hinge_loss_update,
    _hinge_loss_compute,
    _multiclass_confusion_matrix_format,
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
    _multiclass_hinge_loss_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.enums import ClassificationTaskNoMultilabel
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BinaryHingeLoss.plot", "MulticlassHingeLoss.plot"]


class BinaryHingeLoss(Metric):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks.

    .. math::
        \\text{Hinge loss} = \\max(0, 1 - y \\times \\hat{y})

    Where :math:`y \\in {-1, 1}` is the target, and :math:`\\hat{y} \\in \\mathbb{R}` is the prediction.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bhl`` (:class:`~paddle.Tensor`): A tensor containing the hinge loss.

    Args:
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle.metric.classification import BinaryHingeLoss
        >>> preds = paddle.to_tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = paddle.to_tensor([0, 0, 1, 1, 1])
        >>> bhl = BinaryHingeLoss()
        >>> bhl(preds, target)
        tensor(0.6900)
        >>> bhl = BinaryHingeLoss(squared=True)
        >>> bhl(preds, target)
        tensor(0.6905)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    measures: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = False,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_hinge_loss_arg_validation(squared, ignore_index)
        self.validate_args = validate_args
        self.squared = squared
        self.ignore_index = ignore_index
        self.declare(
            "measures", default=paddle.tensor(0.0), dist_reduce_fn="sum"
        )
        self.declare("total", default=paddle.tensor(0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _binary_hinge_loss_tensor_validation(
                preds, target, self.ignore_index
            )
        preds, target = _binary_confusion_matrix_format(
            preds,
            target,
            threshold=0.0,
            ignore_index=self.ignore_index,
            convert_to_labels=False,
        )
        measures, total = _binary_hinge_loss_update(preds, target, self.squared)
        self.measures += measures
        self.total += total

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _hinge_loss_compute(self.measures, self.total)

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

            >>> # Example plotting a single value
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import BinaryHingeLoss
            >>> metric = BinaryHingeLoss()
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import BinaryHingeLoss
            >>> metric = BinaryHingeLoss()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassHingeLoss(Metric):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \\text{Hinge loss} = \\max\\left(0, 1 - \\hat{y}_y + \\max_{i \\ne y} (\\hat{y}_i)\\right)

    Where :math:`y \\in {0, ..., \\mathrm{C}}` is the target class (where :math:`\\mathrm{C}` is the number of classes),
    and :math:`\\hat{y} \\in \\mathbb{R}^\\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mchl`` (:class:`~paddle.Tensor`): A tensor containing the multi-class hinge loss.

    Args:
        num_classes: Integer specifying the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle.metric.classification import MulticlassHingeLoss
        >>> preds = paddle.to_tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
        >>> target = paddle.to_tensor([0, 1, 2, 0])
        >>> mchl = MulticlassHingeLoss(num_classes=3)
        >>> mchl(preds, target)
        tensor(0.9125)
        >>> mchl = MulticlassHingeLoss(num_classes=3, squared=True)
        >>> mchl(preds, target)
        tensor(1.1131)
        >>> mchl = MulticlassHingeLoss(num_classes=3, multiclass_mode='one-vs-all')
        >>> mchl(preds, target)
        tensor([0.8750, 1.1250, 1.1000])

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"
    measures: Tensor
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        squared: bool = False,
        multiclass_mode: Literal[
            "crammer-singer", "one-vs-all"
        ] = "crammer-singer",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_hinge_loss_arg_validation(
                num_classes, squared, multiclass_mode, ignore_index
            )
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.squared = squared
        self.multiclass_mode = multiclass_mode
        self.ignore_index = ignore_index
        self.declare(
            "measures",
            default=paddle.tensor(0.0)
            if self.multiclass_mode == "crammer-singer"
            else paddle.zeros(num_classes),
            dist_reduce_fn="sum",
        )
        self.declare("total", default=paddle.tensor(0), dist_reduce_fn="sum")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _multiclass_hinge_loss_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target = _multiclass_confusion_matrix_format(
            preds, target, self.ignore_index, convert_to_labels=False
        )
        measures, total = _multiclass_hinge_loss_update(
            preds, target, self.squared, self.multiclass_mode
        )
        self.measures += measures
        self.total += total

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        return _hinge_loss_compute(self.measures, self.total)

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
            >>> from paddle import randint, randn
            >>> from paddle.metric.classification import MulticlassHingeLoss
            >>> metric = MulticlassHingeLoss(num_classes=3)
            >>> metric.update(randn(20, 3), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting a multiple values per class
            >>> from paddle import randint, randn
            >>> from paddle.metric.classification import MulticlassHingeLoss
            >>> metric = MulticlassHingeLoss(num_classes=3)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randn(20, 3), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class HingeLoss(_ClassificationTaskWrapper):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryHingeLoss` and :class:`~paddle.metric.classification.MulticlassHingeLoss`
    for the specific details of each argument influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([0, 1, 1])
        >>> preds = tensor([0.5, 0.7, 0.1])
        >>> hinge = HingeLoss(task="binary")
        >>> hinge(preds, target)
        tensor(0.9000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss(task="multiclass", num_classes=3)
        >>> hinge(preds, target)
        tensor(1.5551)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss(task="multiclass", num_classes=3, multiclass_mode="one-vs-all")
        >>> hinge(preds, target)
        tensor([1.3743, 1.1945, 1.2359])

    """

    def __new__(
        cls: type[HingeLoss],
        task: Literal["binary", "multiclass"],
        num_classes: int | None = None,
        squared: bool = False,
        multiclass_mode: Literal["crammer-singer", "one-vs-all"]
        | None = "crammer-singer",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {"ignore_index": ignore_index, "validate_args": validate_args}
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryHingeLoss(squared, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            if multiclass_mode not in ("crammer-singer", "one-vs-all"):
                raise ValueError(
                    f"`multiclass_mode` is expected to be one of 'crammer-singer' or 'one-vs-all' but `{multiclass_mode}` was passed."
                )
            return MulticlassHingeLoss(
                num_classes, squared, multiclass_mode, **kwargs
            )
        raise ValueError(f"Unsupported task `{task}`")
