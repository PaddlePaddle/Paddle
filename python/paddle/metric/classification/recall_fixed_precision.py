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
from paddle.metric.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from paddle.metric.functional.classification.recall_fixed_precision import (
    _binary_recall_at_fixed_precision_arg_validation,
    _binary_recall_at_fixed_precision_compute,
    _multiclass_recall_at_fixed_precision_arg_compute,
    _multiclass_recall_at_fixed_precision_arg_validation,
    _multilabel_recall_at_fixed_precision_arg_compute,
    _multilabel_recall_at_fixed_precision_arg_validation,
)
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.metric import Metric
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryRecallAtFixedPrecision.plot",
        "MulticlassRecallAtFixedPrecision.plot",
        "MultilabelRecallAtFixedPrecision.plot",
    ]


class BinaryRecallAtFixedPrecision(BinaryPrecisionRecallCurve):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

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

    - ``recall`` (:class:`~paddle.Tensor`): A scalar tensor with the maximum recall for the given precision level
    - ``threshold`` (:class:`~paddle.Tensor`): A scalar tensor with the corresponding threshold level

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a
       binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None``
       will activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting
       the `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory
       of size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryRecallAtFixedPrecision
        >>> preds = tensor([0, 0.5, 0.7, 0.8])
        >>> target = tensor([0, 1, 1, 0])
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.5000))
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.5000))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        min_precision: float,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            thresholds, ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _binary_recall_at_fixed_precision_arg_validation(
                min_precision, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_precision = min_precision

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1"]

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _binary_recall_at_fixed_precision_compute(
            state, self.thresholds, self.min_precision
        )

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
            >>> from paddle.metric.classification import BinaryRecallAtFixedPrecision
            >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the maximum recall value by default

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryRecallAtFixedPrecision
            >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
            >>> values = []
            >>> for _ in range(10):
            ...     # we index by 0 such that only the maximum recall value is plotted
            ...     values.append(metric(rand(10), randint(2, (10,)))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)


class MulticlassRecallAtFixedPrecision(MulticlassPrecisionRecallCurve):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported by
    this metric.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 2 tensors or 2 lists containing:

    - ``recall`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_classes, )`` with the maximum recall for the
      given precision level per class
    - ``threshold`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_classes, )`` with the corresponding threshold
      level per class

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
       non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassRecallAtFixedPrecision
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassRecallAtFixedPrecision(num_classes=5, min_precision=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, nan, nan, nan]))
        >>> mcrafp = MulticlassRecallAtFixedPrecision(num_classes=5, min_precision=0.5, thresholds=5)
        >>> mcrafp(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([0.7500, 0.7500, nan, nan, nan]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        min_precision: float,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multiclass_recall_at_fixed_precision_arg_validation(
                num_classes, min_precision, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_precision = min_precision

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1"]

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multiclass_recall_at_fixed_precision_arg_compute(
            state, self.num_classes, self.thresholds, self.min_precision
        )

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
            >>> # Example plotting a single value per class
            >>> from paddle.metric.classification import MulticlassRecallAtFixedPrecision
            >>> metric = MulticlassRecallAtFixedPrecision(num_classes=3, min_precision=0.5)
            >>> metric.update(rand(20, 3).softmax(dim=-1), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the maximum recall value by default

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassRecallAtFixedPrecision
            >>> metric = MulticlassRecallAtFixedPrecision(num_classes=3, min_precision=0.5)
            >>> values = []
            >>> for _ in range(20):
            ...     # we index by 0 such that only the maximum recall value is plotted
            ...     values.append(metric(rand(20, 3).softmax(dim=-1), randint(3, (20,)))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)


class MultilabelRecallAtFixedPrecision(MultilabelPrecisionRecallCurve):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 2 tensors or 2 lists containing:

    - ``recall`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_classes, )`` with the maximum recall for the
      given precision level per class
    - ``threshold`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_classes, )`` with the corresponding threshold
      level per class

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to ```None``` will activate
       the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
       `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~paddle.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelRecallAtFixedPrecision
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor([1., 1., 1.]), tensor([0.0500, 0.5500, 0.0500]))
        >>> mlrafp = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5, thresholds=5)
        >>> mlrafp(preds, target)
        (tensor([1., 1., 1.]), tensor([0.0000, 0.5000, 0.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        min_precision: float,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multilabel_recall_at_fixed_precision_arg_validation(
                num_labels, min_precision, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_precision = min_precision

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1"]

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multilabel_recall_at_fixed_precision_arg_compute(
            state,
            self.num_labels,
            self.thresholds,
            self.ignore_index,
            self.min_precision,
        )

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
            >>> from paddle.metric.classification import MultilabelRecallAtFixedPrecision
            >>> metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5)
            >>> metric.update(rand(20, 3), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the maximum recall value by default

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelRecallAtFixedPrecision
            >>> metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5)
            >>> values = []
            >>> for _ in range(10):
            ...     # we index by 0 such that only the maximum recall value is plotted
            ...     values.append(metric(rand(20, 3), randint(2, (20, 3)))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)


class RecallAtFixedPrecision(_ClassificationTaskWrapper):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryRecallAtFixedPrecision`,
    :class:`~paddle.metric.classification.MulticlassRecallAtFixedPrecision` and
    :class:`~paddle.metric.classification.MultilabelRecallAtFixedPrecision` for the specific details of each argument
    influence and examples.

    """

    def __new__(
        cls: type[RecallAtFixedPrecision],
        task: Literal["binary", "multiclass", "multilabel"],
        min_precision: float,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        if task == ClassificationTask.BINARY:
            return BinaryRecallAtFixedPrecision(
                min_precision, thresholds, ignore_index, validate_args, **kwargs
            )
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassRecallAtFixedPrecision(
                num_classes,
                min_precision,
                thresholds,
                ignore_index,
                validate_args,
                **kwargs,
            )
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelRecallAtFixedPrecision(
                num_labels,
                min_precision,
                thresholds,
                ignore_index,
                validate_args,
                **kwargs,
            )
        raise ValueError(f"Task {task} not supported!")
