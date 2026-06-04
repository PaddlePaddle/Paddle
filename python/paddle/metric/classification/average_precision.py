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
from paddle.metric.functional.classification.average_precision import (
    _binary_average_precision_compute,
    _multiclass_average_precision_arg_validation,
    _multiclass_average_precision_compute,
    _multilabel_average_precision_arg_validation,
    _multilabel_average_precision_compute,
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
        "BinaryAveragePrecision.plot",
        "MulticlassAveragePrecision.plot",
        "MultilabelAveragePrecision.plot",
    ]


class BinaryAveragePrecision(BinaryPrecisionRecallCurve):
    """Compute the average precision (AP) score for binary tasks.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the
      positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bap`` (:class:`~paddle.Tensor`): A single scalar with the average precision score

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryAveragePrecision
        >>> preds = tensor([0, 0.5, 0.7, 0.8])
        >>> target = tensor([0, 1, 1, 0])
        >>> metric = BinaryAveragePrecision(thresholds=None)
        >>> metric(preds, target)
        tensor(0.5833)
        >>> bap = BinaryAveragePrecision(thresholds=5)
        >>> bap(preds, target)
        tensor(0.6667)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _binary_average_precision_compute(state, self.thresholds)

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

            >>> # Example plotting a single
            >>> import paddle
            >>> from paddle.metric.classification import BinaryAveragePrecision
            >>> metric = BinaryAveragePrecision()
            >>> metric.update(
            ...     paddle.rand(
            ...         20,
            ...     ),
            ...     paddle.randint(2, (20,)),
            ... )
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import BinaryAveragePrecision
            >>> metric = BinaryAveragePrecision()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(
            ...         metric(
            ...             paddle.rand(
            ...                 20,
            ...             ),
            ...             paddle.randint(2, (20,)),
            ...         )
            ...     )
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassAveragePrecision(MulticlassPrecisionRecallCurve):
    """Compute the average precision (AP) score for multiclass tasks.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported by
    this metric. By default the reported metric is then the average over all classes, but this behavior can be changed
    by setting the ``average`` argument.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcap`` (:class:`~paddle.Tensor`): If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be
      returned with AP score per class. If `average="macro"|"weighted"` then a single scalar is returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``macro``: Calculate score for each class and average them
            - ``weighted``: calculates score for each class and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each class and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassAveragePrecision
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassAveragePrecision(num_classes=5, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.6250)
        >>> mcap = MulticlassAveragePrecision(num_classes=5, average=None, thresholds=None)
        >>> mcap(preds, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500,    nan])
        >>> mcap = MulticlassAveragePrecision(num_classes=5, average="macro", thresholds=5)
        >>> mcap(preds, target)
        tensor(0.5000)
        >>> mcap = MulticlassAveragePrecision(num_classes=5, average=None, thresholds=5)
        >>> mcap(preds, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500, -0.0000])

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
        average: Literal["macro", "weighted", "none"] | None = "macro",
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
            _multiclass_average_precision_arg_validation(
                num_classes, average, thresholds, ignore_index
            )
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multiclass_average_precision_compute(
            state, self.num_classes, self.average, self.thresholds
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
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single
            >>> import paddle
            >>> from paddle.metric.classification import MulticlassAveragePrecision
            >>> metric = MulticlassAveragePrecision(num_classes=3)
            >>> metric.update(paddle.randn(20, 3), paddle.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MulticlassAveragePrecision
            >>> metric = MulticlassAveragePrecision(num_classes=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randn(20, 3), paddle.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelAveragePrecision(MultilabelPrecisionRecallCurve):
    """Compute the average precision (AP) score for multilabel tasks.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlap`` (:class:`~paddle.Tensor`): If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be
      returned with AP score per class. If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned
    version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate
    the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
    `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelAveragePrecision
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.7500)
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=None)
        >>> mlap(preds, target)
        tensor([0.7500, 0.5833, 0.9167])
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=5)
        >>> mlap(preds, target)
        tensor(0.7778)
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=5)
        >>> mlap(preds, target)
        tensor([0.7500, 0.6667, 0.9167])

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
        average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
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
            _multilabel_average_precision_arg_validation(
                num_labels, average, thresholds, ignore_index
            )
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multilabel_average_precision_compute(
            state,
            self.num_labels,
            self.average,
            self.thresholds,
            self.ignore_index,
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
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single
            >>> import paddle
            >>> from paddle.metric.classification import MultilabelAveragePrecision
            >>> metric = MultilabelAveragePrecision(num_labels=3)
            >>> metric.update(paddle.rand(20, 3), paddle.randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MultilabelAveragePrecision
            >>> metric = MultilabelAveragePrecision(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(20, 3), paddle.randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class AveragePrecision(_ClassificationTaskWrapper):
    """Compute the average precision (AP) score.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddlemetrics.classification.BinaryAveragePrecision`,
    :class:`~paddlemetrics.classification.MulticlassAveragePrecision` and
    :class:`~paddlemetrics.classification.MultilabelAveragePrecision` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> pred = tensor([0, 0.1, 0.8, 0.4])
        >>> target = tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(task="binary")
        >>> average_precision(pred, target)
        tensor(1.)

        >>> pred = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(task="multiclass", num_classes=5, average=None)
        >>> average_precision(pred, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500,    nan])

    """

    def __new__(
        cls: type[AveragePrecision],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: int | list[float] | paddle.Tensor | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["macro", "weighted", "none"] | None = "macro",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {
                "thresholds": thresholds,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryAveragePrecision(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassAveragePrecision(num_classes, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelAveragePrecision(num_labels, average, **kwargs)
        raise ValueError(f"Task {task} not supported!")
