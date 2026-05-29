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
from paddle.metric.classification.roc import (
    BinaryROC,
    MulticlassROC,
    MultilabelROC,
)
from paddle.metric.functional.classification.logauc import (
    _binary_logauc_compute,
    _reduce_logauc,
    _validate_fpr_range,
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
        "BinaryLogAUC.plot",
        "MulticlassLogAUC.plot",
        "MultilabelLogAUC.plot",
    ]


class BinaryLogAUC(BinaryROC):
    """Compute the `Log AUC`_ score for binary classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the
      positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``logauc`` (:class:`~paddle.Tensor`): A single scalar with the logauc score.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryLogAUC
        >>> preds = tensor([0.75, 0.05, 0.05, 0.05, 0.05])
        >>> target = tensor([1, 0, 0, 0, 0])
        >>> metric = BinaryLogAUC()
        >>> metric(preds, target)
        tensor(1.)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        fpr_range: tuple[float, float] = (0.001, 0.1),
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        if validate_args:
            _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range

    def compute(self) -> paddle.Tensor:
        """Computes the log AUC score."""
        fpr, tpr, _ = super().compute()
        return _binary_logauc_compute(fpr, tpr, fpr_range=self.fpr_range)

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
            >>> from paddle.metric.classification import BinaryLogAUC
            >>> metric = BinaryLogAUC()
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
            >>> from paddle.metric.classification import BinaryLogAUC
            >>> metric = BinaryLogAUC()
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


class MulticlassLogAUC(MulticlassROC):
    """Compute the `Log AUC`_ score for multiclass classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``logauc`` (:class:`~paddle.Tensor`): If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will
      be returned with logauc score per class. If `average="macro"` then a single scalar is returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
        average:
            Defines the reduction that is applied over classes. Should be one of the following:

            - ``"macro"``: Calculate score for each class and average them
            - ``"weighted"``: calculates score for each class and computes weighted average using their support
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
        >>> from paddle.metric.classification import MulticlassLogAUC
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassLogAUC(num_classes=5, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.4000)
        >>> metric = MulticlassLogAUC(num_classes=5, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([1., 1., 0., 0., 0.])

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
        fpr_range: tuple[float, float] = (0.001, 0.1),
        average: Literal["macro", "none"] | None = None,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            thresholds=thresholds,
            average=None,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        if validate_args:
            _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range
        self.average2 = average

    def compute(self) -> paddle.Tensor:
        """Computes the log AUC score."""
        fpr, tpr, _ = super().compute()
        return _reduce_logauc(
            fpr, tpr, fpr_range=self.fpr_range, average=self.average2
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
            >>> from paddle.metric.classification import MulticlassLogAUC
            >>> metric = MulticlassLogAUC(num_classes=3)
            >>> metric.update(paddle.randn(20, 3), paddle.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MulticlassLogAUC
            >>> metric = MulticlassLogAUC(num_classes=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randn(20, 3), paddle.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelLogAUC(MultilabelROC):
    """Compute the `Log AUC`_ score for multiclass classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``logauc`` (:class:`~paddle.Tensor`): If `average=None|"none"` then a 1d tensor of shape (num_labels, ) will
      be returned with logauc score per class. If `average="macro"` then a single scalar is returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        fpr_range: 2-element tuple with the lower and upper bound of the false positive rate range to compute the log
            AUC score.
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``"macro"``: Calculate the score for each label and average them
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
        >>> from paddle.metric.classification import MultilabelLogAUC
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = MultilabelLogAUC(num_labels=3, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.3945)
        >>> metric = MultilabelLogAUC(num_labels=3, average=None, thresholds=None)
        >>> metric(preds, target)
        tensor([0.5000, 0.0000, 0.6835])

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
        fpr_range: tuple[float, float] = (0.001, 0.1),
        average: Literal["macro", "none"] | None = None,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        if validate_args:
            _validate_fpr_range(fpr_range)
        self.fpr_range = fpr_range
        self.average2 = average
        super().__init__(
            num_labels=num_labels,
            thresholds=thresholds,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )

    def compute(self) -> paddle.Tensor:
        """Computes the log AUC score."""
        fpr, tpr, _ = super().compute()
        return _reduce_logauc(
            fpr, tpr, fpr_range=self.fpr_range, average=self.average2
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
            >>> from paddle.metric.classification import MultilabelLogAUC
            >>> metric = MultilabelLogAUC(num_labels=3)
            >>> metric.update(paddle.rand(20, 3), paddle.randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MultilabelLogAUC
            >>> metric = MultilabelLogAUC(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(20, 3), paddle.randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class LogAUC(_ClassificationTaskWrapper):
    """Compute the `Log AUC`_ score for multiclass classification tasks.

    The score is computed by first computing the ROC curve, which then is interpolated to the specified range of false
    positive rates (FPR) and then the log is taken of the FPR before the area under the curve (AUC) is computed. The
    score is commonly used in applications where the positive and negative are imbalanced and a low false positive rate
    is of high importance.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryLogAUC`, :class:`~paddle.metric.classification.MulticlassLogAUC` and
    :class:`~paddle.metric.classification.MultilabelLogAUC` for the specific details of each argument influence and
    examples.

    """

    def __new__(
        cls: type[LogAUC],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: int | list[float] | paddle.Tensor | None = None,
        fpr_range: tuple[float, float] | None = (0.001, 0.1),
        num_classes: int | None = None,
        num_labels: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {
                "thresholds": thresholds,
                "fpr_range": fpr_range,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryLogAUC(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassLogAUC(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelLogAUC(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")
