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
from paddle.metric.functional.classification.eer import _eer_compute
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.metric import Metric
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryEER.plot",
        "MulticlassEER.plot",
        "MultilabelEER.plot",
    ]


class BinaryEER(BinaryROC):
    """Compute Equal Error Rate (EER) for multiclass classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + \\text{FRR}}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the
      positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``b_eer`` (:class:`~paddle.Tensor`): A single scalar with the eer score.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a
    binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will
    activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
    `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        thresholds: Can be one of:

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
        >>> from paddle.metric.classification import BinaryEER
        >>> preds = tensor([0, 0.5, 0.7, 0.8])
        >>> target = tensor([0, 1, 1, 0])
        >>> metric = BinaryEER(thresholds=None)
        >>> metric(preds, target)
        tensor(0.5000)
        >>> b_eer = BinaryEER(thresholds=5)
        >>> b_eer(preds, target)
        tensor(0.7500)

    """

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        fpr, tpr, _ = super().compute()
        return _eer_compute(fpr, tpr)

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
            >>> from paddle.metric.classification import BinaryEER
            >>> metric = BinaryEER()
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
            >>> from paddle.metric.classification import BinaryEER
            >>> metric = BinaryEER()
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


class MulticlassEER(MulticlassROC):
    """Compute Equal Error Rate (EER) for multiclass classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + (1 - \\text{FRR})}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
          for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
          apply softmax per sample.
        - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
          therefore only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``mc_eer`` (:class:`~paddle.Tensor`): If `average=None` then a 1d tensor of shape (n_classes, ) will
          be returned with eer score per class. If `average="macro"|"micro"` then a single scalar will be returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a
    binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will
    activate the non-binned version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
    `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        thresholds: Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        average:
            If aggregation of curves should be applied. By default, the curves are not aggregated and a curve for
            each class is returned. If `average` is set to ``"micro"``, the metric will aggregate the curves by one hot
            encoding the targets and flattening the predictions, considering all classes jointly as a binary problem.
            If `average` is set to ``"macro"``, the metric will aggregate the curves by first interpolating the curves
            from each class at a combined set of thresholds and then average over the classwise interpolated curves.
            See `averaging curve objects`_ for more info on the different averaging methods.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Examples:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassEER
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassEER(num_classes=5, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.4667)
        >>> mc_eer = MulticlassEER(num_classes=5, average=None, thresholds=None)
        >>> mc_eer(preds, target)
        tensor([0.0000, 0.0000, 0.6667, 0.6667, 1.0000])

    """

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        fpr, tpr, _ = super().compute()
        return _eer_compute(fpr, tpr)

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
            >>> from paddle.metric.classification import MulticlassEER
            >>> metric = MulticlassEER(num_classes=3)
            >>> metric.update(paddle.randn(20, 3), paddle.randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MulticlassEER
            >>> metric = MulticlassEER(num_classes=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.randn(20, 3), paddle.randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelEER(MultilabelROC):
    """Compute Equal Error Rate (EER) for multiclass classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + (1 - \\text{FRR})}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``ml_eer`` (:class:`~paddle.Tensor`): A 1d tensor of shape (n_classes, ) will be returned with eer score per label.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        average: Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction

        thresholds: Can be one of:

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
        >>> from paddle.metric.classification import MultilabelEER
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> ml_eer = MultilabelEER(num_labels=3, thresholds=None)
        >>> ml_eer(preds, target)
        tensor([0.5000, 0.5000, 0.1667])

    """

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        fpr, tpr, _ = super().compute()
        return _eer_compute(fpr, tpr)

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
            >>> from paddle.metric.classification import MultilabelEER
            >>> metric = MultilabelEER(num_labels=3)
            >>> metric.update(paddle.rand(20, 3), paddle.randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.classification import MultilabelEER
            >>> metric = MultilabelEER(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(paddle.rand(20, 3), paddle.randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class EER(_ClassificationTaskWrapper):
    """Compute Equal Error Rate (EER) for multiclass classification task.

    .. math::
        \\text{EER} = \\frac{\\text{FAR} + (1 - \\text{FRR})}{2}, \\text{where} \\min_t abs(FAR_t-FRR_t)

    The Equal Error Rate (EER) is the point where the False Positive Rate (FPR) and True Positive Rate (TPR) are
    equal, or in practise minimized. A lower EER value signifies higher system accuracy.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryEER`, :class:`~paddle.metric.classification.MulticlassEER` and
    :class:`~paddle.metric.classification.MultilabelEER` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> preds = tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> eer = EER(task="binary")
        >>> eer(preds, target)
        tensor(0.5833)

        >>> preds = tensor([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90], [0.85, 0.05, 0.10], [0.10, 0.10, 0.80]])
        >>> target = tensor([0, 1, 1, 2, 2])
        >>> eer = EER(task="multiclass", num_classes=3)
        >>> eer(preds, target)
        tensor([0.0000, 0.4167, 0.4167])

    """

    def __new__(
        cls: type[EER],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: int | list[float] | paddle.Tensor | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["macro", "micro"] | None = None,
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
            return BinaryEER(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassEER(num_classes, average=average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelEER(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update metric state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} metric does not have a global `update` method. Use the task specific metric."
        )

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError(
            f"{self.__class__.__name__} metric does not have a global `compute` method. Use the task specific metric."
        )
