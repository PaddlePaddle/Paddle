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

import warnings
from typing import TYPE_CHECKING, Any

from typing_extensions import Literal

from paddle.metric.classification.base import _ClassificationTaskWrapper
from paddle.metric.classification.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from paddle.metric.functional.classification.accuracy import _accuracy_reduce
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.metric import Metric
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.framework import in_dynamic_mode, in_pir_mode

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryAccuracy.plot",
        "MulticlassAccuracy.plot",
        "MultilabelAccuracy.plot",
    ]


class BinaryAccuracy(BinaryStatScores):
    """Compute `Accuracy`_ for binary tasks.

    .. math::
        \\text{Accuracy} = \\frac{1}{N}\\sum_i^N 1(y_i = \\hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating
          point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
          per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
        - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``acc`` (:class:`~paddle.Tensor`): If ``multidim_average`` is set to ``global``, metric returns a scalar value.
          If ``multidim_average`` is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar
          value per sample.

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        threshold: Threshold for transforming probability to binary {0,1} predictions
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
        >>> from paddle.metric.classification import BinaryAccuracy
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryAccuracy()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryAccuracy
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryAccuracy()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from paddle.metric.classification import BinaryAccuracy
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryAccuracy(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.3333, 0.1667])

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> paddle.Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
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
            >>> from paddle.metric.classification import BinaryAccuracy
            >>> metric = BinaryAccuracy()
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryAccuracy
            >>> metric = BinaryAccuracy()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassAccuracy(MulticlassStatScores):
    """Compute `Accuracy`_ for multiclass tasks.

    .. math::
        \\text{Accuracy} = \\frac{1}{N}\\sum_i^N 1(y_i = \\hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor
          of shape ``(N, C, ..)``. If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension
          to automatically convert probabilities/logits into an int tensor.
        - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``mca`` (:class:`~paddle.Tensor`): A tensor with the accuracy score whose returned shape depends on the
          ``average`` and ``multidim_average`` arguments:

            - If ``multidim_average`` is set to ``global``:

              - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
              - If ``average=None/'none'``, the shape will be ``(C,)``

            - If ``multidim_average`` is set to ``samplewise``:

              - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
              - If ``average=None/'none'``, the shape will be ``(N, C)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
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
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassAccuracy(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> mca = MulticlassAccuracy(num_classes=3, average=None)
        >>> mca(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassAccuracy(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8333)
        >>> mca = MulticlassAccuracy(num_classes=3, average=None)
        >>> mca(preds, target)
        tensor([0.5000, 1.0000, 1.0000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassAccuracy(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5000, 0.2778])
        >>> mca = MulticlassAccuracy(num_classes=3, multidim_average='samplewise', average=None)
        >>> mca(preds, target)
        tensor([[1.0000, 0.0000, 0.5000],
                [0.0000, 0.3333, 0.5000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int | None = None,
        top_k: int = 1,
        average: str | None = "macro",
        multidim_average: str = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        self._use_cpp_op = (
            self.average == "micro"
            and self.top_k == 1
            and self.multidim_average == "global"
            and self.ignore_index is None
        )
        if self._use_cpp_op:
            self.declare("_correct", default=paddle.zeros([], dtype="int32"))
            self.declare("_total", default=paddle.zeros([], dtype="int32"))
        self._cpp_op_active: bool | None = None  # None = not yet decided

    @staticmethod
    def _eligible_for_cpp_op(
        preds: paddle.Tensor, target: paddle.Tensor
    ) -> bool:
        """Check if inputs satisfy C++ accuracy op constraints."""
        return (
            preds.is_floating_point()
            and preds.ndim == 2
            and preds.dtype in (paddle.float32, paddle.float64)
            and (
                target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1)
            )
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        use_cpp = self._use_cpp_op and self._eligible_for_cpp_op(preds, target)

        # Lock the path on first update; prevent mixing C++ and Python state
        if self._cpp_op_active is None:
            self._cpp_op_active = use_cpp
        elif use_cpp != self._cpp_op_active:
            # Path mismatch — skip this batch to avoid corrupting state
            warnings.warn(
                f"{self.__class__.__name__}: input format changed between "
                "batches (float 2D vs other). Skipping this update to "
                "preserve metric state consistency."
            )
            return

        if self._cpp_op_active:
            if target.dtype == paddle.int32:
                target = paddle.cast(target, paddle.int64)
            if target.ndim == 1:
                target = target.reshape([-1, 1])
            topk_out, topk_indices = paddle.topk(preds, k=self.top_k)
            if in_dynamic_mode():
                _acc, batch_correct, batch_total = _legacy_C_ops.accuracy(
                    topk_out,
                    topk_indices,
                    target,
                    paddle.zeros([1], dtype="int32"),
                    paddle.zeros([1], dtype="int32"),
                )
                self._correct = self._correct + batch_correct
                self._total = self._total + batch_total
            elif in_pir_mode():
                _acc, batch_correct, batch_total = _C_ops.accuracy(
                    topk_out, topk_indices, target
                )
                self._correct = self._correct + batch_correct
                self._total = self._total + batch_total
            return
        return super().update(preds, target)

    def compute(self) -> paddle.Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        if self._cpp_op_active:
            correct = self._correct.astype("float32")
            total = self._total.astype("float32")
            return paddle.where(
                total > 0, correct / total, paddle.zeros_like(total)
            )
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp,
            fp,
            tn,
            fn,
            average=self.average,
            multidim_average=self.multidim_average,
            top_k=self.top_k,
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

            >>> from paddle import randint
            >>> # Example plotting a single value per class
            >>> from paddle.metric.classification import MulticlassAccuracy
            >>> metric = MulticlassAccuracy(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassAccuracy
            >>> metric = MulticlassAccuracy(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelAccuracy(MultilabelStatScores):
    """Compute `Accuracy`_ for multilabel tasks.

    .. math::
        \\text{Accuracy} = \\frac{1}{N}\\sum_i^N 1(y_i = \\hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, C, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mla`` (:class:`~paddle.Tensor`): A tensor with the accuracy score whose returned shape depends on the
      ``average`` and ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

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
        >>> from paddle.metric.classification import MultilabelAccuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelAccuracy(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> mla = MultilabelAccuracy(num_labels=3, average=None)
        >>> mla(preds, target)
        tensor([1.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelAccuracy
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelAccuracy(num_labels=3)
        >>> metric(preds, target)
        tensor(0.6667)
        >>> mla = MultilabelAccuracy(num_labels=3, average=None)
        >>> mla(preds, target)
        tensor([1.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MultilabelAccuracy
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
        >>> mla = MultilabelAccuracy(num_labels=3, multidim_average='samplewise')
        >>> mla(preds, target)
        tensor([0.3333, 0.1667])
        >>> mla = MultilabelAccuracy(num_labels=3, multidim_average='samplewise', average=None)
        >>> mla(preds, target)
        tensor([[0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def compute(self) -> paddle.Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _accuracy_reduce(
            tp,
            fp,
            tn,
            fn,
            average=self.average,
            multidim_average=self.multidim_average,
            multilabel=True,
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

            >>> from paddle import rand, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import MultilabelAccuracy
            >>> metric = MultilabelAccuracy(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelAccuracy
            >>> metric = MultilabelAccuracy(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class Accuracy(_ClassificationTaskWrapper):
    """Compute `Accuracy`_.

    .. math::
        \\text{Accuracy} = \\frac{1}{N}\\sum_i^N 1(y_i = \\hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\\hat{y}` is a tensor of predictions.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryAccuracy`, :class:`~paddle.metric.classification.MulticlassAccuracy` and
    :class:`~paddle.metric.classification.MultilabelAccuracy` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([0, 1, 2, 3])
        >>> preds = tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy(task="multiclass", num_classes=4)
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy = Accuracy(task="multiclass", num_classes=3, top_k=2)
        >>> accuracy(preds, target)
        tensor(0.6667)

    """

    def __new__(
        cls: type[Accuracy],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        multidim_average: Literal["global", "samplewise"] = "global",
        top_k: int | None = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {
                "multidim_average": multidim_average,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryAccuracy(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"Optional arg `num_classes` must be type `int` when task is {task}. Got {type(num_classes)}"
                )
            if not isinstance(top_k, int):
                raise ValueError(
                    f"Optional arg `top_k` must be type `int` when task is {task}. Got {type(top_k)}"
                )
            return MulticlassAccuracy(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"Optional arg `num_labels` must be type `int` when task is {task}. Got {type(num_labels)}"
                )
            return MultilabelAccuracy(num_labels, threshold, average, **kwargs)
        raise ValueError(f"Not handled value: {task}")
