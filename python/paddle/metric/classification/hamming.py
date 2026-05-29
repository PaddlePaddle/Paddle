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
from paddle.metric.classification.stat_scores import (
    BinaryStatScores,
    MulticlassStatScores,
    MultilabelStatScores,
)
from paddle.metric.functional.classification.hamming import (
    _hamming_distance_reduce,
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
        "BinaryHammingDistance.plot",
        "MulticlassHammingDistance.plot",
        "MultilabelHammingDistance.plot",
    ]


class BinaryHammingDistance(BinaryStatScores):
    """Compute the average `Hamming distance`_ (also known as Hamming loss) for binary tasks.

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bhd`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``, the metric returns a scalar value.
        - If ``multidim_average`` is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a
          scalar value per sample.

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
        >>> from paddle.metric.classification import BinaryHammingDistance
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryHammingDistance()
        >>> metric(preds, target)
        tensor(0.3333)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryHammingDistance
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryHammingDistance()
        >>> metric(preds, target)
        tensor(0.3333)

    Example (multidim tensors):
        >>> from paddle.metric.classification import BinaryHammingDistance
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryHammingDistance(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.6667, 0.8333])

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(
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

            >>> # Example plotting a single value
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import BinaryHammingDistance
            >>> metric = BinaryHammingDistance()
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import BinaryHammingDistance
            >>> metric = BinaryHammingDistance()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassHammingDistance(MulticlassStatScores):
    """Compute the average `Hamming distance`_ (also known as Hamming loss) for multiclass tasks.

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mchd`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

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
        >>> from paddle.metric.classification import MulticlassHammingDistance
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassHammingDistance(num_classes=3)
        >>> metric(preds, target)
        tensor(0.1667)
        >>> mchd = MulticlassHammingDistance(num_classes=3, average=None)
        >>> mchd(preds, target)
        tensor([0.5000, 0.0000, 0.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MulticlassHammingDistance
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassHammingDistance(num_classes=3)
        >>> metric(preds, target)
        tensor(0.1667)
        >>> mchd = MulticlassHammingDistance(num_classes=3, average=None)
        >>> mchd(preds, target)
        tensor([0.5000, 0.0000, 0.0000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MulticlassHammingDistance
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassHammingDistance(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5000, 0.7222])
        >>> mchd = MulticlassHammingDistance(num_classes=3, multidim_average='samplewise', average=None)
        >>> mchd(preds, target)
        tensor([[0.0000, 1.0000, 0.5000],
                [1.0000, 0.6667, 0.5000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(
            tp,
            fp,
            tn,
            fn,
            average=self.average,
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

            >>> # Example plotting a single value per class
            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassHammingDistance
            >>> metric = MulticlassHammingDistance(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting a multiple values per class
            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassHammingDistance
            >>> metric = MulticlassHammingDistance(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelHammingDistance(MultilabelStatScores):
    """Compute the average `Hamming distance`_ (also known as Hamming loss) for multilabel tasks.

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor or float tensor of shape ``(N, C, ...)``. If preds is a
      floating point tensor with values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element. Additionally, we convert to int tensor with thresholding using the value in
      ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlhd`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

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
        >>> from paddle.metric.classification import MultilabelHammingDistance
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelHammingDistance(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)
        >>> mlhd = MultilabelHammingDistance(num_labels=3, average=None)
        >>> mlhd(preds, target)
        tensor([0.0000, 0.5000, 0.5000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelHammingDistance
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelHammingDistance(num_labels=3)
        >>> metric(preds, target)
        tensor(0.3333)
        >>> mlhd = MultilabelHammingDistance(num_labels=3, average=None)
        >>> mlhd(preds, target)
        tensor([0.0000, 0.5000, 0.5000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MultilabelHammingDistance
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelHammingDistance(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.6667, 0.8333])
        >>> mlhd = MultilabelHammingDistance(num_labels=3, multidim_average='samplewise', average=None)
        >>> mlhd(preds, target)
        tensor([[0.5000, 0.5000, 1.0000],
                [1.0000, 1.0000, 0.5000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _hamming_distance_reduce(
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

            >>> # Example plotting a single value
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelHammingDistance
            >>> metric = MultilabelHammingDistance(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelHammingDistance
            >>> metric = MultilabelHammingDistance(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class HammingDistance(_ClassificationTaskWrapper):
    """Compute the average `Hamming distance`_ (also known as Hamming loss).

    .. math::
        \\text{Hamming distance} = \\frac{1}{N \\cdot L} \\sum_i^N \\sum_l^L 1(y_{il} \\neq \\hat{y}_{il})

    Where :math:`y` is a tensor of target values, :math:`\\hat{y}` is a tensor of predictions,
    and :math:`\\bullet_{il}` refers to the :math:`l`-th label of the :math:`i`-th sample of that
    tensor.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryHammingDistance`,
    :class:`~paddle.metric.classification.MulticlassHammingDistance` and
    :class:`~paddle.metric.classification.MultilabelHammingDistance` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([[0, 1], [1, 1]])
        >>> preds = tensor([[0, 1], [0, 1]])
        >>> hamming_distance = HammingDistance(task="multilabel", num_labels=2)
        >>> hamming_distance(preds, target)
        tensor(0.2500)

    """

    def __new__(
        cls: type[HammingDistance],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        multidim_average: Literal["global", "samplewise"] | None = "global",
        top_k: int | None = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        assert multidim_average is not None
        kwargs.update(
            {
                "multidim_average": multidim_average,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryHammingDistance(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            if not isinstance(top_k, int):
                raise ValueError(
                    f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`"
                )
            return MulticlassHammingDistance(
                num_classes, top_k, average, **kwargs
            )
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelHammingDistance(
                num_labels, threshold, average, **kwargs
            )
        raise ValueError(f"Task {task} not supported!")
