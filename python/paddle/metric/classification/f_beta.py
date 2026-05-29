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
from paddle.metric.functional.classification.f_beta import (
    _binary_fbeta_score_arg_validation,
    _fbeta_reduce,
    _multiclass_fbeta_score_arg_validation,
    _multilabel_fbeta_score_arg_validation,
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
        "BinaryFBetaScore.plot",
        "MulticlassFBetaScore.plot",
        "MultilabelFBetaScore.plot",
        "BinaryF1Score.plot",
        "MulticlassF1Score.plot",
        "MultilabelF1Score.plot",
    ]


class BinaryFBetaScore(BinaryStatScores):
    """Compute `F-score`_ metric for binary tasks.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered a score of `zero_division`
    (0 or 1, default is 0) is returned.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor or float tensor of shape ``(N, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bfbs`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global`` the output will be a scalar tensor
        - If ``multidim_average`` is set to ``samplewise`` the output will be a tensor of shape ``(N,)`` consisting of
          a scalar value per sample.

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryFBetaScore
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryFBetaScore(beta=2.0)
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryFBetaScore
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryFBetaScore(beta=2.0)
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from paddle.metric.classification import BinaryFBetaScore
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryFBetaScore(beta=2.0, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5882, 0.0000])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        beta: float,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _binary_fbeta_score_arg_validation(
                beta, threshold, multidim_average, ignore_index, zero_division
            )
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.beta = beta

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            self.beta,
            average="binary",
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
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
            >>> from paddle.metric.classification import BinaryFBetaScore
            >>> metric = BinaryFBetaScore(beta=2.0)
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryFBetaScore
            >>> metric = BinaryFBetaScore(beta=2.0)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassFBetaScore(MulticlassStatScores):
    """Compute `F-score`_ metric for multiclass tasks.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any class, the metric for that class
    will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcfbs`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
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
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassFBetaScore
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3)
        >>> metric(preds, target)
        tensor(0.7963)
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, average=None)
        >>> mcfbs(preds, target)
        tensor([0.5556, 0.8333, 1.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MulticlassFBetaScore
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3)
        >>> metric(preds, target)
        tensor(0.7963)
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, average=None)
        >>> mcfbs(preds, target)
        tensor([0.5556, 0.8333, 1.0000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MulticlassFBetaScore
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassFBetaScore(beta=2.0, num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4697, 0.2706])
        >>> mcfbs = MulticlassFBetaScore(beta=2.0, num_classes=3, multidim_average='samplewise', average=None)
        >>> mcfbs(preds, target)
        tensor([[0.9091, 0.0000, 0.5000],
                [0.0000, 0.3571, 0.4545]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        beta: float,
        num_classes: int,
        top_k: int = 1,
        average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multiclass_fbeta_score_arg_validation(
                beta,
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.beta = beta

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            self.beta,
            average=self.average,
            multidim_average=self.multidim_average,
            zero_division=self.zero_division,
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
            >>> from paddle.metric.classification import MulticlassFBetaScore
            >>> metric = MulticlassFBetaScore(num_classes=3, beta=2.0, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassFBetaScore
            >>> metric = MulticlassFBetaScore(num_classes=3, beta=2.0, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelFBetaScore(MultilabelStatScores):
    """Compute `F-score`_ metric for multilabel tasks.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any label, the metric for that label
    will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, C, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlfbs`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
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
        beta: Weighting between precision and recall in calculation. Setting to 1 corresponds to equal weight
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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelFBetaScore
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelFBetaScore(beta=2.0, num_labels=3)
        >>> metric(preds, target)
        tensor(0.6111)
        >>> mlfbs = MultilabelFBetaScore(beta=2.0, num_labels=3, average=None)
        >>> mlfbs(preds, target)
        tensor([1.0000, 0.0000, 0.8333])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelFBetaScore
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelFBetaScore(beta=2.0, num_labels=3)
        >>> metric(preds, target)
        tensor(0.6111)
        >>> mlfbs = MultilabelFBetaScore(beta=2.0, num_labels=3, average=None)
        >>> mlfbs(preds, target)
        tensor([1.0000, 0.0000, 0.8333])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MultilabelFBetaScore
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5556, 0.0000])
        >>> mlfbs = MultilabelFBetaScore(num_labels=3, beta=2.0, multidim_average='samplewise', average=None)
        >>> mlfbs(preds, target)
        tensor([[0.8333, 0.8333, 0.0000],
                [0.0000, 0.0000, 0.0000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        beta: float,
        num_labels: int,
        threshold: float = 0.5,
        average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=False,
            **kwargs,
        )
        if validate_args:
            _multilabel_fbeta_score_arg_validation(
                beta,
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.validate_args = validate_args
        self.zero_division = zero_division
        self.beta = beta

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            self.beta,
            average=self.average,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
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
            >>> from paddle.metric.classification import MultilabelFBetaScore
            >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelFBetaScore
            >>> metric = MultilabelFBetaScore(num_labels=3, beta=2.0)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class BinaryF1Score(BinaryFBetaScore):
    """Compute F-1 score for binary tasks.

    .. math::
        F_{1} = 2\\frac{\\text{precision} * \\text{recall}}{(\\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered a score of `zero_division`
    (0 or 1, default is 0) is returned.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bf1s`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``multidim_average`` argument:

        - If ``multidim_average`` is set to ``global``, the metric returns a scalar value.
        - If ``multidim_average`` is set to ``samplewise``, the metric returns ``(N,)`` vector consisting of a scalar
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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryF1Score
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryF1Score()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryF1Score
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryF1Score()
        >>> metric(preds, target)
        tensor(0.6667)

    Example (multidim tensors):
        >>> from paddle.metric.classification import BinaryF1Score
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryF1Score(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.5000, 0.0000])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        threshold: float = 0.5,
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            beta=1.0,
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
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
            >>> from paddle.metric.classification import BinaryF1Score
            >>> metric = BinaryF1Score()
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryF1Score
            >>> metric = BinaryF1Score()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassF1Score(MulticlassFBetaScore):
    """Compute F-1 score for multiclass tasks.

    .. math::
        F_{1} = 2\\frac{\\text{precision} * \\text{recall}}{(\\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively.  If this case is encountered for any class, the metric for that class
    will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``paddle.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcf1s`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
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
        preds: Tensor with predictions
        target: Tensor with true labels
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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassF1Score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassF1Score(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7778)
        >>> mcf1s = MulticlassF1Score(num_classes=3, average=None)
        >>> mcf1s(preds, target)
        tensor([0.6667, 0.6667, 1.0000])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MulticlassF1Score
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassF1Score(num_classes=3)
        >>> metric(preds, target)
        tensor(0.7778)
        >>> mcf1s = MulticlassF1Score(num_classes=3, average=None)
        >>> mcf1s(preds, target)
        tensor([0.6667, 0.6667, 1.0000])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MulticlassF1Score
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassF1Score(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4333, 0.2667])
        >>> mcf1s = MulticlassF1Score(num_classes=3, multidim_average='samplewise', average=None)
        >>> mcf1s(preds, target)
        tensor([[0.8000, 0.0000, 0.5000],
                [0.0000, 0.4000, 0.4000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            beta=1.0,
            num_classes=num_classes,
            top_k=top_k,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
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
            >>> from paddle.metric.classification import MulticlassF1Score
            >>> metric = MulticlassF1Score(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> # Example plotting a multiple values per class
            >>> from paddle.metric.classification import MulticlassF1Score
            >>> metric = MulticlassF1Score(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MultilabelF1Score(MultilabelFBetaScore):
    """Compute F-1 score for multilabel tasks.

    .. math::
        F_{1} = 2\\frac{\\text{precision} * \\text{recall}}{(\\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any label, the metric for that label
    will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, C, ...)``.
      If preds is a floating point tensor with values outside [0,1] range we consider the input to be logits and
      will auto apply sigmoid per element. Additionally, we convert to int tensor with thresholding using the value
      in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlf1s`` (:class:`~paddle.Tensor`): A tensor whose returned shape depends on the ``average`` and
      ``multidim_average`` arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)```

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
        zero_division: Should be `0` or `1`. The value returned when
            :math:`\\text{TP} + \\text{FP} = 0 \\wedge \\text{TP} + \\text{FN} = 0`.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelF1Score
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelF1Score(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5556)
        >>> mlf1s = MultilabelF1Score(num_labels=3, average=None)
        >>> mlf1s(preds, target)
        tensor([1.0000, 0.0000, 0.6667])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelF1Score
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelF1Score(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5556)
        >>> mlf1s = MultilabelF1Score(num_labels=3, average=None)
        >>> mlf1s(preds, target)
        tensor([1.0000, 0.0000, 0.6667])

    Example (multidim tensors):
        >>> from paddle.metric.classification import MultilabelF1Score
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]], [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = MultilabelF1Score(num_labels=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.4444, 0.0000])
        >>> mlf1s = MultilabelF1Score(num_labels=3, multidim_average='samplewise', average=None)
        >>> mlf1s(preds, target)
        tensor([[0.6667, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.0000]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Literal["micro", "macro", "weighted", "none"] | None = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            beta=1.0,
            num_labels=num_labels,
            threshold=threshold,
            average=average,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
            zero_division=zero_division,
            **kwargs,
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
            >>> from paddle.metric.classification import MultilabelF1Score
            >>> metric = MultilabelF1Score(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import MultilabelF1Score
            >>> metric = MultilabelF1Score(num_labels=3)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class FBetaScore(_ClassificationTaskWrapper):
    """Compute `F-score`_ metric.

    .. math::
        F_{\\beta} = (1 + \\beta^2) * \\frac{\\text{precision} * \\text{recall}}
        {(\\beta^2 * \\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any class/label, the metric for that
    class/label will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be
    affected in turn.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryFBetaScore`,
    :class:`~paddle.metric.classification.MulticlassFBetaScore` and
    :class:`~paddle.metric.classification.MultilabelFBetaScore` for the specific details of each argument influence
    and examples.

    Legcy Example:
        >>> from paddle import tensor
        >>> target = tensor([0, 1, 2, 0, 1, 2])
        >>> preds = tensor([0, 2, 1, 0, 0, 1])
        >>> f_beta = FBetaScore(task="multiclass", num_classes=3, beta=0.5)
        >>> f_beta(preds, target)
        tensor(0.3333)

    """

    def __new__(
        cls: type[FBetaScore],
        task: Literal["binary", "multiclass", "multilabel"],
        beta: float = 1.0,
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        multidim_average: Literal["global", "samplewise"] | None = "global",
        top_k: int | None = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
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
                "zero_division": zero_division,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryFBetaScore(beta, threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            if not isinstance(top_k, int):
                raise ValueError(
                    f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`"
                )
            return MulticlassFBetaScore(
                beta, num_classes, top_k, average, **kwargs
            )
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelFBetaScore(
                beta, num_labels, threshold, average, **kwargs
            )
        raise ValueError(f"Task {task} not supported!")


class F1Score(_ClassificationTaskWrapper):
    """Compute F-1 score.

    .. math::
        F_{1} = 2\\frac{\\text{precision} * \\text{recall}}{(\\text{precision}) + \\text{recall}}

    The metric is only proper defined when :math:`\\text{TP} + \\text{FP} \\neq 0 \\wedge \\text{TP} + \\text{FN} \\neq 0`
    where :math:`\\text{TP}`, :math:`\\text{FP}` and :math:`\\text{FN}` represent the number of true positives, false
    positives and false negatives respectively. If this case is encountered for any class/label, the metric for that
    class/label will be set to `zero_division` (0 or 1, default is 0) and the overall metric may therefore be
    affected in turn.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryF1Score`, :class:`~paddle.metric.classification.MulticlassF1Score` and
    :class:`~paddle.metric.classification.MultilabelF1Score` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([0, 1, 2, 0, 1, 2])
        >>> preds = tensor([0, 2, 1, 0, 0, 1])
        >>> f1 = F1Score(task="multiclass", num_classes=3)
        >>> f1(preds, target)
        tensor(0.3333)

    """

    def __new__(
        cls: type[F1Score],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: Literal["micro", "macro", "weighted", "none"] | None = "micro",
        multidim_average: Literal["global", "samplewise"] | None = "global",
        top_k: int | None = 1,
        ignore_index: int | None = None,
        validate_args: bool = True,
        zero_division: float = 0,
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
                "zero_division": zero_division,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryF1Score(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            if not isinstance(top_k, int):
                raise ValueError(
                    f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`"
                )
            return MulticlassF1Score(num_classes, top_k, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelF1Score(num_labels, threshold, average, **kwargs)
        raise ValueError(f"Task {task} not supported!")
