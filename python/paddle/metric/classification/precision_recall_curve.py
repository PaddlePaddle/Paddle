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

from typing import Any

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.classification.base import _ClassificationTaskWrapper
from paddle.metric.functional.classification.auroc import _reduce_auroc
from paddle.metric.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.compute import _auc_compute_without_check
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryPrecisionRecallCurve.plot",
        "MulticlassPrecisionRecallCurve.plot",
        "MultilabelPrecisionRecallCurve.plot",
    ]


class BinaryPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for binary tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

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

    - ``precision`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds+1, )`` with precision values (length may differ between classes). If `thresholds`
      is set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with precision values
      is returned.
    - ``recall`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d tensor
      of size ``(n_thresholds+1, )`` with recall values (length may differ between classes). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with recall values is returned.
    - ``thresholds`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds, )`` with increasing threshold values (length may differ between classes). If
      `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )`` is returned with
      shared threshold values for all classes.

    .. note::
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

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        normalization:
            Specifies a normalization method that is used for batch-wise update regarding negative logits.
            Set to ``None`` if negative logits are desired in evaluation.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle.metric.classification import BinaryPrecisionRecallCurve
        >>> preds = paddle.to_tensor([0, 0.5, 0.7, 0.8])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> bprc = BinaryPrecisionRecallCurve(thresholds=None)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([0.0000, 0.5000, 0.7000, 0.8000]))
        >>> bprc = BinaryPrecisionRecallCurve(thresholds=5)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.6667, 0.0000,    nan, 1.0000]),
         tensor([1., 1., 1., 0., 0., 0.]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]
    confmat: Tensor

    def __init__(
        self,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        normalization: Literal["sigmoid", "softmax"] | None = "sigmoid",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(
                thresholds, ignore_index
            )
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.normalization = normalization
        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.declare("preds", default=[], dist_reduce_fn="cat")
            self.declare("target", default=[], dist_reduce_fn="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.declare(
                "confmat",
                default=paddle.zeros(len(thresholds), 2, 2, dtype=paddle.long),
                dist_reduce_fn="sum",
            )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(
                preds, target, self.ignore_index
            )
        preds, target, _ = _binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
            self.normalization,
        )
        state = _binary_precision_recall_curve_update(
            preds, target, self.thresholds
        )
        if isinstance(state, paddle.Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _binary_precision_recall_curve_compute(state, self.thresholds)

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor] | None = None,
        score: paddle.Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single curve from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import BinaryPrecisionRecallCurve
            >>> preds = rand(20)
            >>> target = randint(2, (20,))
            >>> metric = BinaryPrecisionRecallCurve()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        curve_computed = curve_computed[1], curve_computed[0], curve_computed[2]
        score = (
            _auc_compute_without_check(
                curve_computed[0], curve_computed[1], direction=-1.0
            )
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("Recall", "Precision"),
            name=self.__class__.__name__,
        )


class MulticlassPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for multiclass tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported by
    this metric.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input to
      be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``precision`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with precision values
    - ``recall`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with recall values
    - ``thresholds`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds, )`` with increasing threshold values

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to a 1D `tensor` of floats, will use the indicated thresholds in the tensor as
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

    Example:
        >>> from paddle.metric.classification import MulticlassPrecisionRecallCurve
        >>> preds = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> mcprc = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=None)
        >>> precision, recall, thresholds = mcprc(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 1., 0.]), tensor([1., 1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]),
         tensor(0.0500)]
        >>> mcprc = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=5)
        >>> mcprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.2500, 1.0000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.2500, 1.0000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000,    nan, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000,    nan, 1.0000],
                 [0.0000,    nan,    nan,    nan,    nan, 1.0000]]),
         tensor([[1., 1., 1., 1., 0., 0.],
                 [1., 1., 1., 1., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [nan, nan, nan, nan, nan, 0.]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        average: Literal["micro", "macro"] | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_precision_recall_curve_arg_validation(
                num_classes, thresholds, ignore_index, average
            )
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.declare("preds", default=[], dist_reduce_fn="cat")
            self.declare("target", default=[], dist_reduce_fn="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.declare(
                "confmat",
                default=paddle.zeros(
                    len(thresholds), num_classes, 2, 2, dtype=paddle.long
                ),
                dist_reduce_fn="sum",
            )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multiclass_precision_recall_curve_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target, _ = _multiclass_precision_recall_curve_format(
            preds,
            target,
            self.num_classes,
            self.thresholds,
            self.ignore_index,
            self.average,
        )
        state = _multiclass_precision_recall_curve_update(
            preds, target, self.num_classes, self.thresholds, self.average
        )
        if isinstance(state, paddle.Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(
        self,
    ) -> (
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
    ):
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multiclass_precision_recall_curve_compute(
            state, self.num_classes, self.thresholds, self.average
        )

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
        | None = None,
        score: paddle.Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randn, randint
            >>> from paddle.metric.classification import MulticlassPrecisionRecallCurve
            >>> preds = randn(20, 3).softmax(dim=-1)
            >>> target = randint(3, (20,))
            >>> metric = MulticlassPrecisionRecallCurve(num_classes=3)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        curve_computed = curve_computed[1], curve_computed[0], curve_computed[2]
        score = (
            _reduce_auroc(
                curve_computed[0],
                curve_computed[1],
                average=None,
                direction=-1.0,
            )
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("Recall", "Precision"),
            name=self.__class__.__name__,
        )


class MultilabelPrecisionRecallCurve(Metric):
    """Compute the precision-recall curve for multilabel tasks.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input to
      be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following a tuple of either 3 tensors or
    3 lists containing:

    - ``precision`` (:class:`~paddle.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is returned
      with an 1d tensor of size ``(n_thresholds+1, )`` with precision values (length may differ between labels). If
      `thresholds` is set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with
      precision values is returned.
    - ``recall`` (:class:`~paddle.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is returned
      with an 1d tensor of size ``(n_thresholds+1, )`` with recall values (length may differ between labels). If
      `thresholds` is set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with recall
      values is returned.
    - ``thresholds`` (:class:`~paddle.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is
      returned with an 1d tensor of size ``(n_thresholds, )`` with increasing threshold values (length may differ
      between labels). If `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )``
      is returned with shared threshold values for all labels.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
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

    Example:
        >>> from paddle.metric.classification import MultilabelPrecisionRecallCurve
        >>> preds = paddle.to_tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = paddle.to_tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> mlprc = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=None)
        >>> precision, recall, thresholds = mlprc(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.5000, 0.5000, 1.0000, 1.0000]), tensor([0.5000, 0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([0.7500, 1.0000, 1.0000, 1.0000])]
        >>> recall  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([1.0000, 1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([1.0000, 0.6667, 0.3333, 0.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0500, 0.4500, 0.7500]), tensor([0.0500, 0.5500, 0.6500, 0.7500]), tensor([0.0500, 0.3500, 0.7500])]
        >>> mlprc = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=5)
        >>> mlprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000,    nan, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000,    nan, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000,    nan, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    preds: list[paddle.Tensor]
    target: list[paddle.Tensor]
    confmat: Tensor

    def __init__(
        self,
        num_labels: int,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_precision_recall_curve_arg_validation(
                num_labels, thresholds, ignore_index
            )
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.declare("preds", default=[], dist_reduce_fn="cat")
            self.declare("target", default=[], dist_reduce_fn="cat")
        else:
            self.register_buffer("thresholds", thresholds, persistent=False)
            self.declare(
                "confmat",
                default=paddle.zeros(
                    len(thresholds), num_labels, 2, 2, dtype=paddle.long
                ),
                dist_reduce_fn="sum",
            )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_precision_recall_curve_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target, _ = _multilabel_precision_recall_curve_format(
            preds, target, self.num_labels, self.thresholds, self.ignore_index
        )
        state = _multilabel_precision_recall_curve_update(
            preds, target, self.num_labels, self.thresholds
        )
        if isinstance(state, paddle.Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(
        self,
    ) -> (
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
    ):
        """Compute metric."""
        state = (
            (dim_zero_cat(self.preds), dim_zero_cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multilabel_precision_recall_curve_compute(
            state, self.num_labels, self.thresholds, self.ignore_index
        )

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
        | None = None,
        score: paddle.Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelPrecisionRecallCurve
            >>> preds = rand(20, 3)
            >>> target = randint(2, (20, 3))
            >>> metric = MultilabelPrecisionRecallCurve(num_labels=3)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        curve_computed = curve_computed[1], curve_computed[0], curve_computed[2]
        score = (
            _reduce_auroc(
                curve_computed[0],
                curve_computed[1],
                average=None,
                direction=-1.0,
            )
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("Recall", "Precision"),
            name=self.__class__.__name__,
        )


class PrecisionRecallCurve(_ClassificationTaskWrapper):
    """Compute the precision-recall curve.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryPrecisionRecallCurve`,
    :class:`~paddle.metric.classification.MulticlassPrecisionRecallCurve` and
    :class:`~paddle.metric.classification.MultilabelPrecisionRecallCurve` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> pred = paddle.to_tensor([0, 0.1, 0.8, 0.4])
        >>> target = paddle.to_tensor([0, 1, 1, 0])
        >>> pr_curve = PrecisionRecallCurve(task="binary")
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.5000, 0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.0000, 0.1000, 0.4000, 0.8000])

        >>> pred = paddle.to_tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = paddle.to_tensor([0, 1, 3, 2])
        >>> pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        [tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 1., 0.]), tensor([1., 1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]),
         tensor(0.0500)]

    """

    def __new__(
        cls: type[PrecisionRecallCurve],
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: int | list[float] | paddle.Tensor | None = None,
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
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryPrecisionRecallCurve(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassPrecisionRecallCurve(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelPrecisionRecallCurve(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")
