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
from paddle.metric.functional.classification.auroc import _reduce_auroc
from paddle.metric.functional.classification.roc import (
    _binary_roc_compute,
    _multiclass_roc_compute,
    _multilabel_roc_compute,
)
from paddle.metric.utils.compute import _auc_compute_without_check
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

if TYPE_CHECKING:
    import paddle
    from paddle.metric.metric import Metric

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryROC.plot",
        "MulticlassROC.plot",
        "MultilabelROC.plot",
    ]


class BinaryROC(BinaryPrecisionRecallCurve):
    """Compute the Receiver Operating Characteristic (ROC) for binary tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns a tuple of 3 tensors containing:

    - ``fpr`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with false positive rate values
    - ``tpr`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with true positive rate values
    - ``thresholds`` (:class:`~paddle.Tensor`): A 1d tensor of size ``(n_thresholds, )`` with decreasing threshold
      values

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a
       binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will
       activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
       `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    .. attention::
       The outputted thresholds will be in reversed order to ensure that they correspond to both fpr and
       tpr which are sorted in reversed order during their calculation, such that they are monotome increasing.

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
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryROC
        >>> preds = tensor([0, 0.5, 0.7, 0.8])
        >>> target = tensor([0, 1, 1, 0])
        >>> metric = BinaryROC(thresholds=None)
        >>> metric(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([1.0000, 0.8000, 0.7000, 0.5000, 0.0000]))
        >>> broc = BinaryROC(thresholds=5)
        >>> broc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 1., 1., 1.]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1", f"{self.name}_2"]

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
            if self.thresholds is None
            else self.confmat
        )
        return _binary_roc_compute(state, self.thresholds)

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor] | None = None,
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
            >>> from paddle.metric.classification import BinaryROC
            >>> preds = rand(20)
            >>> target = randint(2, (20,))
            >>> metric = BinaryROC()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        score = (
            _auc_compute_without_check(
                curve_computed[0], curve_computed[1], 1.0
            )
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("False positive rate", "True positive rate"),
            name=self.__class__.__name__,
        )


class MulticlassROC(MulticlassPrecisionRecallCurve):
    """Compute the Receiver Operating Characteristic (ROC) for binary tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

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

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing

    - ``fpr`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d tensor of
      size ``(n_thresholds+1, )`` with false positive rate values (length may differ between classes). If `thresholds`
      is set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with false positive rate
      values is returned.
    - ``tpr`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d tensor of
      size ``(n_thresholds+1, )`` with true positive rate values (length may differ between classes). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with true positive rate
      values is returned.
    - ``thresholds`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds, )`` with decreasing threshold values (length may differ between classes). If
      `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )`` is returned with shared
      threshold values for all classes.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a
       binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will
       activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
       `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    .. attention::
       Note that outputted thresholds will be in reversed order to ensure that they correspond to both fpr
       and tpr which are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
        num_classes: Integer specifying the number of classes
        thresholds:
            Can be one of:

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

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassROC
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassROC(num_classes=5, thresholds=None)
        >>> fpr, tpr, thresholds = metric(preds, target)
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]),
         tensor([0.0000, 0.3333, 1.0000]), tensor([0., 1.])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0., 0.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.7500, 0.0500]), tensor([1.0000, 0.0500])]
        >>> mcroc = MulticlassROC(num_classes=5, thresholds=5)
        >>> mcroc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.3333, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0., 1., 1., 1., 1.],
                 [0., 1., 1., 1., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 1.],
                 [0., 0., 0., 0., 0.]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1", f"{self.name}_2"]

    def compute(
        self,
    ) -> (
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
    ):
        """Compute metric."""
        state = (
            [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
            if self.thresholds is None
            else self.confmat
        )
        return _multiclass_roc_compute(
            state, self.num_classes, self.thresholds, self.average
        )

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
        | None = None,
        score: paddle.Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
        labels: list[str] | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis
            labels: a list of strings, if provided will be added to the plot to indicate the different classes

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randn, randint
            >>> from paddle.metric.classification import MulticlassROC
            >>> preds = randn(20, 3).softmax(dim=-1)
            >>> target = randint(3, (20,))
            >>> metric = MulticlassROC(num_classes=3)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        score = (
            _reduce_auroc(curve_computed[0], curve_computed[1], average=None)
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("False positive rate", "True positive rate"),
            name=self.__class__.__name__,
            labels=labels,
        )


class MultilabelROC(MultilabelPrecisionRecallCurve):
    """Compute the Receiver Operating Characteristic (ROC) for binary tasks.

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor
      containing ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 3 tensors or 3 lists containing

    - ``fpr`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each label is returned with an 1d tensor of
      size ``(n_thresholds+1, )`` with false positive rate values (length may differ between labels). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with false positive rate
      values is returned.
    - ``tpr`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each label is returned with an 1d tensor of
      size ``(n_thresholds+1, )`` with true positive rate values (length may differ between labels). If `thresholds` is
      set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with true positive rate
      values is returned.
    - ``thresholds`` (:class:`~paddle.Tensor`): if `thresholds=None` a list for each label is returned with an 1d
      tensor of size ``(n_thresholds, )`` with decreasing threshold values (length may differ between labels). If
      `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )`` is returned with shared
      threshold values for all labels.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a
       binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will
       activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
       `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    .. attention::
       The outputted thresholds will be in reversed order to ensure that they correspond to both fpr and tpr
       which are sorted in reversed order during their calculation, such that they are monotome increasing.

    Args:
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
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelROC
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = MultilabelROC(num_labels=3, thresholds=None)
        >>> fpr, tpr, thresholds = metric(preds, target)
        >>> fpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.0000, 0.5000, 1.0000]),
         tensor([0.0000, 0.5000, 0.5000, 0.5000, 1.0000]),
         tensor([0., 0., 0., 1.])]
        >>> tpr  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0000, 0.5000, 0.5000, 1.0000]),
         tensor([0.0000, 0.0000, 0.5000, 1.0000, 1.0000]),
         tensor([0.0000, 0.3333, 0.6667, 1.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.4500, 0.0500]),
         tensor([1.0000, 0.7500, 0.6500, 0.5500, 0.0500]),
         tensor([1.0000, 0.7500, 0.3500, 0.0500])]
        >>> mlroc = MultilabelROC(num_labels=3, thresholds=5)
        >>> mlroc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000],
                 [0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                 [0.0000, 0.0000, 1.0000, 1.0000, 1.0000],
                 [0.0000, 0.3333, 0.3333, 0.6667, 1.0000]]),
         tensor([1.0000, 0.7500, 0.5000, 0.2500, 0.0000]))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Label"

    @property
    def names(self) -> list[str]:
        return [f"{self.name}_0", f"{self.name}_1", f"{self.name}_2"]

    def compute(
        self,
    ) -> (
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
    ):
        """Compute metric."""
        state = (
            [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
            if self.thresholds is None
            else self.confmat
        )
        return _multilabel_roc_compute(
            state, self.num_labels, self.thresholds, self.ignore_index
        )

    def plot(
        self,
        curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
        | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]]
        | None = None,
        score: paddle.Tensor | bool | None = None,
        ax: _AX_TYPE | None = None,
        labels: list[str] | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            curve: the output of either `metric.compute` or `metric.forward`. If no value is provided, will
                automatically call `metric.compute` and plot that result.
            score: Provide a area-under-the-curve score to be displayed on the plot. If `True` and no curve is provided,
                will automatically compute the score. The score is computed by using the trapezoidal rule to compute the
                area under the curve.
            ax: An matplotlib axis object. If provided will add plot to that axis
            labels: a list of strings, if provided will be added to the plot to indicate the different classes

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> from paddle.metric.classification import MultilabelROC
            >>> preds = rand(20, 3)
            >>> target = randint(2, (20, 3))
            >>> metric = MultilabelROC(num_labels=3)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot(score=True)

        """
        curve_computed = curve or self.compute()
        score = (
            _reduce_auroc(curve_computed[0], curve_computed[1], average=None)
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("False positive rate", "True positive rate"),
            name=self.__class__.__name__,
            labels=labels,
        )


class ROC(_ClassificationTaskWrapper):
    """Compute the Receiver Operating Characteristic (ROC).

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryROC`,
    :class:`~paddle.metric.classification.MulticlassROC` and
    :class:`~paddle.metric.classification.MultilabelROC` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> pred = tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = tensor([0, 1, 1, 1])
        >>> roc = ROC(task="binary")
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([1.0000, 0.9526, 0.8808, 0.7311, 0.5000])

        >>> pred = tensor([[0.75, 0.05, 0.05, 0.05], [0.05, 0.75, 0.05, 0.05], [0.05, 0.05, 0.75, 0.05], [0.05, 0.05, 0.05, 0.75]])
        >>> target = tensor([0, 1, 3, 2])
        >>> roc = ROC(task="multiclass", num_classes=4)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500])]

        >>> pred = tensor([[0.8191, 0.3680, 0.1138], [0.3584, 0.7576, 0.1183], [0.2286, 0.3468, 0.1338], [0.8603, 0.0745, 0.1837]])
        >>> target = tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> roc = ROC(task='multilabel', num_labels=3)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr
        [tensor([0., 0., 1., 1., 1.]),
         tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
         tensor([0., 1., 1., 1., 1.])]
        >>> thresholds
        [tensor([1.0000, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.0000, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.0000, 0.1837, 0.1338, 0.1183, 0.1138])]

    """

    def __new__(
        cls: type[ROC],
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
            return BinaryROC(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassROC(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelROC(num_labels, **kwargs)
        raise ValueError(f"Task {task} not supported!")
