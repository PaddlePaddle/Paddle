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
from paddle.metric.functional.classification.specificity_sensitivity import (
    _binary_specificity_at_sensitivity_arg_validation,
    _binary_specificity_at_sensitivity_compute,
    _multiclass_specificity_at_sensitivity_arg_validation,
    _multiclass_specificity_at_sensitivity_compute,
    _multilabel_specificity_at_sensitivity_arg_validation,
    _multilabel_specificity_at_sensitivity_compute,
)
from paddle.metric.utils.data import dim_zero_cat as _cat
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    import paddle
    from paddle.metric.metric import Metric

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinarySpecificityAtSensitivity.plot",
        "MulticlassSpecificityAtSensitivity.plot",
        "MultilabelSpecificityAtSensitivity.plot",
    ]


class BinarySpecificityAtSensitivity(BinaryPrecisionRecallCurve):
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the specificity for a given sensitivity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        min_sensitivity: float value specifying minimum sensitivity threshold.
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

    Returns:
        (tuple): a tuple of 2 tensors containing:

        - specificity: an scalar tensor with the maximum specificity for the given sensitivity level
        - threshold: an scalar tensor with the corresponding threshold level

    Example:
        >>> from paddle.metric.classification import BinarySpecificityAtSensitivity
        >>> from paddle import tensor
        >>> preds = tensor([0, 0.5, 0.4, 0.1])
        >>> target = tensor([0, 1, 1, 1])
        >>> metric = BinarySpecificityAtSensitivity(min_sensitivity=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.4000))
        >>> metric = BinarySpecificityAtSensitivity(min_sensitivity=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.2500))

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        min_sensitivity: float,
        thresholds: int | list[float] | paddle.Tensor | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            thresholds, ignore_index, validate_args=False, **kwargs
        )
        if validate_args:
            _binary_specificity_at_sensitivity_arg_validation(
                min_sensitivity, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_sensitivity = min_sensitivity

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (_cat(self.preds), _cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _binary_specificity_at_sensitivity_compute(
            state, self.thresholds, self.min_sensitivity
        )


class MulticlassSpecificityAtSensitivity(MulticlassPrecisionRecallCurve):
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the specificity for a given sensitivity level.

    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported by
    this metric.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        min_sensitivity: float value specifying minimum sensitivity threshold.
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

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - specificity: an 1d tensor of size (n_classes, ) with the maximum specificity for the given
            sensitivity level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class


    Example:
        >>> from paddle.metric.classification import MulticlassSpecificityAtSensitivity
        >>> from paddle import tensor
        >>> preds = tensor(
        ...     [
        ...         [0.75, 0.05, 0.05, 0.05, 0.05],
        ...         [0.05, 0.75, 0.05, 0.05, 0.05],
        ...         [0.05, 0.05, 0.75, 0.05, 0.05],
        ...         [0.05, 0.05, 0.05, 0.75, 0.05],
        ...     ]
        ... )
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassSpecificityAtSensitivity(num_classes=5, min_sensitivity=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([7.5000e-01, 7.5000e-01, 5.0000e-02, 5.0000e-02, 1.0000e+06]))
        >>> metric = MulticlassSpecificityAtSensitivity(num_classes=5, min_sensitivity=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([7.5000e-01, 7.5000e-01, 0.0000e+00, 0.0000e+00, 1.0000e+06]))

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
        min_sensitivity: float,
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
            _multiclass_specificity_at_sensitivity_arg_validation(
                num_classes, min_sensitivity, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_sensitivity = min_sensitivity

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (_cat(self.preds), _cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multiclass_specificity_at_sensitivity_compute(
            state, self.num_classes, self.thresholds, self.min_sensitivity
        )


class MultilabelSpecificityAtSensitivity(MultilabelPrecisionRecallCurve):
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the specificity for a given sensitivity level.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        min_sensitivity: float value specifying minimum sensitivity threshold.
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

    Returns:
        (tuple): a tuple of either 2 tensors or 2 lists containing

        - specificity: an 1d tensor of size (n_classes, ) with the maximum specificity for the given
            sensitivity level per class
        - thresholds: an 1d tensor of size (n_classes, ) with the corresponding threshold level per class

    Example:
        >>> from paddle.metric.classification import MultilabelSpecificityAtSensitivity
        >>> from paddle import tensor
        >>> preds = tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric = MultilabelSpecificityAtSensitivity(num_labels=3, min_sensitivity=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor([1.0000, 0.5000, 1.0000]), tensor([0.7500, 0.6500, 0.3500]))
        >>> metric = MultilabelSpecificityAtSensitivity(num_labels=3, min_sensitivity=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor([1.0000, 0.5000, 1.0000]), tensor([0.7500, 0.5000, 0.2500]))

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
        min_sensitivity: float,
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
            _multilabel_specificity_at_sensitivity_arg_validation(
                num_labels, min_sensitivity, thresholds, ignore_index
            )
        self.validate_args = validate_args
        self.min_sensitivity = min_sensitivity

    def compute(self) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Compute metric."""
        state = (
            (_cat(self.preds), _cat(self.target))
            if self.thresholds is None
            else self.confmat
        )
        return _multilabel_specificity_at_sensitivity_compute(
            state,
            self.num_labels,
            self.thresholds,
            self.ignore_index,
            self.min_sensitivity,
        )


class SpecificityAtSensitivity(_ClassificationTaskWrapper):
    """Compute the highest possible specificity value given the minimum sensitivity thresholds provided.

    This is done by first calculating the Receiver Operating Characteristic (ROC) curve for different thresholds and the
    find the specificity for a given sensitivity level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinarySpecificityAtSensitivity`,
    :class:`~paddle.metric.classification.MulticlassSpecificityAtSensitivity` and
    :class:`~paddle.metric.classification.MultilabelSpecificityAtSensitivity` for the specific details of each argument
    influence and examples.

    """

    def __new__(
        cls: type[SpecificityAtSensitivity],
        task: Literal["binary", "multiclass", "multilabel"],
        min_sensitivity: float,
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
            return BinarySpecificityAtSensitivity(
                min_sensitivity,
                thresholds,
                ignore_index,
                validate_args,
                **kwargs,
            )
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassSpecificityAtSensitivity(
                num_classes,
                min_sensitivity,
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
            return MultilabelSpecificityAtSensitivity(
                num_labels,
                min_sensitivity,
                thresholds,
                ignore_index,
                validate_args,
                **kwargs,
            )
        raise ValueError(f"Task {task} not supported!")
