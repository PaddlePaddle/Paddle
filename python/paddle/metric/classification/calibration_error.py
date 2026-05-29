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
from paddle.metric.functional.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _binary_calibration_error_tensor_validation,
    _binary_calibration_error_update,
    _binary_confusion_matrix_format,
    _ce_compute,
    _multiclass_calibration_error_arg_validation,
    _multiclass_calibration_error_tensor_validation,
    _multiclass_calibration_error_update,
    _multiclass_confusion_matrix_format,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.data import dim_zero_cat
from paddle.metric.utils.enums import ClassificationTaskNoMultilabel
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE

if TYPE_CHECKING:
    from collections.abc import Sequence

    import paddle
    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryCalibrationError.plot",
        "MulticlassCalibrationError.plot",
    ]


class BinaryCalibrationError(Metric):
    """`Top-label Calibration Error`_ for binary tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the
      positive class.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bce`` (:class:`~paddle.Tensor`): A scalar tensor containing the calibration error

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import BinaryCalibrationError
        >>> preds = tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
        >>> metric(preds, target)
        tensor(0.2900)
        >>> bce = BinaryCalibrationError(n_bins=2, norm='l2')
        >>> bce(preds, target)
        tensor(0.2918)
        >>> bce = BinaryCalibrationError(n_bins=2, norm='max')
        >>> bce(preds, target)
        tensor(0.3167)

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confidences: list[paddle.Tensor]
    accuracies: list[paddle.Tensor]

    def __init__(
        self,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.validate_args = validate_args
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.declare("confidences", [], dist_reduce_fn="cat")
        self.declare("accuracies", [], dist_reduce_fn="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _binary_calibration_error_tensor_validation(
                preds, target, self.ignore_index
            )
        preds, target = _binary_confusion_matrix_format(
            preds,
            target,
            threshold=0.0,
            ignore_index=self.ignore_index,
            convert_to_labels=False,
        )
        confidences, accuracies = _binary_calibration_error_update(
            preds, target
        )
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

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
            >>> from paddle.metric.classification import BinaryCalibrationError
            >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
            >>> metric.update(rand(10), randint(2, (10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import rand, randint
            >>> # Example plotting multiple values
            >>> from paddle.metric.classification import BinaryCalibrationError
            >>> metric = BinaryCalibrationError(n_bins=2, norm='l1')
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MulticlassCalibrationError(Metric):
    """`Top-label Calibration Error`_ for multiclass tasks.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits for
      each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)`` containing ground truth labels, and
      therefore only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    .. tip::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcce`` (:class:`~paddle.Tensor`): A scalar tensor containing the calibration error

    Args:
        num_classes: Integer specifying the number of classes
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassCalibrationError
        >>> preds = tensor([[0.25, 0.20, 0.55], [0.55, 0.05, 0.40], [0.10, 0.30, 0.60], [0.90, 0.05, 0.05]])
        >>> target = tensor([0, 1, 2, 0])
        >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l1')
        >>> metric(preds, target)
        tensor(0.2000)
        >>> mcce = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l2')
        >>> mcce(preds, target)
        tensor(0.2082)
        >>> mcce = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='max')
        >>> mcce(preds, target)
        tensor(0.2333)

    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"
    confidences: list[paddle.Tensor]
    accuracies: list[paddle.Tensor]

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_calibration_error_arg_validation(
                num_classes, n_bins, norm, ignore_index
            )
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.norm = norm
        self.ignore_index = ignore_index
        self.declare("confidences", [], dist_reduce_fn="cat")
        self.declare("accuracies", [], dist_reduce_fn="cat")

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update metric states with predictions and targets."""
        if self.validate_args:
            _multiclass_calibration_error_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target = _multiclass_confusion_matrix_format(
            preds,
            target,
            ignore_index=self.ignore_index,
            convert_to_labels=False,
        )
        confidences, accuracies = _multiclass_calibration_error_update(
            preds, target
        )
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> paddle.Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)

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

            >>> from paddle import randn, randint
            >>> # Example plotting a single value
            >>> from paddle.metric.classification import MulticlassCalibrationError
            >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l1')
            >>> metric.update(randn(20, 3).softmax(dim=-1), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from paddle import randn, randint
            >>> # Example plotting a multiple values
            >>> from paddle.metric.classification import MulticlassCalibrationError
            >>> metric = MulticlassCalibrationError(num_classes=3, n_bins=3, norm='l1')
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randn(20, 3).softmax(dim=-1), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class CalibrationError(_ClassificationTaskWrapper):
    """`Top-label Calibration Error`_.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryCalibrationError` and
    :class:`~paddle.metric.classification.MulticlassCalibrationError` for the specific details of each argument influence
    and examples.

    """

    def __new__(
        cls: type[CalibrationError],
        task: Literal["binary", "multiclass"],
        n_bins: int = 15,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassCalibrationError(num_classes, **kwargs)
        raise ValueError(f"Not handled value: {task}")
