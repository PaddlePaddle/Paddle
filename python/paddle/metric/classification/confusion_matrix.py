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
from paddle.metric.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_arg_validation,
    _binary_confusion_matrix_compute,
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_compute,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_compute,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from paddle.metric.metric import Metric
from paddle.metric.utils.enums import ClassificationTask
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.utils.plot import (
    _AX_TYPE,
    _CMAP_TYPE,
    _PLOT_OUT_TYPE,
    plot_confusion_matrix,
)

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = [
        "BinaryConfusionMatrix.plot",
        "MulticlassConfusionMatrix.plot",
        "MultilabelConfusionMatrix.plot",
    ]


class BinaryConfusionMatrix(Metric):
    """Compute the `confusion matrix`_ for binary tasks.

    The confusion matrix :math:`C` is constructed such that :math:`C_{i, j}` is equal to the number of observations
    known to be in class :math:`i` but predicted to be in class :math:`j`. Thus row indices of the confusion matrix
    correspond to the true class labels and column indices correspond to the predicted class labels.

    For binary tasks, the confusion matrix is a 2x2 matrix with the following structure:

    - :math:`C_{0, 0}`: True negatives
    - :math:`C_{0, 1}`: False positives
    - :math:`C_{1, 0}`: False negatives
    - :math:`C_{1, 1}`: True positives

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``confusion_matrix`` (:class:`~paddle.Tensor`): A tensor containing a ``(2, 2)`` matrix

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from paddle.metric.classification import BinaryConfusionMatrix
        >>> target = paddle.to_tensor([1, 1, 0, 0])
        >>> preds = paddle.to_tensor([0, 1, 0, 0])
        >>> bcm = BinaryConfusionMatrix()
        >>> bcm(preds, target)
        tensor([[2, 0],
                [1, 1]])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import BinaryConfusionMatrix
        >>> target = paddle.to_tensor([1, 1, 0, 0])
        >>> preds = paddle.to_tensor([0.35, 0.85, 0.48, 0.01])
        >>> bcm = BinaryConfusionMatrix()
        >>> bcm(preds, target)
        tensor([[2, 0],
                [1, 1]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    confmat: Tensor

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        normalize: Literal["true", "pred", "all", "none"] | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_confusion_matrix_arg_validation(
                threshold, ignore_index, normalize
            )
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args
        self.declare(
            "confmat",
            paddle.zeros(2, 2, dtype=paddle.long),
            dist_reduce_fn="sum",
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _binary_confusion_matrix_tensor_validation(
                preds, target, self.ignore_index
            )
        preds, target = _binary_confusion_matrix_format(
            preds, target, self.threshold, self.ignore_index
        )
        confmat = _binary_confusion_matrix_update(preds, target)
        self.confmat += confmat

    def compute(self) -> paddle.Tensor:
        """Compute confusion matrix."""
        return _binary_confusion_matrix_compute(self.confmat, self.normalize)

    def plot(
        self,
        val: paddle.Tensor | None = None,
        ax: _AX_TYPE | None = None,
        add_text: bool = True,
        labels: list[str] | None = None,
        cmap: _CMAP_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis
            add_text: if the value of each cell should be added to the plot
            labels: a list of strings, if provided will be added to the plot to indicate the different classes
            cmap: matplotlib colormap to use for the confusion matrix
                https://matplotlib.org/stable/users/explain/colors/colormaps.html

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassConfusionMatrix
            >>> metric = MulticlassConfusionMatrix(num_classes=5)
            >>> metric.update(randint(5, (20,)), randint(5, (20,)))
            >>> fig_, ax_ = metric.plot()

        """
        val = val if val is not None else self.compute()
        if not isinstance(val, paddle.Tensor):
            raise TypeError(f"Expected val to be a single tensor but got {val}")
        fig, ax = plot_confusion_matrix(
            val, ax=ax, add_text=add_text, labels=labels, cmap=cmap
        )
        return fig, ax


class MulticlassConfusionMatrix(Metric):
    """Compute the `confusion matrix`_ for multiclass tasks.

    The confusion matrix :math:`C` is constructed such that :math:`C_{i, j}` is equal to the number of observations
    known to be in class :math:`i` but predicted to be in class :math:`j`. Thus row indices of the confusion matrix
    correspond to the true class labels and column indices correspond to the predicted class labels.

    For multiclass tasks, the confusion matrix is a NxN matrix, where:

    - :math:`C_{i, i}` represents the number of true positives for class :math:`i`
    - :math:`\\sum_{j=1, j\\neq i}^N C_{i, j}` represents the number of false negatives for class :math:`i`
    - :math:`\\sum_{j=1, j\\neq i}^N C_{j, i}` represents the number of false positives for class :math:`i`
    - the sum of the remaining cells in the matrix represents the number of true negatives for class :math:`i`

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~paddle.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~paddle.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``confusion_matrix``: [num_classes, num_classes] matrix

    Args:
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    Example (pred is float tensor):
        >>> from paddle.metric.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58], [0.22, 0.61, 0.17], [0.71, 0.09, 0.20], [0.05, 0.82, 0.13]])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    confmat: Tensor

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        normalize: Literal["none", "true", "pred", "all"] | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_confusion_matrix_arg_validation(
                num_classes, ignore_index, normalize
            )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args
        self.declare(
            "confmat",
            paddle.zeros(num_classes, num_classes, dtype=paddle.long),
            dist_reduce_fn="sum",
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multiclass_confusion_matrix_tensor_validation(
                preds, target, self.num_classes, self.ignore_index
            )
        preds, target = _multiclass_confusion_matrix_format(
            preds, target, self.ignore_index
        )
        confmat = _multiclass_confusion_matrix_update(
            preds, target, self.num_classes
        )
        self.confmat += confmat

    def compute(self) -> paddle.Tensor:
        """Compute confusion matrix."""
        return _multiclass_confusion_matrix_compute(
            self.confmat, self.normalize
        )

    def plot(
        self,
        val: paddle.Tensor | None = None,
        ax: _AX_TYPE | None = None,
        add_text: bool = True,
        labels: list[str] | None = None,
        cmap: _CMAP_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis
            add_text: if the value of each cell should be added to the plot
            labels: a list of strings, if provided will be added to the plot to indicate the different classes
            cmap: matplotlib colormap to use for the confusion matrix
                https://matplotlib.org/stable/users/explain/colors/colormaps.html

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassConfusionMatrix
            >>> metric = MulticlassConfusionMatrix(num_classes=5)
            >>> metric.update(randint(5, (20,)), randint(5, (20,)))
            >>> fig_, ax_ = metric.plot()

        """
        val = val if val is not None else self.compute()
        if not isinstance(val, paddle.Tensor):
            raise TypeError(f"Expected val to be a single tensor but got {val}")
        fig, ax = plot_confusion_matrix(
            val, ax=ax, add_text=add_text, labels=labels, cmap=cmap
        )
        return fig, ax


class MultilabelConfusionMatrix(Metric):
    """Compute the `confusion matrix`_ for multilabel tasks.

    The confusion matrix :math:`C` is constructed such that :math:`C_{i, j}` is equal to the number of observations
    known to be in class :math:`i` but predicted to be in class :math:`j`. Thus row indices of the confusion matrix
    correspond to the true class labels and column indices correspond to the predicted class labels.

    For multilabel tasks, the confusion matrix is a Nx2x2 tensor, where each 2x2 matrix corresponds to the confusion
    for that label. The structure of each 2x2 matrix is as follows:

    - :math:`C_{0, 0}`: True negatives
    - :math:`C_{0, 1}`: False positives
    - :math:`C_{1, 0}`: False negatives
    - :math:`C_{1, 1}`: True positives

    As input to 'update' the metric accepts the following input:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    As output of 'compute' the metric returns the following output:

    - ``confusion matrix``: [num_labels,2,2] matrix

    Args:
        num_classes: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from paddle import tensor
        >>> from paddle.metric.classification import MultilabelConfusionMatrix
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelConfusionMatrix(num_labels=3)
        >>> metric(preds, target)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])

    Example (preds is float tensor):
        >>> from paddle.metric.classification import MultilabelConfusionMatrix
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelConfusionMatrix(num_labels=3)
        >>> metric(preds, target)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])

    """

    is_differentiable: bool = False
    higher_is_better: bool | None = None
    full_state_update: bool = False
    confmat: Tensor

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        ignore_index: int | None = None,
        normalize: Literal["none", "true", "pred", "all"] | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold, ignore_index, normalize
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args
        self.declare(
            "confmat",
            paddle.zeros(num_labels, 2, 2, dtype=paddle.long),
            dist_reduce_fn="sum",
        )

    def update(self, preds: paddle.Tensor, target: paddle.Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multilabel_confusion_matrix_tensor_validation(
                preds, target, self.num_labels, self.ignore_index
            )
        preds, target = _multilabel_confusion_matrix_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        confmat = _multilabel_confusion_matrix_update(
            preds, target, self.num_labels
        )
        self.confmat += confmat

    def compute(self) -> paddle.Tensor:
        """Compute confusion matrix."""
        return _multilabel_confusion_matrix_compute(
            self.confmat, self.normalize
        )

    def plot(
        self,
        val: paddle.Tensor | None = None,
        ax: _AX_TYPE | None = None,
        add_text: bool = True,
        labels: list[str] | None = None,
        cmap: _CMAP_TYPE | None = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis
            add_text: if the value of each cell should be added to the plot
            labels: a list of strings, if provided will be added to the plot to indicate the different classes
            cmap: matplotlib colormap to use for the confusion matrix
                https://matplotlib.org/stable/users/explain/colors/colormaps.html

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from paddle import randint
            >>> from paddle.metric.classification import MulticlassConfusionMatrix
            >>> metric = MulticlassConfusionMatrix(num_classes=5)
            >>> metric.update(randint(5, (20,)), randint(5, (20,)))
            >>> fig_, ax_ = metric.plot()

        """
        val = val if val is not None else self.compute()
        if not isinstance(val, paddle.Tensor):
            raise TypeError(f"Expected val to be a single tensor but got {val}")
        fig, ax = plot_confusion_matrix(
            val, ax=ax, add_text=add_text, labels=labels, cmap=cmap
        )
        return fig, ax


class ConfusionMatrix(_ClassificationTaskWrapper):
    """Compute the `confusion matrix`_.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :class:`~paddle.metric.classification.BinaryConfusionMatrix`,
    :class:`~paddle.metric.classification.MulticlassConfusionMatrix` and
    :class:`~paddle.metric.classification.MultilabelConfusionMatrix` for the specific details of each argument influence
    and examples.

    Legacy Example:
        >>> from paddle import tensor
        >>> target = tensor([1, 1, 0, 0])
        >>> preds = tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(task="binary", num_classes=2)
        >>> confmat(preds, target)
        tensor([[2, 0],
                [1, 1]])

        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> confmat = ConfusionMatrix(task="multiclass", num_classes=3)
        >>> confmat(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> confmat = ConfusionMatrix(task="multilabel", num_labels=3)
        >>> confmat(preds, target)
        tensor([[[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]]])

    """

    def __new__(
        cls: type[ConfusionMatrix],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        normalize: Literal["true", "pred", "all", "none"] | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update(
            {
                "normalize": normalize,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTask.BINARY:
            return BinaryConfusionMatrix(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassConfusionMatrix(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(
                    f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`"
                )
            return MultilabelConfusionMatrix(num_labels, threshold, **kwargs)
        raise ValueError(f"Task {task} not supported!")
