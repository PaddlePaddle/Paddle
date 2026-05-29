from __future__ import annotations

from typing_extensions import Literal

import paddle
from paddle.metric.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
)
from paddle.metric.utils.compute import normalize_logits_if_needed
from paddle.metric.utils.enums import ClassificationTaskNoMultilabel


def _binning_bucketize(
    confidences: paddle.Tensor, accuracies: paddle.Tensor, bin_boundaries: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Compute calibration bins using ``paddle.bucketize``. Use for ``pytorch >=1.6``.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities

    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = paddle.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    conf_bin = paddle.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    count_bin = paddle.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    indices = paddle.bucketize(confidences, bin_boundaries, right=True) - 1
    count_bin.scatter_add_(dim=0, index=indices, src=paddle.ones_like(confidences))
    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = paddle.nan_to_num(x=conf_bin / count_bin)
    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = paddle.nan_to_num(x=acc_bin / count_bin)
    prop_bin = count_bin / count_bin.sum()
    return acc_bin, conf_bin, prop_bin


def _ce_compute(
    confidences: paddle.Tensor,
    accuracies: paddle.Tensor,
    bin_boundaries: paddle.Tensor | int,
    norm: str = "l1",
    debias: bool = False,
) -> paddle.Tensor:
    """Compute the calibration error given the provided bin boundaries and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1 prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        bin_boundaries: Bin boundaries separating the ``linspace`` from 0 to 1.
        norm: Norm function to use when computing calibration error. Defaults to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`_. Defaults to False.

    Raises:
        ValueError: If an unsupported norm function is provided.

    Returns:
        Tensor: Calibration error scalar.

    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = paddle.linspace(
            0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device
        )
    if norm not in {"l1", "l2", "max"}:
        raise ValueError(
            f"Argument `norm` is expected to be one of 'l1', 'l2', 'max' but got {norm}"
        )
    with paddle.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(
            confidences, accuracies, bin_boundaries
        )
    if norm == "l1":
        return paddle.sum(paddle.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        ce = paddle.max(paddle.abs(acc_bin - conf_bin))
    if norm == "l2":
        ce = paddle.sum(paddle.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:
            debias_bins = (
                acc_bin
                * (acc_bin - 1)
                * prop_bin
                / (prop_bin * accuracies.size()[0] - 1)
            )
            ce += paddle.sum(paddle.nan_to_num(x=debias_bins))
        return paddle.sqrt(ce) if ce > 0 else paddle.tensor(0)
    return ce


def _binary_calibration_error_arg_validation(
    n_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: int | None = None,
) -> None:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(
            f"Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}"
        )
    allowed_norm = "l1", "l2", "max"
    if norm not in allowed_norm:
        raise ValueError(
            f"Expected argument `norm` to be one of {allowed_norm}, but got {norm}."
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )


def _binary_calibration_error_tensor_validation(
    preds: paddle.Tensor, target: paddle.Tensor, ignore_index: int | None = None
) -> None:
    _binary_confusion_matrix_tensor_validation(preds, target, ignore_index)
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected argument `preds` to be floating tensor with probabilities/logits but got tensor with dtype {preds.dtype}"
        )


def _binary_calibration_error_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    confidences, accuracies = preds, target
    return confidences, accuracies


def binary_calibration_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
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

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle.metric.functional.classification import binary_calibration_error
        >>> preds = paddle.to_tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = paddle.to_tensor([0, 0, 1, 1, 1])
        >>> binary_calibration_error(preds, target, n_bins=2, norm='l1')
        tensor(0.2900)
        >>> binary_calibration_error(preds, target, n_bins=2, norm='l2')
        tensor(0.2918)
        >>> binary_calibration_error(preds, target, n_bins=2, norm='max')
        tensor(0.3167)

    """
    if validate_args:
        _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        _binary_calibration_error_tensor_validation(preds, target, ignore_index)
    preds, target = _binary_confusion_matrix_format(
        preds, target, threshold=0.0, ignore_index=ignore_index, convert_to_labels=False
    )
    confidences, accuracies = _binary_calibration_error_update(preds, target)
    return _ce_compute(confidences, accuracies, n_bins, norm)


def _multiclass_calibration_error_arg_validation(
    num_classes: int,
    n_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: int | None = None,
) -> None:
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(
            f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
        )
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(
            f"Expected argument `n_bins` to be an integer larger than 0, but got {n_bins}"
        )
    allowed_norm = "l1", "l2", "max"
    if norm not in allowed_norm:
        raise ValueError(
            f"Expected argument `norm` to be one of {allowed_norm}, but got {norm}."
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(
            f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}"
        )


def _multiclass_calibration_error_tensor_validation(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> None:
    _multiclass_confusion_matrix_tensor_validation(
        preds, target, num_classes, ignore_index
    )
    if not preds.is_floating_point():
        raise ValueError(
            f"Expected argument `preds` to be floating tensor with probabilities/logits but got tensor with dtype {preds.dtype}"
        )


def _multiclass_calibration_error_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    preds = normalize_logits_if_needed(preds, "softmax")
    confidences, predictions = preds.max(axis=1), preds.argmax(axis=1)
    accuracies = predictions.eq(target)
    return confidences.float(), accuracies.float()


def multiclass_calibration_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    num_classes: int,
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
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

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        n_bins: Number of bins to use when computing the metric.
        norm: Norm used to compare empirical and expected probability bins.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from paddle.metric.functional.classification import multiclass_calibration_error
        >>> preds = paddle.to_tensor([[0.25, 0.20, 0.55],
        ...                       [0.55, 0.05, 0.40],
        ...                       [0.10, 0.30, 0.60],
        ...                       [0.90, 0.05, 0.05]])
        >>> target = paddle.to_tensor([0, 1, 2, 0])
        >>> multiclass_calibration_error(preds, target, num_classes=3, n_bins=3, norm='l1')
        tensor(0.2000)
        >>> multiclass_calibration_error(preds, target, num_classes=3, n_bins=3, norm='l2')
        tensor(0.2082)
        >>> multiclass_calibration_error(preds, target, num_classes=3, n_bins=3, norm='max')
        tensor(0.2333)

    """
    if validate_args:
        _multiclass_calibration_error_arg_validation(
            num_classes, n_bins, norm, ignore_index
        )
        _multiclass_calibration_error_tensor_validation(
            preds, target, num_classes, ignore_index
        )
    preds, target = _multiclass_confusion_matrix_format(
        preds, target, ignore_index, convert_to_labels=False
    )
    confidences, accuracies = _multiclass_calibration_error_update(preds, target)
    return _ce_compute(confidences, accuracies, n_bins, norm)


def calibration_error(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    task: Literal["binary", "multiclass"],
    n_bins: int = 15,
    norm: Literal["l1", "l2", "max"] = "l1",
    num_classes: int | None = None,
    ignore_index: int | None = None,
    validate_args: bool = True,
) -> paddle.Tensor:
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
    :func:`~paddle.metric.functional.classification.binary_calibration_error` and
    :func:`~paddle.metric.functional.classification.multiclass_calibration_error` for the specific details of
    each argument influence and examples.

    """
    task = ClassificationTaskNoMultilabel.from_str(task)
    assert norm is not None
    if task == ClassificationTaskNoMultilabel.BINARY:
        return binary_calibration_error(
            preds, target, n_bins, norm, ignore_index, validate_args
        )
    if task == ClassificationTaskNoMultilabel.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return multiclass_calibration_error(
            preds, target, num_classes, n_bins, norm, ignore_index, validate_args
        )
    raise ValueError(
        f"Expected argument `task` to either be `'binary'` or `'multiclass'` but got {task}"
    )
