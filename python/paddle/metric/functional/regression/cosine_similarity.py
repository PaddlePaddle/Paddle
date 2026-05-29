from __future__ import annotations

import paddle
from paddle.metric.utils.checks import _check_same_shape


def _cosine_similarity_update(
    preds: paddle.Tensor, target: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Update and returns variables required to compute Cosine Similarity. Checks for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    _check_same_shape(preds, target)
    if preds.ndim != 2:
        raise ValueError(
            f"Expected input to cosine similarity to be 2D tensors of shape `[N,D]` where `N` is the number of samples and `D` is the number of dimensions, but got tensor of shape {preds.shape}"
        )
    preds = preds.float()
    target = target.float()
    return preds, target


def _cosine_similarity_compute(
    preds: paddle.Tensor, target: paddle.Tensor, reduction: str | None = "sum"
) -> paddle.Tensor:
    """Compute Cosine Similarity.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> target = paddle.to_tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        >>> preds = paddle.to_tensor([[1, 2, 3, 4], [-1, -2, -3, -4]])
        >>> preds, target = _cosine_similarity_update(preds, target)
        >>> _cosine_similarity_compute(preds, target, 'none')
        tensor([ 1.0000, -1.0000])

    """
    dot_product = (preds * target).sum(dim=-1)
    preds_norm = preds.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    similarity = dot_product / (preds_norm * target_norm)
    reduction_mapping = {
        "sum": paddle.sum,
        "mean": paddle.mean,
        "none": lambda x: x,
        None: lambda x: x,
    }
    return reduction_mapping[reduction](similarity)


def cosine_similarity(
    preds: paddle.Tensor, target: paddle.Tensor, reduction: str | None = "sum"
) -> paddle.Tensor:
    """Compute the `Cosine Similarity`_.

    .. math::
        cos_{sim}(x,y) = \\frac{x \\cdot y}{||x|| \\cdot ||y||} =
        \\frac{\\sum_{i=1}^n x_i y_i}{\\sqrt{\\sum_{i=1}^n x_i^2}\\sqrt{\\sum_{i=1}^n y_i^2}}

    where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    Args:
        preds: Predicted tensor with shape ``(N,d)``
        target: Ground truth tensor with shape ``(N,d)``
        reduction:
            The method of reducing along the batch dimension using sum, mean or taking the individual scores

    Example:
        >>> from paddle.metric.functional.regression import cosine_similarity
        >>> target = paddle.to_tensor([[1, 2, 3, 4],
        ...                        [1, 2, 3, 4]])
        >>> preds = paddle.to_tensor([[1, 2, 3, 4],
        ...                       [-1, -2, -3, -4]])
        >>> cosine_similarity(preds, target, 'none')
        tensor([ 1.0000, -1.0000])

    """
    preds, target = _cosine_similarity_update(preds, target)
    return _cosine_similarity_compute(preds, target, reduction)
