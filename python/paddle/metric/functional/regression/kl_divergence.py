from __future__ import annotations

from typing_extensions import Literal

import paddle
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.compute import _safe_xlogy


def _kld_update(
    p: paddle.Tensor, q: paddle.Tensor, log_prob: bool
) -> tuple[paddle.Tensor, int]:
    """Update and returns KL divergence scores for each observation and the total number of observations.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1

    """
    _check_same_shape(p, q)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(
            f"Expected both p and q distribution to be 2D but got {p.ndim} and {q.ndim} respectively"
        )
    total = p.shape[0]
    if log_prob:
        measures = paddle.sum(p.exp() * (p - q), axis=-1)
    else:
        p = p / p.sum(axis=-1, keepdim=True)
        q = q / q.sum(axis=-1, keepdim=True)
        measures = _safe_xlogy(p, p / q).sum(axis=-1)
    return measures, total


def _kld_compute(
    measures: paddle.Tensor,
    total: int | paddle.Tensor,
    reduction: Literal["mean", "sum", "none"] | None = "mean",
) -> paddle.Tensor:
    """Compute the KL divergenece based on the type of reduction.

    Args:
        measures: Tensor of KL divergence scores for each observation
        total: Number of observations
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> p = paddle.to_tensor([[0.36, 0.48, 0.16]])
        >>> q = paddle.to_tensor([[1/3, 1/3, 1/3]])
        >>> measures, total = _kld_update(p, q, log_prob=False)
        >>> _kld_compute(measures, total)
        tensor(0.0853)

    """
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def kl_divergence(
    p: paddle.Tensor,
    q: paddle.Tensor,
    log_prob: bool = False,
    reduction: Literal["mean", "sum", "none"] | None = "mean",
) -> paddle.Tensor:
    """Compute `KL divergence`_.

    .. math::
        D_{KL}(P||Q) = \\sum_{x\\in\\mathcal{X}} P(x) \\log\\frac{P(x)}{Q{x}}

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. It should be noted that the KL divergence
    is a non-symmetrical metric i.e. :math:`D_{KL}(P||Q) \\neq D_{KL}(Q||P)`.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1
        reduction:
            Determines how to reduce over the ``N``/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

    Example:
        >>> from paddle import tensor
        >>> p = tensor([[0.36, 0.48, 0.16]])
        >>> q = tensor([[1/3, 1/3, 1/3]])
        >>> kl_divergence(p, q)
        tensor(0.0853)

    """
    measures, total = _kld_update(p, q, log_prob)
    return _kld_compute(measures, total, reduction)
