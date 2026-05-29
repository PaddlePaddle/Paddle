from __future__ import annotations

from typing_extensions import Literal

import paddle
from paddle.metric.functional.regression.kl_divergence import kl_divergence
from paddle.metric.utils.checks import _check_same_shape


def _jsd_update(
    p: paddle.Tensor, q: paddle.Tensor, log_prob: bool
) -> tuple[paddle.Tensor, int]:
    """Update and returns jensen-shannon divergence scores for each observation and the total number of observations.

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
        mean = paddle.logsumexp(paddle.stack([p, q]), axis=0) - paddle.log(
            paddle.tensor(2.0)
        )
        measures = 0.5 * kl_divergence(
            p, mean, log_prob=log_prob, reduction=None
        ) + 0.5 * kl_divergence(q, mean, log_prob=log_prob, reduction=None)
    else:
        p = p / p.sum(axis=-1, keepdim=True)
        q = q / q.sum(axis=-1, keepdim=True)
        mean = (p + q) / 2
        measures = 0.5 * kl_divergence(
            p, mean, log_prob=log_prob, reduction=None
        ) + 0.5 * kl_divergence(q, mean, log_prob=log_prob, reduction=None)
    return measures, total


def _jsd_compute(
    measures: paddle.Tensor,
    total: int | paddle.Tensor,
    reduction: Literal["mean", "sum", "none"] | None = "mean",
) -> paddle.Tensor:
    """Compute and reduce the Jensen-Shannon divergence based on the type of reduction."""
    if reduction == "sum":
        return measures.sum()
    if reduction == "mean":
        return measures.sum() / total
    if reduction is None or reduction == "none":
        return measures
    return measures / total


def jensen_shannon_divergence(
    p: paddle.Tensor,
    q: paddle.Tensor,
    log_prob: bool = False,
    reduction: Literal["mean", "sum", "none"] | None = "mean",
) -> paddle.Tensor:
    """Compute `Jensen-Shannon divergence`_.

    .. math::
        D_{JS}(P||Q) = \\frac{1}{2} D_{KL}(P||M) + \\frac{1}{2} D_{KL}(Q||M)

    Where :math:`P` and :math:`Q` are probability distributions where :math:`P` usually represents a distribution
    over data and :math:`Q` is often a prior or approximation of :math:`P`. :math:`D_{KL}` is the `KL divergence`_ and
    :math:`M` is the average of the two distributions. It should be noted that the Jensen-Shannon divergence is a
    symmetrical metric i.e. :math:`D_{JS}(P||Q) = D_{JS}(Q||P)`.

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
        >>> jensen_shannon_divergence(p, q)
        tensor(0.0225)

    """
    measures, total = _jsd_update(p, q, log_prob)
    return _jsd_compute(measures, total, reduction)
