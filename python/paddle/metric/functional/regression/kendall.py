from __future__ import annotations

from typing_extensions import Literal

import paddle
from paddle import Tensor
from paddle.metric.functional.regression.utils import (
    _check_data_shape_to_num_outputs,
)
from paddle.metric.utils.checks import _check_same_shape
from paddle.metric.utils.data import _bincount, _cumsum, dim_zero_cat
from paddle.metric.utils.enums import EnumStr


class _MetricVariant(EnumStr):
    """Enumerate for metric variants."""

    A = "a"
    B = "b"
    C = "c"

    @staticmethod
    def _name() -> str:
        return "variant"


class _TestAlternative(EnumStr):
    """Enumerate for test alternative options."""

    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"

    @staticmethod
    def _name() -> str:
        return "alternative"


def _sort_on_first_sequence(
    x: paddle.Tensor, y: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Sort sequences in an ascent order according to the sequence ``x``."""
    y = paddle.clone(x=y)
    x, y = x.T, y.T
    x, perm = paddle.sort(x=x), paddle.argsort(x=x)
    for i in range(x.shape[0]):
        y[i] = y[i][perm[i]]
    return x.T, y.T


def _concordant_element_sum(
    x: paddle.Tensor, y: paddle.Tensor, i: int
) -> paddle.Tensor:
    """Count a total number of concordant pairs in a single sequence."""
    return paddle.logical_and(x[i] < x[i + 1 :], y[i] < y[i + 1 :]).sum(0).unsqueeze(0)


def _count_concordant_pairs(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Count a total number of concordant pairs in given sequences."""
    return paddle.concat(
        [_concordant_element_sum(preds, target, i) for i in range(preds.shape[0])]
    ).sum(0)


def _discordant_element_sum(
    x: paddle.Tensor, y: paddle.Tensor, i: int
) -> paddle.Tensor:
    """Count a total number of discordant pairs in a single sequences."""
    return (
        paddle.logical_or(
            paddle.logical_and(x[i] > x[i + 1 :], y[i] < y[i + 1 :]),
            paddle.logical_and(x[i] < x[i + 1 :], y[i] > y[i + 1 :]),
        )
        .sum(0)
        .unsqueeze(0)
    )


def _count_discordant_pairs(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Count a total number of discordant pairs in given sequences."""
    return paddle.concat(
        [_discordant_element_sum(preds, target, i) for i in range(preds.shape[0])]
    ).sum(0)


def _convert_sequence_to_dense_rank(
    x: paddle.Tensor, sort: bool = False
) -> paddle.Tensor:
    """Convert a sequence to the rank tensor."""
    if sort:
        x = (paddle.sort(axis=0, x=x), paddle.argsort(axis=0, x=x)).values
    _ones = paddle.zeros(1, x.shape[1], dtype=paddle.int32, device=x.place)
    return _cumsum(paddle.concat([_ones, (x[1:] != x[:-1]).int()], axis=0), axis=0)


def _get_ties(x: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Get a total number of ties and staistics for p-value calculation for  a given sequence."""
    ties = paddle.zeros(x.shape[1], dtype=x.dtype, device=x.place)
    ties_p1 = paddle.zeros(x.shape[1], dtype=x.dtype, device=x.place)
    ties_p2 = paddle.zeros(x.shape[1], dtype=x.dtype, device=x.place)
    for dim in range(x.shape[1]):
        n_ties = _bincount(x[:, dim])
        n_ties = n_ties[n_ties > 1]
        ties[dim] = (n_ties * (n_ties - 1) // 2).sum()
        ties_p1[dim] = (n_ties * (n_ties - 1.0) * (n_ties - 2)).sum()
        ties_p2[dim] = (n_ties * (n_ties - 1.0) * (2 * n_ties + 5)).sum()
    return ties, ties_p1, ties_p2


def _get_metric_metadata(
    preds: paddle.Tensor, target: paddle.Tensor, variant: _MetricVariant
) -> tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor | None,
    paddle.Tensor | None,
    paddle.Tensor | None,
    paddle.Tensor | None,
    paddle.Tensor | None,
    paddle.Tensor | None,
    paddle.Tensor,
]:
    """Obtain statistics to calculate metric value."""
    preds, target = _sort_on_first_sequence(preds, target)
    concordant_pairs = _count_concordant_pairs(preds, target)
    discordant_pairs = _count_discordant_pairs(preds, target)
    n_total = paddle.tensor(preds.shape[0], device=preds.place)
    preds_ties = target_ties = None
    preds_ties_p1 = preds_ties_p2 = target_ties_p1 = target_ties_p2 = None
    if variant != _MetricVariant.A:
        preds = _convert_sequence_to_dense_rank(preds)
        target = _convert_sequence_to_dense_rank(target, sort=True)
        preds_ties, preds_ties_p1, preds_ties_p2 = _get_ties(preds)
        target_ties, target_ties_p1, target_ties_p2 = _get_ties(target)
    return (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    )


def _calculate_tau(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    concordant_pairs: paddle.Tensor,
    discordant_pairs: paddle.Tensor,
    con_min_dis_pairs: paddle.Tensor,
    n_total: paddle.Tensor,
    preds_ties: paddle.Tensor | None,
    target_ties: paddle.Tensor | None,
    variant: _MetricVariant,
) -> paddle.Tensor:
    """Calculate Kendall's tau from metric metadata."""
    if variant == _MetricVariant.A:
        return con_min_dis_pairs / (concordant_pairs + discordant_pairs)
    if variant == _MetricVariant.B:
        total_combinations: Tensor = n_total * (n_total - 1) // 2
        if preds_ties is None:
            preds_ties = paddle.tensor(
                0.0, dtype=total_combinations.dtype, device=total_combinations.device
            )
        if target_ties is None:
            target_ties = paddle.tensor(
                0.0, dtype=total_combinations.dtype, device=total_combinations.device
            )
        denominator = (total_combinations - preds_ties) * (
            total_combinations - target_ties
        )
        return con_min_dis_pairs / paddle.sqrt(denominator)
    preds_unique = paddle.tensor(
        [len(p.unique()) for p in preds.T], dtype=preds.dtype, device=preds.device
    )
    target_unique = paddle.tensor(
        [len(t.unique()) for t in target.T], dtype=target.dtype, device=target.device
    )
    min_classes = paddle.minimum(preds_unique, target_unique)
    return 2 * con_min_dis_pairs / ((min_classes - 1) / min_classes * n_total**2)


def _get_p_value_for_t_value_from_dist(t_value: paddle.Tensor) -> paddle.Tensor:
    """Obtain p-value for a given Tensor of t-values. Handle ``nan`` which cannot be passed into torch distributions.

    When t-value is ``nan``, a resulted p-value should be alson ``nan``.

    """
    device = t_value
    normal_dist = paddle.distribution.Normal(
        loc=paddle.tensor([0.0]).to(device), scale=paddle.tensor([1.0]).to(device)
    )
    is_nan = t_value.isnan()
    t_value = t_value.nan_to_num()
    """Not Support auto convert *.cdf, please judge whether it is Pytorch API and convert by yourself"""
def _calculate_p_value(
    con_min_dis_pairs: paddle.Tensor,
    n_total: paddle.Tensor,
    preds_ties: paddle.Tensor | None,
    preds_ties_p1: paddle.Tensor | None,
    preds_ties_p2: paddle.Tensor | None,
    target_ties: paddle.Tensor | None,
    target_ties_p1: paddle.Tensor | None,
    target_ties_p2: paddle.Tensor | None,
    variant: _MetricVariant,
    alternative: _TestAlternative | None,
) -> paddle.Tensor:
    """Calculate p-value for Kendall's tau from metric metadata."""
    t_value_denominator_base = n_total * (n_total - 1) * (2 * n_total + 5)
    if variant == _MetricVariant.A:
        t_value = 3 * con_min_dis_pairs / paddle.sqrt(t_value_denominator_base / 2)
    else:
        m = n_total * (n_total - 1)
        t_value_denominator: Tensor = (
            t_value_denominator_base
            - (preds_ties_p2 if preds_ties_p2 is not None else 0)
            - (target_ties_p2 if target_ties_p2 is not None else 0)
        ) / 18
        t_value_denominator += (
            2
            * (preds_ties if preds_ties is not None else 0)
            * (target_ties if target_ties is not None else 0)
            / m
        )
        t_value_denominator += (
            (preds_ties_p1 if preds_ties_p1 is not None else 0)
            * (target_ties_p1 if target_ties_p1 is not None else 0)
            / (9 * m * (n_total - 2))
        )
        t_value = con_min_dis_pairs / paddle.sqrt(t_value_denominator)
    if alternative == _TestAlternative.TWO_SIDED:
        t_value = paddle.abs(t_value)
    if alternative in [_TestAlternative.TWO_SIDED, _TestAlternative.GREATER]:
        t_value *= -1
    p_value = _get_p_value_for_t_value_from_dist(t_value)
    if alternative == _TestAlternative.TWO_SIDED:
        p_value *= 2
    return p_value


def _kendall_corrcoef_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    concat_preds: list[paddle.Tensor] | None = None,
    concat_target: list[paddle.Tensor] | None = None,
    num_outputs: int = 1,
) -> tuple[list[paddle.Tensor], list[paddle.Tensor]]:
    """Update variables required to compute Kendall rank correlation coefficient.

    Args:
        preds: Sequence of data
        target: Sequence of data
        concat_preds: List of batches of preds sequence to be concatenated
        concat_target: List of batches of target sequence to be concatenated
        num_outputs: Number of outputs in multioutput setting

    Raises:
        RuntimeError: If ``preds`` and ``target`` do not have the same shape

    """
    concat_preds = concat_preds or []
    concat_target = concat_target or []
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    if num_outputs == 1:
        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)
    concat_preds.append(preds)
    concat_target.append(target)
    return concat_preds, concat_target


def _kendall_corrcoef_compute(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    variant: _MetricVariant,
    alternative: _TestAlternative | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    """Compute Kendall rank correlation coefficient, and optionally p-value of corresponding statistical test.

    Args:
        Args:
        preds: Sequence of data
        target: Sequence of data
        variant: Indication of which variant of Kendall's tau to be used
        alternative: Alternative hypothesis for for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    """
    (
        concordant_pairs,
        discordant_pairs,
        preds_ties,
        preds_ties_p1,
        preds_ties_p2,
        target_ties,
        target_ties_p1,
        target_ties_p2,
        n_total,
    ) = _get_metric_metadata(preds, target, variant)
    con_min_dis_pairs = concordant_pairs - discordant_pairs
    tau = _calculate_tau(
        preds,
        target,
        concordant_pairs,
        discordant_pairs,
        con_min_dis_pairs,
        n_total,
        preds_ties,
        target_ties,
        variant,
    )
    p_value = (
        _calculate_p_value(
            con_min_dis_pairs,
            n_total,
            preds_ties,
            preds_ties_p1,
            preds_ties_p2,
            target_ties,
            target_ties_p1,
            target_ties_p2,
            variant,
            alternative,
        )
        if alternative
        else None
    )
    if tau.shape[0] == 1:
        tau = tau.squeeze()
        p_value = p_value.squeeze() if p_value is not None else None
    return tau.clamp(-1, 1), p_value


def kendall_rank_corrcoef(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    variant: Literal["a", "b", "c"] = "b",
    t_test: bool = False,
    alternative: Literal["two-sided", "less", "greater"] | None = "two-sided",
) -> paddle.Tensor | tuple[paddle.Tensor, paddle.Tensor]:
    """Compute `Kendall Rank Correlation Coefficient`_.

    .. math::
        tau_a = \\frac{C - D}{C + D}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs.

    .. math::
        tau_b = \\frac{C - D}{\\sqrt{(C + D + T_{preds}) * (C + D + T_{target})}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs and :math:`T` represents
    a total number of ties.

    .. math::
        tau_c = 2 * \\frac{C - D}{n^2 * \\frac{m - 1}{m}}

    where :math:`C` represents concordant pairs, :math:`D` stands for discordant pairs, :math:`n` is a total number
    of observations and :math:`m` is a ``min`` of unique values in ``preds`` and ``target`` sequence.

    Definitions according to Definition according to `The Treatment of Ties in Ranking Problems`_.

    Args:
        preds: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        target: Sequence of data of either shape ``(N,)`` or ``(N,d)``
        variant: Indication of which variant of Kendall's tau to be used
        t_test: Indication whether to run t-test
        alternative: Alternative hypothesis for t-test. Possible values:
            - 'two-sided': the rank correlation is nonzero
            - 'less': the rank correlation is negative (less than zero)
            - 'greater':  the rank correlation is positive (greater than zero)

    Return:
        Correlation tau statistic
        (Optional) p-value of corresponding statistical test (asymptotic)

    Raises:
        ValueError: If ``t_test`` is not of a type bool
        ValueError: If ``t_test=True`` and ``alternative=None``

    Example (single output regression):
        >>> from paddle.metric.functional.regression import kendall_rank_corrcoef
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> target = paddle.to_tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target)
        tensor(0.3333)

    Example (multi output regression):
        >>> from paddle.metric.functional.regression import kendall_rank_corrcoef
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> target = paddle.to_tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target)
        tensor([1., 1.])

    Example (single output regression with t-test)
        >>> from paddle.metric.functional.regression import kendall_rank_corrcoef
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> target = paddle.to_tensor([3, -0.5, 2, 1])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
        (tensor(0.3333), tensor(0.4969))

    Example (multi output regression with t-test):
        >>> from paddle.metric.functional.regression import kendall_rank_corrcoef
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> target = paddle.to_tensor([[3, -0.5], [2, 1]])
        >>> kendall_rank_corrcoef(preds, target, t_test=True, alternative='two-sided')
            (tensor([1., 1.]), tensor([nan, nan]))

    """
    if not isinstance(t_test, bool):
        raise ValueError(
            f"Argument `t_test` is expected to be of a type `bool`, but got {type(t_test)}."
        )
    if t_test and alternative is None:
        raise ValueError(
            "Argument `alternative` is required if `t_test=True` but got `None`."
        )
    _variant = _MetricVariant.from_str(str(variant))
    _alternative = _TestAlternative.from_str(str(alternative)) if t_test else None
    _preds, _target = _kendall_corrcoef_update(
        preds, target, [], [], num_outputs=1 if preds.ndim == 1 else preds.shape[-1]
    )
    tau, p_value = _kendall_corrcoef_compute(
        dim_zero_cat(_preds), dim_zero_cat(_target), _variant, _alternative
    )
    if p_value is not None:
        return tau, p_value
    return tau
