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

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import paddle
from paddle.metric.metric import Metric
from paddle.metric.utils.data import apply_to_collection
from paddle.metric.utils.imports import _MATPLOTLIB_AVAILABLE
from paddle.metric.wrappers.abstract import WrapperMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from paddle.metric.utils.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["BootStrapper.plot"]


def _bootstrap_sampler(
    size: int, sampling_strategy: str = "poisson"
) -> paddle.Tensor:
    """Resample a tensor along its first dimension with replacement.

    Args:
        size: number of samples
        sampling_strategy: the strategy to use for sampling, either ``'poisson'`` or ``'multinomial'``

    Returns:
        resampled tensor

    """
    if sampling_strategy == "poisson":
        # Poisson resampling not yet implemented for paddle
        raise NotImplementedError(
            "Poisson sampling strategy not yet implemented"
        )


class BootStrapper(WrapperMetric):
    """Using `Turn a Metric into a Bootstrapped`_.

    That can automate the process of getting confidence intervals for metric values. This wrapper
    class basically keeps multiple copies of the same base metric in memory and whenever ``update`` or
    ``forward`` is called, all input tensors are resampled (with replacement) along the first dimension.

    Args:
        base_metric: base metric class to wrap
        num_bootstraps: number of copies to make of the base metric for bootstrapping
        mean: if ``True`` return the mean of the bootstraps
        std: if ``True`` return the standard deviation of the bootstraps
        quantile: if given, returns the quantile of the bootstraps. Can only be used with pytorch version 1.6 or higher
        raw: if ``True``, return all bootstrapped values
        sampling_strategy:
            Determines how to produce bootstrapped samplings. Either ``'poisson'`` or ``multinomial``.
            If ``'possion'`` is chosen, the number of times each sample will be included in the bootstrap
            will be given by :math:`n\\sim Poisson(\\lambda=1)`, which approximates the true bootstrap distribution
            when the number of samples is large. If ``'multinomial'`` is chosen, we will apply true bootstrapping
            at the batch level to approximate bootstrapping over the hole dataset.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> from pprint import pprint
        >>> from paddle import randint
        >>> from paddle.metric.wrappers import BootStrapper
        >>> from paddle.metric.classification import MulticlassAccuracy
        >>> base_metric = MulticlassAccuracy(num_classes=5, average='micro')
        >>> bootstrap = BootStrapper(base_metric, num_bootstraps=20)
        >>> bootstrap.update(randint(5, (20,)), randint(5, (20,)))
        >>> output = bootstrap.compute()
        >>> pprint(output)
        {'mean': tensor(0.2089), 'std': tensor(0.0772)}

    """

    full_state_update: bool | None = True

    def __init__(
        self,
        base_metric: Metric,
        num_bootstraps: int = 10,
        mean: bool = True,
        std: bool = True,
        quantile: float | paddle.Tensor | None = None,
        raw: bool = False,
        sampling_strategy: str = "poisson",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected base metric to be an instance of paddle.metric.Metric but received {base_metric}"
            )
        self.metrics = paddle.nn.LayerList(
            [deepcopy(base_metric) for _ in range(num_bootstraps)]
        )
        self.num_bootstraps = num_bootstraps
        self.mean = mean
        self.std = std
        self.quantile = quantile
        self.raw = raw
        allowed_sampling = "poisson", "multinomial"
        if sampling_strategy not in allowed_sampling:
            raise ValueError(
                f"Expected argument ``sampling_strategy`` to be one of {allowed_sampling} but received {sampling_strategy}"
            )
        self.sampling_strategy = sampling_strategy

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the state of the base metric.

        Any tensor passed in will be bootstrapped along dimension 0.

        """
        args_sizes = apply_to_collection(args, paddle.Tensor, len)
        kwargs_sizes = apply_to_collection(kwargs, paddle.Tensor, len)
        if len(args_sizes) > 0:
            size = args_sizes[0]
        elif len(kwargs_sizes) > 0:
            size = next(iter(kwargs_sizes.values()))
        else:
            raise ValueError(
                "None of the input contained tensors, so could not determine the sampling size"
            )
        for idx in range(self.num_bootstraps):
            sample_idx = _bootstrap_sampler(
                size, sampling_strategy=self.sampling_strategy
            ).to(self.place)
            if sample_idx.size == 0:
                continue
            new_args = apply_to_collection(
                args,
                paddle.Tensor,
                paddle.index_select,
                axis=0,
                index=sample_idx,
            )
            new_kwargs = apply_to_collection(
                kwargs,
                paddle.Tensor,
                paddle.index_select,
                axis=0,
                index=sample_idx,
            )
            self.metrics[idx].update(*new_args, **new_kwargs)

    def compute(self) -> dict[str, paddle.Tensor]:
        """Compute the bootstrapped metric values.

        Always returns a dict of tensors, which can contain the following keys: ``mean``, ``std``, ``quantile`` and
        ``raw`` depending on how the class was initialized.

        """
        computed_vals = paddle.stack(
            [cast("Metric", m).compute() for m in self.metrics], axis=0
        )
        output_dict = {}
        if self.mean:
            output_dict["mean"] = computed_vals.mean(dim=0)
        if self.std:
            output_dict["std"] = computed_vals.std(axis=0)
        if self.quantile is not None:
            output_dict["quantile"] = paddle.quantile(
                computed_vals, self.quantile
            )
        if self.raw:
            output_dict["raw"] = computed_vals
        return output_dict

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the original forward method of the base metric class."""
        return super(WrapperMetric, self).forward(*args, **kwargs)

    def reset(self) -> None:
        """Reset the state of the base metric."""
        for m in self.metrics:
            m = cast("Metric", m)
            m.reset()
        super().reset()

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
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import paddle
            >>> from paddle.metric.wrappers import BootStrapper
            >>> from paddle.metric.regression import MeanSquaredError
            >>> metric = BootStrapper(MeanSquaredError(), num_bootstraps=20)
            >>> metric.update(
            ...     paddle.randn(
            ...         100,
            ...     ),
            ...     paddle.randn(
            ...         100,
            ...     ),
            ... )
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import paddle
            >>> from paddle.metric.wrappers import BootStrapper
            >>> from paddle.metric.regression import MeanSquaredError
            >>> metric = BootStrapper(MeanSquaredError(), num_bootstraps=20)
            >>> values = []
            >>> for _ in range(3):
            ...     values.append(
            ...         metric(
            ...             paddle.randn(
            ...                 100,
            ...             ),
            ...             paddle.randn(
            ...                 100,
            ...             ),
            ...         )
            ...     )
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
