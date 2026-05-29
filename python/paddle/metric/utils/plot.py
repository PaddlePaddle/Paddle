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

from collections.abc import Generator, Sequence
from itertools import product
from math import ceil, floor, sqrt
from typing import Any, Union, no_type_check

import numpy as np

import paddle
from paddle.metric.utils.imports import (
    _LATEX_AVAILABLE,
    _MATPLOTLIB_AVAILABLE,
    _SCIENCEPLOT_AVAILABLE,
)

if _MATPLOTLIB_AVAILABLE:
    import matplotlib
    import matplotlib.axes
    import matplotlib.pyplot as plt

    _PLOT_OUT_TYPE = tuple[plt.Figure, Union[matplotlib.axes.Axes, np.ndarray]]
    _AX_TYPE = matplotlib.axes.Axes
    _CMAP_TYPE = Union[matplotlib.colors.Colormap, str]
    style_change = plt.style.context
else:
    _PLOT_OUT_TYPE = tuple[object, object]
    _AX_TYPE = object
    _CMAP_TYPE = object
    from contextlib import contextmanager

    @contextmanager
    def style_change(*args: Any, **kwargs: Any) -> Generator:
        """No-ops decorator if matplotlib is not installed."""
        yield


if _SCIENCEPLOT_AVAILABLE:
    _style = ["science", "no-latex"]
_style = (
    ["science"] if _SCIENCEPLOT_AVAILABLE and _LATEX_AVAILABLE else ["default"]
)


def _error_on_missing_matplotlib() -> None:
    """Raise error if matplotlib is not installed."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ModuleNotFoundError(
            "Plot function expects `matplotlib` to be installed. Please install with `pip install matplotlib`"
        )


@style_change(_style)
def plot_single_or_multi_val(
    val: paddle.Tensor
    | Sequence[paddle.Tensor]
    | dict[str, paddle.Tensor]
    | Sequence[dict[str, paddle.Tensor]],
    ax: _AX_TYPE | None = None,
    higher_is_better: bool | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    legend_name: str | None = None,
    name: str | None = None,
) -> _PLOT_OUT_TYPE:
    """Plot a single metric value or multiple, including bounds of value if existing.

    Args:
        val: A single tensor with one or multiple values (multiclass/label/output format) or a list of such tensors.
            If a list is provided the values are interpreted as a time series of evolving values.
        ax: Axis from a figure.
        higher_is_better: Indicates if a label indicating where the optimal value it should be added to the figure
        lower_bound: lower value that the metric can take
        upper_bound: upper value that the metric can take
        legend_name: for class based metrics specify the legend prefix e.g. Class or Label to use when multiple values
            are provided
        name: Name of the metric to use for the y-axis label

    Returns:
        A tuple consisting of the figure and respective ax objects of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed

    """
    _error_on_missing_matplotlib()
    fig, ax = plt.subplots() if ax is None else (None, ax)
    ax.get_xaxis().set_visible(False)
    if isinstance(val, paddle.Tensor):
        if val.size == 1:
            ax.plot([val.detach().cpu()], marker="o", markersize=10)
        else:
            for i, v in enumerate(val):
                label = f"{legend_name} {i}" if legend_name else f"{i}"
                ax.plot(
                    i,
                    v.detach().cpu(),
                    marker="o",
                    markersize=10,
                    linestyle="None",
                    label=label,
                )
    elif isinstance(val, dict):
        for i, (k, v) in enumerate(val.items()):
            if v.size != 1:
                ax.plot(
                    v.detach().cpu(),
                    marker="o",
                    markersize=10,
                    linestyle="-",
                    label=k,
                )
                ax.get_xaxis().set_visible(True)
                ax.set_xlabel("Step")
                ax.set_xticks(paddle.arange(len(v)))
            else:
                ax.plot(i, v.detach().cpu(), marker="o", markersize=10, label=k)
    elif isinstance(val, Sequence):
        n_steps = len(val)
        if isinstance(val[0], dict):
            val = {
                k: paddle.stack([val[i][k] for i in range(n_steps)])
                for k in val[0]
            }
            for k, v in val.items():
                ax.plot(
                    v.detach().cpu(),
                    marker="o",
                    markersize=10,
                    linestyle="-",
                    label=k,
                )
        else:
            val = paddle.stack(val, 0)
            multi_series = val.ndim != 1
            val = val.T if multi_series else val.unsqueeze(0)
            for i, v in enumerate(val):
                label = (
                    (f"{legend_name} {i}" if legend_name else f"{i}")
                    if multi_series
                    else ""
                )
                ax.plot(
                    v.detach().cpu(),
                    marker="o",
                    markersize=10,
                    linestyle="-",
                    label=label,
                )
        ax.get_xaxis().set_visible(True)
        ax.set_xlabel("Step")
        ax.set_xticks(paddle.arange(n_steps))
    else:
        raise ValueError("Got unknown format for argument `val`.")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
    ylim = ax.get_ylim()
    if lower_bound is not None and upper_bound is not None:
        factor = 0.1 * (upper_bound - lower_bound)
    else:
        factor = 0.1 * (ylim[1] - ylim[0])
    ax.set_ylim(
        bottom=lower_bound - factor
        if lower_bound is not None
        else ylim[0] - factor,
        top=upper_bound + factor
        if upper_bound is not None
        else ylim[1] + factor,
    )
    ax.grid(True)
    ax.set_ylabel(name if name is not None else None)
    xlim = ax.get_xlim()
    factor = 0.1 * (xlim[1] - xlim[0])
    y_lines = []
    if lower_bound is not None:
        y_lines.append(lower_bound)
    if upper_bound is not None:
        y_lines.append(upper_bound)
    ax.hlines(y_lines, xlim[0], xlim[1], linestyles="dashed", colors="k")
    if higher_is_better is not None:
        if lower_bound is not None and not higher_is_better:
            ax.set_xlim(xlim[0] - factor, xlim[1])
            ax.text(
                xlim[0],
                lower_bound,
                s="Optimal \n value",
                horizontalalignment="center",
                verticalalignment="center",
            )
        if upper_bound is not None and higher_is_better:
            ax.set_xlim(xlim[0] - factor, xlim[1])
            ax.text(
                xlim[0],
                upper_bound,
                s="Optimal \n value",
                horizontalalignment="center",
                verticalalignment="center",
            )
    return fig, ax


def _get_col_row_split(n: int) -> tuple[int, int]:
    """Split `n` figures into `rows` x `cols` figures."""
    nsq = sqrt(n)
    if int(nsq) == nsq:
        return int(nsq), int(nsq)
    if floor(nsq) * ceil(nsq) >= n:
        return floor(nsq), ceil(nsq)
    return ceil(nsq), ceil(nsq)


def _get_text_color(patch_color: tuple[float, float, float, float]) -> str:
    """Get the text color for a given value and colormap.

    Following Wikipedia's recommendations: https://en.wikipedia.org/wiki/Relative_luminance.

    Args:
        patch_color: RGBA color tuple

    """
    r, g, b, a = patch_color
    r, g, b = (
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in (r, g, b)
    )
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return ".1" if y > 0.4 else "white"


def trim_axs(axs: _AX_TYPE | np.ndarray, nb: int) -> np.ndarray | _AX_TYPE:
    """Reduce `axs` to `nb` Axes.

    All further Axes are removed from the figure.

    """
    if isinstance(axs, _AX_TYPE):
        return axs
    axs = axs.flat
    for ax in axs[nb:]:
        ax.remove()
    return axs[:nb]


@style_change(_style)
@no_type_check
def plot_confusion_matrix(
    confmat: paddle.Tensor,
    ax: _AX_TYPE | None = None,
    add_text: bool = True,
    labels: list[int | str] | None = None,
    cmap: _CMAP_TYPE | None = None,
) -> _PLOT_OUT_TYPE:
    """Plot an confusion matrix.

    Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/confusion_matrix.py.
    Works for both binary, multiclass and multilabel confusion matrices.

    Args:
        confmat: the confusion matrix. Either should be an [N,N] matrix in the binary and multiclass cases or an
            [N, 2, 2] matrix for multilabel classification
        ax: Axis from a figure. If not provided, a new figure and axis will be created
        add_text: if text should be added to each cell with the given value
        labels: labels to add the x- and y-axis
        cmap: matplotlib colormap to use for the confusion matrix
            https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed

    """
    _error_on_missing_matplotlib()
    if confmat.ndim == 3:
        nb, n_classes = confmat.shape[0], 2
        rows, cols = _get_col_row_split(nb)
    else:
        nb, n_classes, rows, cols = 1, confmat.shape[0], 1, 1
    if labels is not None and confmat.ndim != 3 and len(labels) != n_classes:
        raise ValueError(
            f"Expected number of elements in arg `labels` to match number of labels in confmat but got {len(labels)} and {n_classes}"
        )
    if confmat.ndim == 3:
        fig_label = labels or np.arange(nb)
        labels = list(map(str, range(n_classes)))
    else:
        fig_label = None
        labels = labels or np.arange(n_classes).tolist()
    fig, axs = (
        plt.subplots(nrows=rows, ncols=cols, constrained_layout=True)
        if ax is None
        else (ax.get_figure(), ax)
    )
    axs = trim_axs(axs, nb)
    for i in range(nb):
        ax = axs[i] if rows != 1 or cols != 1 else axs
        if fig_label is not None:
            ax.set_title(f"Label {fig_label[i]}", fontsize=15)
        im = ax.imshow(
            confmat[i].cpu().detach()
            if confmat.ndim == 3
            else confmat.cpu().detach(),
            cmap=cmap,
        )
        if i // cols == rows - 1:
            ax.set_xlabel("Predicted class", fontsize=15)
        if i % cols == 0:
            ax.set_ylabel("True class", fontsize=15)
        ax.set_xticks(list(range(n_classes)))
        ax.set_yticks(list(range(n_classes)))
        ax.set_xticklabels(labels, rotation=45, fontsize=10)
        ax.set_yticklabels(labels, rotation=25, fontsize=10)
        if add_text:
            for ii, jj in product(range(n_classes), range(n_classes)):
                val = (
                    confmat[i, ii, jj] if confmat.ndim == 3 else confmat[ii, jj]
                )
                patch_color = im.cmap(im.norm(val.item()))
                c = _get_text_color(patch_color)
                ax.text(
                    jj,
                    ii,
                    str(round(val.item(), 2)),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=c,
                )
    return fig, axs


@style_change(_style)
def plot_curve(
    curve: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]
    | tuple[list[paddle.Tensor], list[paddle.Tensor], list[paddle.Tensor]],
    score: paddle.Tensor | None = None,
    ax: _AX_TYPE | None = None,
    label_names: tuple[str, str] | None = None,
    legend_name: str | None = None,
    name: str | None = None,
    labels: list[int | str] | None = None,
) -> _PLOT_OUT_TYPE:
    """Inspired by: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/roc_curve.py.

    Plots a curve object

    Args:
        curve: a tuple of (x, y, t) where x and y are the coordinates of the curve and t are the thresholds used
            to compute the curve
        score: optional area under the curve added as label to the plot
        ax: Axis from a figure
        label_names: Tuple containing the names of the x and y axis
        legend_name: Name of the curve to be used in the legend
        name: Custom name to describe the metric
        labels: Optional labels for the different curves that will be added to the plot

    Returns:
        A tuple consisting of the figure and respective ax objects (or array of ax objects) of the generated figure

    Raises:
        ModuleNotFoundError:
            If `matplotlib` is not installed
        ValueError:
            If `curve` does not have 3 elements, being in the wrong format
    """
    if len(curve) < 2:
        raise ValueError(
            f"Expected 2 or 3 elements in curve but got {len(curve)}"
        )
    x, y = curve[:2]
    _error_on_missing_matplotlib()
    fig, ax = plt.subplots() if ax is None else (None, ax)
    if (
        isinstance(x, paddle.Tensor)
        and isinstance(y, paddle.Tensor)
        and x.ndim == 1
        and y.ndim == 1
    ):
        label = f"AUC={score.item():0.3f}" if score is not None else None
        ax.plot(
            x.detach().cpu(),
            y.detach().cpu(),
            linestyle="-",
            linewidth=2,
            label=label,
        )
        if label is not None:
            ax.legend()
    elif (
        isinstance(x, list)
        and isinstance(y, list)
        or isinstance(x, paddle.Tensor)
        and isinstance(y, paddle.Tensor)
        and x.ndim == 2
        and y.ndim == 2
    ):
        n_classes = len(x)
        if labels is not None and len(labels) != n_classes:
            raise ValueError(
                f"Expected number of elements in arg `labels` to match number of labels in roc curves but got {len(labels)} and {n_classes}"
            )
        for i, (x_, y_) in enumerate(zip(x, y)):
            label = (
                f"{legend_name}_{i}"
                if legend_name is not None
                else str(i)
                if labels is None
                else str(labels[i])
            )
            label += f" AUC={score[i].item():0.3f}" if score is not None else ""
            ax.plot(
                x_.detach().cpu(),
                y_.detach().cpu(),
                linestyle="-",
                linewidth=2,
                label=label,
            )
            ax.legend()
    else:
        raise ValueError(
            f"Unknown format for argument `x` and `y`. Expected either list or tensors but got {type(x)} and {type(y)}."
        )
    if label_names is not None:
        ax.set_xlabel(label_names[0])
        ax.set_ylabel(label_names[1])
    ax.grid(True)
    ax.set_title(name)
    return fig, ax
