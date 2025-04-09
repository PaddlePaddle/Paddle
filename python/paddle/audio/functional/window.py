# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import paddle

if TYPE_CHECKING:
    from paddle import Tensor

    from ..features.layers import _WindowLiteral


class WindowFunctionRegister:
    def __init__(self):
        self._functions_dict = {}

    def register(self, func=None):
        def add_subfunction(func):
            name = func.__name__
            self._functions_dict[name] = func
            return func

        return add_subfunction

    def get(self, name):
        return self._functions_dict[name]


window_function_register = WindowFunctionRegister()


@window_function_register.register()
def _cat(x: list[Tensor], data_type: str) -> Tensor:
    l = []
    for t in x:
        if np.isscalar(t) and not isinstance(t, str):
            l.append(paddle.to_tensor([t], data_type))
        else:
            l.append(paddle.to_tensor(t, data_type))
    return paddle.concat(l)


@window_function_register.register()
def _bartlett(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """
    Computes the Bartlett window.
    This function is consistent with scipy.signal.windows.bartlett().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype)
    M = paddle.to_tensor(M, dtype=dtype)
    w = paddle.where(
        paddle.less_equal(n, (M - 1) / 2.0),
        2.0 * n / (M - 1),
        2.0 - 2.0 * n / (M - 1),
    )

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _kaiser(
    M: int, beta: float, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute the Kaiser window.
    This function is consistent with scipy.signal.windows.kaiser().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    beta = paddle.to_tensor(beta, dtype=dtype)

    n = paddle.arange(0, M, dtype=dtype)
    M = paddle.to_tensor(M, dtype=dtype)
    alpha = (M - 1) / 2.0
    w = paddle.i0(
        beta * paddle.sqrt(1 - ((n - alpha) / alpha) ** 2.0)
    ) / paddle.i0(beta)

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _nuttall(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Nuttall window.
    This function is consistent with scipy.signal.windows.nuttall().
    """
    a = paddle.to_tensor(
        [0.3635819, 0.4891775, 0.1365995, 0.0106411], dtype=dtype
    )
    return _general_cosine(M, a=a, sym=sym, dtype=dtype)


@window_function_register.register()
def _acosh(x: Tensor | float) -> Tensor:
    if isinstance(x, float):
        return math.log(x + math.sqrt(x**2 - 1))
    return paddle.log(x + paddle.sqrt(paddle.square(x) - 1))


@window_function_register.register()
def _extend(M: int, sym: bool) -> bool:
    """Extend window by 1 sample if needed for DFT-even symmetry."""
    if not sym:
        return M + 1, True
    else:
        return M, False


@window_function_register.register()
def _len_guards(M: int) -> bool:
    """Handle small or incorrect window lengths."""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')

    return M <= 1


@window_function_register.register()
def _truncate(w: Tensor, needed: bool) -> Tensor:
    """Truncate window by 1 sample if needed for DFT-even symmetry."""
    if needed:
        return w[:-1]
    else:
        return w


@window_function_register.register()
def _general_gaussian(
    M: int, p, sig, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a window with a generalized Gaussian shape.
    This function is consistent with scipy.signal.windows.general_gaussian().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype) - (M - 1.0) / 2.0
    w = paddle.exp(-0.5 * paddle.abs(n / sig) ** (2 * p))

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _general_cosine(
    M: int, a: list[float], sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a generic weighted sum of cosine terms window.
    This function is consistent with scipy.signal.windows.general_cosine().
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)
    fac = paddle.linspace(-math.pi, math.pi, M, dtype=dtype)
    w = paddle.zeros((M,), dtype=dtype)
    for k in range(len(a)):
        w += a[k] * paddle.cos(k * fac)
    return _truncate(w, needs_trunc)


@window_function_register.register()
def _general_hamming(
    M: int, alpha: float, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a generalized Hamming window.
    This function is consistent with scipy.signal.windows.general_hamming()
    """
    return _general_cosine(M, [alpha, 1.0 - alpha], sym, dtype=dtype)


@window_function_register.register()
def _taylor(
    M: int, nbar=4, sll=30, norm=True, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a Taylor window.
    The Taylor window taper function approximates the Dolph-Chebyshev window's
    constant sidelobe level for a parameterized number of near-in sidelobes.
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)
    # Original text uses a negative sidelobe level parameter and then negates
    # it in the calculation of B. To keep consistent with other methods we
    # assume the sidelobe level parameter to be positive.
    B = 10 ** (sll / 20)
    A = _acosh(B) / math.pi
    s2 = nbar**2 / (A**2 + (nbar - 0.5) ** 2)
    ma = paddle.arange(1, nbar, dtype=dtype)

    Fm = paddle.empty((nbar - 1,), dtype=dtype)
    signs = paddle.empty_like(ma)
    signs[::2] = 1
    signs[1::2] = -1
    m2 = ma * ma
    for mi in range(len(ma)):
        number = signs[mi] * paddle.prod(
            1 - m2[mi] / s2 / (A**2 + (ma - 0.5) ** 2)
        )
        if mi == 0:
            denom = 2 * paddle.prod(1 - m2[mi] / m2[mi + 1 :])
        elif mi == len(ma) - 1:
            denom = 2 * paddle.prod(1 - m2[mi] / m2[:mi])
        else:
            denom = (
                2
                * paddle.prod(1 - m2[mi] / m2[:mi])
                * paddle.prod(1 - m2[mi] / m2[mi + 1 :])
            )

        Fm[mi] = number / denom

    def W(n):
        return 1 + 2 * paddle.matmul(
            Fm.unsqueeze(0),
            paddle.cos(2 * math.pi * ma.unsqueeze(1) * (n - M / 2.0 + 0.5) / M),
        )

    w = W(paddle.arange(0, M, dtype=dtype))

    # normalize (Note that this is not described in the original text [1])
    if norm:
        scale = 1.0 / W((M - 1) / 2)
        w *= scale
    w = w.squeeze()
    return _truncate(w, needs_trunc)


@window_function_register.register()
def _hamming(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Hamming window.
    The Hamming window is a taper formed by using a raised cosine with
    non-zero endpoints, optimized to minimize the nearest side lobe.
    """
    return _general_hamming(M, 0.54, sym, dtype=dtype)


@window_function_register.register()
def _hann(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Hann window.
    The Hann window is a taper formed by using a raised cosine or sine-squared
    with ends that touch zero.
    """
    return _general_hamming(M, 0.5, sym, dtype=dtype)


@window_function_register.register()
def _tukey(
    M: int, alpha=0.5, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a Tukey window.
    The Tukey window is also known as a tapered cosine window.
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)

    if alpha <= 0:
        return paddle.ones((M,), dtype=dtype)
    elif alpha >= 1.0:
        return _hann(M, sym=sym)

    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype)
    width = int(alpha * (M - 1) / 2.0)
    n1 = n[0 : width + 1]
    n2 = n[width + 1 : M - width - 1]
    n3 = n[M - width - 1 :]

    w1 = 0.5 * (1 + paddle.cos(math.pi * (-1 + 2.0 * n1 / alpha / (M - 1))))
    w2 = paddle.ones(n2.shape, dtype=dtype)
    w3 = 0.5 * (
        1
        + paddle.cos(math.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (M - 1)))
    )
    w = paddle.concat([w1, w2, w3])

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _gaussian(
    M: int, std: float, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute a Gaussian window.
    The Gaussian widows has a Gaussian shape defined by the standard deviation(std).
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(0, M, dtype=dtype) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = paddle.exp(-(n**2) / sig2)

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _exponential(
    M: int, center=None, tau=1.0, sym: bool = True, dtype: str = 'float64'
) -> Tensor:
    """Compute an exponential (or Poisson) window."""
    if sym and center is not None:
        raise ValueError("If sym==True, center must be None.")
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    if center is None:
        center = (M - 1) / 2

    n = paddle.arange(0, M, dtype=dtype)
    w = paddle.exp(-paddle.abs(n - center) / tau)

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _triang(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a triangular window."""
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    n = paddle.arange(1, (M + 1) // 2 + 1, dtype=dtype)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = paddle.concat([w, w[::-1]])
    else:
        w = 2 * n / (M + 1.0)
        w = paddle.concat([w, w[-2::-1]])

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _bohman(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Bohman window.
    The Bohman window is the autocorrelation of a cosine window.
    """
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)

    fac = paddle.abs(paddle.linspace(-1, 1, M, dtype=dtype)[1:-1])
    w = (1 - fac) * paddle.cos(math.pi * fac) + 1.0 / math.pi * paddle.sin(
        math.pi * fac
    )
    w = _cat([0, w, 0], dtype)

    return _truncate(w, needs_trunc)


@window_function_register.register()
def _blackman(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a Blackman window.
    The Blackman window is a taper formed by using the first three terms of
    a summation of cosines. It was designed to have close to the minimal
    leakage possible.  It is close to optimal, only slightly worse than a
    Kaiser window.
    """
    return _general_cosine(M, [0.42, 0.50, 0.08], sym, dtype=dtype)


@window_function_register.register()
def _cosine(M: int, sym: bool = True, dtype: str = 'float64') -> Tensor:
    """Compute a window with a simple cosine shape."""
    if _len_guards(M):
        return paddle.ones((M,), dtype=dtype)
    M, needs_trunc = _extend(M, sym)
    w = paddle.sin(math.pi / M * (paddle.arange(0, M, dtype=dtype) + 0.5))

    return _truncate(w, needs_trunc)


def get_window(
    window: _WindowLiteral | tuple[_WindowLiteral, float],
    win_length: int,
    fftbins: bool = True,
    dtype: str = 'float64',
) -> Tensor:
    """Return a window of a given length and type.

    Args:
        window (Union[str, Tuple[str, float]]): The window function applied to the signal before the Fourier transform. Supported window functions: 'hamming', 'hann', 'gaussian', 'general_gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor', 'bartlett', 'kaiser', 'nuttall'.
        win_length (int): Number of samples.
        fftbins (bool, optional): If True, create a "periodic" window. Otherwise, create a "symmetric" window, for use in filter design. Defaults to True.
        dtype (str, optional): The data type of the return window. Defaults to 'float64'.

    Returns:
        Tensor: The window represented as a tensor.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> n_fft = 512
            >>> cosine_window = paddle.audio.functional.get_window('cosine', n_fft)

            >>> std = 7
            >>> gaussian_window = paddle.audio.functional.get_window(('gaussian', std), n_fft)
    """
    sym = not fftbins
    args = ()
    if isinstance(window, tuple):
        winstr = window[0]
        if len(window) > 1:
            args = window[1:]
    elif isinstance(window, str):
        if window in ['gaussian', 'exponential', 'kaiser']:
            raise ValueError(
                "The '" + window + "' window needs one or "
                "more parameters -- pass a tuple."
            )
        else:
            winstr = window
    else:
        raise ValueError(f"{type(window)} as window type is not supported.")

    try:
        winfunc = window_function_register.get('_' + winstr)
    except KeyError as e:
        raise ValueError("Unknown window type.") from e
    params = (win_length, *args)
    kwargs = {'sym': sym}
    return winfunc(*params, dtype=dtype, **kwargs)
