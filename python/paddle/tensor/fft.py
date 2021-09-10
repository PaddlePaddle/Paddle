# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Sequence
import numpy as np
import paddle
from .attribute import is_complex, is_floating_point, is_interger, _real_to_complex_dtype, _complex_to_real_dtype
from ..fluid.framework import in_dygraph_mode
from .. import _C_ops
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.layer_helper import LayerHelper


def _check_normalization(norm):
    if norm not in ['forward', 'backward', 'ortho']:
        raise ValueError(
            "Unexpected norm: {}. Norm should be forward, backward or ortho".
            format(norm))


def _check_fft_n(n):
    if not isinstance(n, int):
        raise ValueError(
            "Invalid FFT argument n({}), it shoule be an integer.".format(n))
    if n <= 0:
        raise ValueError(
            "Invalid FFT argument n({}), it should be positive.".format(n))


def _check_fft_shape(x, s):
    ndim = x.ndim
    if not isinstance(s, Sequence):
        raise ValueError(
            "Invaid FFT argument s({}), it should be a sequence of integers.")

    if len(s) > ndim:
        raise ValueError(
            "Length of FFT argument s should not be larger than the rank of input. "
            "Received s: {}, rank of x: {}".format(s, ndim))
    for size in s:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("FFT sizes {} contains invalid value ({})".format(
                s, size))


def _check_fft_axis(x, axis):
    ndim = x.ndim
    if not isinstance(axis, int):
        raise ValueError(
            "Invalid FFT axis ({}), it shoule be an integer.".format(axis))
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            "Invalid FFT axis ({}), it should be in range [-{}, {})".format(
                axis, ndim, ndim))


def _check_fft_axes(x, axes):
    ndim = x.ndim
    if not isinstance(axes, Sequence):
        raise ValueError(
            "Invalid FFT axes ({}), it should be a sequence of integers.".
            format(axes))
    if len(axes) > ndim:
        raise ValueError(
            "Length of fft axes should not be larger than the rank of input. "
            "Received, len of axes: {}, rank of x: {}".format(len(axes), ndim))
    for axis in axes:
        if not isinstance(axis, int) or axis < -ndim or axis >= ndim:
            raise ValueError(
                "FFT axes {} contains invalid value ({}), it should be in range [-{}, {})".
                format(axes, axis, ndim, ndim))


def _resize_fft_input(x, s, axes):
    if len(s) != len(axes):
        raise ValueError("length of `s` should equals length of `axes`.")
    shape = x.shape
    ndim = x.ndim

    axes_to_pad = []
    paddings = []
    axes_to_slice = []
    slices = []
    for i, axis in enumerate(axes):
        if shape[axis] < s[i]:
            axes_to_pad.append(axis)
            paddings.append(s[i] - shape[axis])
        elif shape[axis] > s[i]:
            axes_to_slice.append(axis)
            slices.append((0, s[i]))

    if axes_to_slice:
        x = paddle.slice(
            x,
            axes_to_slice,
            starts=[item[0] for item in slices],
            ends=[item[1] for item in slices])
    if axes_to_pad:
        padding_widths = [0] * (2 * ndim)
        for axis, pad in zip(axes_to_pad, paddings):
            padding_widths[2 * axis + 1] = pad
        x = paddle.nn.functional.pad(x, padding_widths)
    return x


def _normalize_axes(x, axes):
    ndim = x.ndim
    return [item if item >= 0 else (item + ndim) for item in axes]


def _check_at_least_ndim(x, rank):
    if x.ndim < rank:
        raise ValueError("The rank of the input ({}) should >= {}".format(
            x.ndim, rank))


# public APIs 1d
def fft(x, n=None, axis=-1, norm="backward", name=None):
    if not is_complex(x):
        return fft_r2c(
            x, n, axis, norm, forward=True, onesided=False, name=name)
    else:
        return fft_c2c(x, n, axis, norm, forward=True, name=name)


def ifft(x, n=None, axis=-1, norm="backward", name=None):
    if not is_complex(x):
        return fft_r2c(
            x, n, axis, norm, forward=False, onesided=False, name=name)
    else:
        return fft_c2c(x, n, axis, norm, forward=False, name=name)


def rfft(x, n=None, axis=-1, norm="backward", name=None):
    """
    The one dimensional FFT for real input.

    This function computes the one dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued tensor by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric. This function does not compute the negative frequency 
    terms, and the length of the transformed axis of the output is therefore 
    ``n//2 + 1``.

    Args:
        x(Tensor) : Real-valued input tensor 
        n(int, optional): Number of points along transformation axis in the 
            input to use. If `n` is smaller than the length of the input, the 
            input is cropped. If it is larger, the input is padded with zeros. 
            If `n` is not given, the length of the input along the axis 
            specified by `axis` is used.
        axis(int, optional): Axis over which to compute the FFT. Default value 
            is last axis.
        norm(str, optional) : Normalization mode, indicates which direction of 
            the forward/backward  pair of transforms is scaled and with what 
            normalization factor. Include {"backward", "ortho", "forward"}, 
            default value is "backward".
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns:
        out(Tensor) : complex tensor

    Raises:


    Examples:
    .. code-block:: python
        import paddle

        x = paddle.to_tensor([0.0, 1.0, 0.0, 0.0])
        print(paddle.fft.rfft(x))
        # Tensor(shape=[3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #        [ (1+0j), -1j    , (-1+0j)])
    """
    return fft_r2c(x, n, axis, norm, forward=True, onesided=True, name=name)


def irfft(x, n=None, axis=-1, norm="backward", name=None):
    """
    Computes the inverse of `rfft`.

    This function calculates the inverse of the one-dimensional *n* point discrete 
    Fourier transform of the actual input calculated by "rfft". In other words, 
    ``irfft(rfft(a),len(a)) == a`` is within the numerical accuracy range.

    The input shall be in the form of "rfft", i.e. the actual zero frequency term, 
    followed by the complex positive frequency term, in the order of increasing frequency. 
    Because the discrete Fourier transform of the actual input is Hermite symmetric, 
    the negative frequency term is regarded as the complex conjugate term of the corresponding 
    positive frequency term.

    Args:
        x (Tensor): The input data. It's a Tensor type. Data type: float32, float64.
        n (int, optional): The length of the output transform axis. For `n` output
            points, ``n//2 + 1``input points are necessary. If the length of the input tensor is greater 
            than `n`, it will be cropped, if it is shorter than this, fill in zero. If `n` is not given, 
            it is considered to be ``2 * (k-1)``, where ``k`` is the length of the input axis specified 
            along the ` axis'.
        axis (int, optional): Axis used to calculate FFT. If not specified, the last axis 
            is used by default.       
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 

    Returns:
        Real tensor. Truncated or zero fill input for the transformation along the axis indicated by 
        `axis`, or the last input if `axis` is not specified. The length of the conversion axis 
        is `n`, or ``2 * k-2``, if `k` is None, where `k` is the length of the input conversion axis. 
        If the output is an odd number, you need to specify the value of 'n', such as ``2 * k-1``
        in some cases.
    
    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([1, 2, 3, 4, 3, 2])
            xp = paddle.to_tensor(x)
            irfft_xp = paddle.tensor.fft.irfft(xp).numpy()
            print(irfft_xp)
            #  [15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]

    """
    return fft_c2r(x, n, axis, norm, forward=False, name=name)


def hfft(x, n=None, axis=-1, norm="backward", name=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, a real
    spectrum.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        n (int, optional): The length of the output transform axis. For `n` output
            points, ``n//2 + 1`` input points are necessary. If the length of the input tensor is greater 
            than `n`, it will be cropped, if it is shorter than this, fill in zero. If `n` is not given, 
            it is considered to be ``2 * (k-1)``, where ``k`` is the length of the input axis specified 
            along the ` axis'.
        axis (int,optional): Axis used to calculate FFT. If not specified, the last axis 
            is used by default.       
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 

    Returns:
        Real tensor. Truncated or zero fill input for the transformation along the axis indicated by 
        `axis`, or the last input if `axis` is not specified. The length of the conversion axis 
        is `n`, or ``2 * k-2``, if `k` is None, where `k` is the length of the input conversion axis. 
        If the output is an odd number, you need to specify the value of 'n', such as ``2 * k-1`` in 
        some cases.
    
    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([1, 2, 3, 4, 3, 2])
            xp = paddle.to_tensor(x)
            hfft_xp = paddle.tensor.fft.hfft(xp).numpy()
            print(hfft_xp)
            #  [15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]
    """

    return fft_c2r(x, n, axis, norm, forward=True, name=name)


def ihfft(x, n=None, axis=-1, norm="backward", name=None):
    """
    The inverse FFT of a signal that has Hermitian symmetry.

    This function computes the one dimensional *n*-point inverse FFT of a signal 
    that has Hermitian symmetry by means of an efficient algorithm called 
    the Fast Fourier Transform (FFT).

    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric. This function does not compute the negative frequency 
    terms, and the length of the transformed axis of the output is therefore 
    ``n//2 + 1``.

    Args:
        x(Tensor): Input tensor.
        n(int, optional): The number of points along transformation axis in the 
            input to use.  If `n` is smaller than the length of the input, the 
            input is cropped.  If it is larger, the input is padded with zeros. 
            If `n` is not given, the length of the input along the axis 
            specified by `axis` is used.
        axis(int, optional) : Axis over which to compute the inverse FFT. If not
            given, the last axis is used.
        norm(str, optional) : Normalization mode, indicates which direction of 
            the forward/backward pair of transforms is scaled and with what 
            normalization factor. Include {"backward", "ortho", "forward"}, 
            default value is "backward".
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns:
        out(Tensor) : complex tensor.

    Examples:
    .. code-block:: python
        import paddle 

        spectrum = paddle.to_tensor([10.0, -5.0, 0.0, -1.0, 0.0, -5.0])
        print(paddle.fft.ifft(spectrum))
        # Tensor(shape=[6], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #       [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j), (2.3333334922790527+1.9868215517249155e-08j),  (1+1.9868215517249155e-08j)])
        print(paddle.fft.ihfft(spectrum))
        #  Tensor(shape = [4], dtype = complex64, place = CUDAPlace(0), stop_gradient = True,
        #         [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j)])

    """
    return fft_r2c(x, n, axis, norm, forward=False, onesided=True, name=name)


# public APIs nd
def fftn(x, s=None, axes=None, norm="backward", name=None):
    if not is_complex(x):
        return fftn_r2c(
            x, s, axes, norm, forward=True, onesided=False, name=name)
    else:
        return fftn_c2c(x, s, axes, norm, forward=True, name=name)


def ifftn(x, s=None, axes=None, norm="backward", name=None):
    if not is_complex(x):
        return fftn_r2c(
            x, s, axes, norm, forward=False, onesided=False, name=name)
    else:
        return fftn_c2c(x, s, axes, norm, forward=False, name=name)


def rfftn(x, s=None, axes=None, norm="backward", name=None):
    """
    The N dimensional FFT for real input.

    This function computes the N-dimensional discrete Fourier Transform over
    any number of axes in an M-dimensional real array by means of the Fast
    Fourier Transform (FFT).  By default, all axes are transformed, with the
    real transform performed over the last axis, while the remaining
    transforms are complex.

    The transform for real input is performed over the last transformation
    axis, as by `rfft`, then the transform over the remaining axes is
    performed as by `fftn`.  The order of the output is as for `rfft` for the
    final transformation axis, and as for `fftn` for the remaining
    transformation axes.

    Args:
        x(Tensor) : Input tensor, taken to be real.
        s(Sequence[int]) : Shape to use from the exec fft. The final element of 
            `s` corresponds to `n` for ``rfft(x, n)``, while for the remaining 
            axes, it corresponds to `n` for ``fft(x, n)``. Along any axis, if 
            the given shape is smaller than that of the input, the input is 
            cropped.  If it is larger, the input is padded with zeros. if `s` is 
            not given, the shape of the input along the axes specified by `axes` 
            is used.
        axes(Sequence[int]) : Axes over which to compute the FFT.  If not given, 
            the last ``len(s)`` axes are used, or all axes if `s` is also not 
            specified.
        norm(str, optional) : Normalization mode, indicates which direction of 
            the forward/backward pair of transforms is scaled and with what 
            normalization factor. Include {"backward", "ortho", "forward"}, 
            default value is "backward".
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns:
        out(Tensor): complex tensor


    Raises:
        ValueError: If `s` and `axes` have different length.

    Examples:
    .. code-block:: python
        import paddle

        # default, all axis will be used to exec fft
        x = paddle.ones((2, 3, 4))
        print(paddle.fft.rfftn(x))
        # Tensor(shape=[2, 3, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[(24+0j), 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ]],
        #
        #         [[0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ]]])

        # use axes(2, 0)
        print(paddle.fft.rfftn(x, axes=(2, 0)))
        # Tensor(shape=[2, 3, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[(24+0j), 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ]],
        #
        #         [[0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ],
        #          [0j     , 0j     , 0j     ]]])

    """
    return fftn_r2c(x, s, axes, norm, forward=True, onesided=True, name=name)


def irfftn(x, s=None, axes=None, norm="backward", name=None):
    """
    Computes the inverse of `rfftn`.

    This function computes the inverse of the N-D discrete
    Fourier Transform for real input over any number of axes in an
    M-D array by means of the Fast Fourier Transform (FFT). In
    other words, ``irfftn(rfftn(x), x.shape) == x`` to within numerical
    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
    and for the same reason.)

    The input should be ordered in the same way as is returned by `rfftn`,
    i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
    along all the other axes.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints, optional): The length of the output transform axis. 
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
            number of input points used along this axis, except for the last axis,
            where ``s[-1]//2+1`` points of the input are used. Along any axis, if 
            the shape indicated by `s` is smaller than that of the input, the input 
            is cropped. If it is larger, the input is padded with zeros. 
            If `s` is not given, the shape of the input along the axes specified by axes 
            is used. Except for the last axis which is taken to be ``2*(k-1)`` where 
            ``k`` is the length of the input along that axis.
        axis (sequence of ints, optional): Axes over which to compute the inverse FFT. If not given, the last
            `len(s)` axes are used, or all axes if `s` is also not specified.      
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 
    
    Returns:
        Real tensor. The truncated or zero-padded input, transformed along the axes indicated by `axes`, 
        or by a combination of `s` or `x`, as explained in the parameters section above. The length of 
        each transformed axis is as given by the corresponding element of `s`, or the length of the input
        in every axis except for the last one if `s` is not given. In the final transformed axis the length
        of the output when `s` is not given is ``2*(m-1)``, where ``m`` is the length of the final 
        transformed axis of the input. To get an odd number of output points in the final axis, 
        `s` must be specified.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.zeros((3, 2, 2))
            x[0, 0, 0] = 3 * 2 * 2
            xp = paddle.to_tensor(x)
            irfftn_xp = paddle.tensor.fft.irfftn(xp).numpy()
            print(irfftn_xp)
            #  [[[1.,  1.],
            #    [1.,  1.]],
            #   [[1.,  1.],
            #    [1.,  1.]],
            #   [[1.,  1.],
            #   [1.,  1.]]]
    
    """
    return fftn_c2r(x, s, axes, norm, forward=False, name=name)


def hfftn(x, s=None, axes=None, norm="backward", name=None):
    """
    Compute the N-D FFT of Hermitian symmetric complex input, i.e., a
    signal with a real spectrum.

    This function calculates the n-D discrete Fourier transform of Hermite symmetric 
    complex input on any axis in M-D array by fast Fourier transform (FFT). 
    In other words, ``ihfftn(hfftn(x, s)) == x is within the numerical accuracy range. 
    (``s`` here are ``x.shape`` and ``s[-1] = x.shape[- 1] * 2 - 1``. This is necessary 
    for the same reason that ``irfft` requires ``x.shape``.)

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints, optional): The length of the output transform axis. 
            (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
            number of input points used along this axis, except for the last axis,
            where ``s[-1]//2+1`` points of the input are used. Along any axis, if 
            the shape indicated by `s` is smaller than that of the input, the input 
            is cropped. If it is larger, the input is padded with zeros. 
            If `s` is not given, the shape of the input along the axes specified by axes 
            is used. Except for the last axis which is taken to be ``2*(k-1)`` where 
            ``k`` is the length of the input along that axis.
        axis (sequence of ints, optional): Axes over which to compute the inverse FFT. If not given, the last
            `len(s)` axes are used, or all axes if `s` is also not specified.      
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 
    
    Returns:
        Real tensor. Truncate or zero fill input, transforming along the axis indicated by axis or 
        a combination of `s` or `X`.
    
    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([1, 2, 3, 4, 3, 2])
            xp = paddle.to_tensor(x)
            hfftn_xp = paddle.tensor.fft.hfftn(xp).numpy()
            print(hfftn_xp)
            #  [15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]


    """
    return fftn_c2r(x, s, axes, norm, forward=True, name=name)


def ihfftn(x, s=None, axes=None, norm="backward", name=None):
    """
    The n dimensional inverse FFT of a signal that has Hermitian symmetry.

    This function computes the n dimensional inverse FFT over any number of axes 
    in an M-dimensional of a signal that has Hermitian symmetry by means of an 
    efficient algorithm called the Fast Fourier Transform (FFT).

    Args:
        x(Tensor): Input tensor.
        s(Sequence[int], optional) : Shape (length along each transformed axis) 
            to use from the input. (``s[0]`` refers to axis 0, ``s[1]`` to axis 
            1, etc.). Along any axis, if the given shape is smaller than that 
            of the input, the input is cropped. If it is larger, the input is 
            padded with zeros. if `s` is not given, the shape of the input 
            along the axes specified by `axes` is used.
        axis(Sequence[int], optional) : Axis over which to compute the inverse FFT. If not
            given, the last axis is used.
        norm(str, optional) : Normalization mode, indicates which direction of 
            the forward/backward pair of transforms is scaled and with what 
            normalization factor. Include {"backward", "ortho", "forward"}, 
            default value is "backward".
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns:
        out(Tensor) : complex tensor.

    Examples:
    .. code-block:: python
        import paddle 

        spectrum = paddle.to_tensor([10.0, -5.0, 0.0, -1.0, 0.0, -5.0])
        print(paddle.fft.ifft(spectrum))
        # Tensor(shape=[6], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #       [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j), (2.3333334922790527+1.9868215517249155e-08j),  (1+1.9868215517249155e-08j)])
        print(paddle.fft.ihfft(spectrum))
        #  Tensor(shape = [4], dtype = complex64, place = CUDAPlace(0), stop_gradient = True,
        #         [(-0.1666666716337204+0j),  (1-1.9868215517249155e-08j), (2.3333334922790527-1.9868215517249155e-08j),  (3.5+0j)])

    """
    return fftn_r2c(x, s, axes, norm, forward=False, onesided=True, name=name)


# public APIs 2d
def fft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return fftn(x, s, axes, norm, name)


def ifft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return ifftn(x, s, axes, norm, name)


def rfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    """
    The two dimensional FFT with real tensor input.

    This is really just `rfftn` with different default behavior.
    For more details see `rfftn`.

    Args:
        x(Tensor): Input tensor, taken to be real.
        s(Sequence[int]) : Shape of the FFT.
        axes(Sequence[int], optional): Axes over which to compute the FFT.
        norm(str, optional) : {"backward", "ortho", "forward"}, 
            default is "backward". Indicates which direction of the 
            forward/backward pair of transforms is scaled and with what 
            normalization factor.
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns: 
        out(Tensor): The result of the real 2-D FFT.

    Raises:


    Examples:

    .. code-block:: python
        import paddle
        import numpy as np

        x = paddle.to_tensor(np.mgrid[:5, :5][0].astype(np.float32))
        print(paddle.fft.rfft2(x)))
        # Tensor(shape=[5, 3], dtype=complex64, place=CUDAPlace(0), stop_gradient=True,
        #        [[ (50+0j)                                        ,  (1.1920928955078125e-07+0j)                    ,  0j                                             ],
        #         [(-12.5+17.204774856567383j)                     , (-9.644234211236835e-08+7.006946134424652e-08j) ,  0j                                             ],
        #         [(-12.500000953674316+4.061495304107666j)        , (3.6837697336977726e-08-1.1337477445749755e-07j),  0j                                             ],
        #         [(-12.500000953674316-4.061495304107666j)        , (3.6837697336977726e-08+1.1337477445749755e-07j),  0j                                             ],
        #         [(-12.5-17.204774856567383j)                     , (-9.644234211236835e-08-7.006946134424652e-08j) ,  0j                                             ]])
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return rfftn(x, s, axes, norm, name)


def irfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    """
    Computes the inverse of `rfft2`.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints, optional): Shape of the real output to the inverse FFT.
        axis (sequence of ints, optional): The axes over which to compute the inverse FFT. If not specified,
            the last two axes are used by default.       
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 
    
    Returns:
        Real tensor. The result of the inverse real 2-D FFT.
    
    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([1, 2, 3, 4, 3, 2])
            xp = paddle.to_tensor(x)
            irfft2_xp = paddle.tensor.fft.irfft2(xp).numpy()
            print(irfft2_xp)
            #  [15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]

    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return irfftn(x, s, axes, norm, name)


def hfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    """
    Compute the 2-D FFT of a Hermitian complex array.

    Args:
        x (Tensor): The input data. It's a Tensor type.
        s (sequence of ints, optional): Shape of the real output.
        axis (sequence of ints, optional):  Axes over which to compute the FFT. If not specified,
            the last two axes are used by default.       
        norm (str): Indicates which direction to scale the `forward` or `backward` transform
            pair and what normalization factor to use. The parameter value must be one 
            of "forward" or "backward" or "ortho". Default is "backward".
        name (str, optional): The default value is None.  Normally there is no need for user to set 
            this property. For more information, please refer to :ref:`api_guide_Name` . 
    
    Returns:
        Real tensor. The real result of the 2-D Hermitian complex real FFT.
    
    Examples:

        .. code-block:: python

            import numpy as np
            import paddle

            x = np.array([1, 2, 3, 4, 3, 2])
            xp = paddle.to_tensor(x)
            hfft2_xp = paddle.tensor.fft.hfft2(xp).numpy()
            print(hfft2_xp)
            #  [15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j]


    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return hfftn(x, s, axes, norm, name)


def ihfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    """
    Compute the two dimensional inverse FFT of a real spectrum.

    This is really `ihfftn` with different defaults.
    For more details see `ihfftn`.

    Args:
        x(Tensor): Input tensor
        s(Sequence[int], optional): Shape of the real input to the inverse FFT.
        axes(Sequance[int], optional): The axes over which to compute the 
            inverse fft. Default is the last two axes.
        norm(str, optional): {"backward", "ortho", "forward"}. Default is 
        "backward".
        name(str, optional): The default value is None.  Normally there is no 
            need for user to set this property. For more information, please 
            refer to :ref:`api_guide_Name` . 

    Returns:
        out(Tensor) : The result of the inverse real 2-D FFT.
    """
    _check_at_least_ndim(x, 2)
    if s is not None:
        if not isinstance(s, Sequence) or len(s) != 2:
            raise ValueError(
                "Invalid FFT argument s ({}), it should be a sequence of 2 integers.".
                format(s))
    if axes is not None:
        if not isinstance(axes, Sequence) or len(axes) != 2:
            raise ValueError(
                "Invalid FFT argument axes ({}), it should be a sequence of 2 integers.".
                format(axes))
    return ihfftn(x, s, axes, norm, name)


# public APIs utilities
def fftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = (n + 1) // 2
    neg_max = n // 2
    indices = paddle.arange(-neg_max, pos_max, dtype=dtype, name=name)
    indices = paddle.roll(indices, -neg_max, name=name)
    return indices * val


def rfftfreq(n, d=1.0, dtype=None, name=None):
    dtype = paddle.framework.get_default_dtype()
    val = 1.0 / (n * d)
    pos_max = 1 + n // 2
    indices = paddle.arange(0, pos_max, dtype=dtype, name=name)
    return indices * val


def fftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = [shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)


def ifftshift(x, axes=None, name=None):
    shape = paddle.shape(x)
    if axes is None:
        # shift all axes
        rank = paddle.rank(x).reshape([1])
        axes = axes or paddle.arange(0, rank)
        shifts = [-size // 2 for size in shape]
    elif isinstance(axes, int):
        shifts = -shape[axes] // 2
    else:
        shifts = [-shape[ax] // 2 for ax in axes]
    return paddle.roll(x, shifts, axes, name=name)


# internal functions
def fft_c2c(x, n, axis, norm, forward, name):
    if is_interger(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    _check_normalization(norm)

    axis = axis or -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n]
        x = _resize_fft_input(x, s, axes)
    op_type = 'fft_c2c'

    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)
    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fft_r2c(x, n, axis, norm, forward, onesided, name):
    if is_interger(x):
        x = paddle.cast(x, paddle.get_default_dtype())
    _check_normalization(norm)
    axis = axis or -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n]
        x = _resize_fft_input(x, s, axes)
    op_type = 'fft_r2c'
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], op_type)

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                 'onesided', onesided)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _real_to_complex_dtype(dtype))
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fft_c2r(x, n, axis, norm, forward, name):
    if is_interger(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    _check_normalization(norm)
    axis = axis or -1
    _check_fft_axis(x, axis)
    axes = [axis]
    axes = _normalize_axes(x, axes)
    if n is not None:
        _check_fft_n(n)
        s = [n // 2 + 1]
        x = _resize_fft_input(x, s, axes)
    op_type = 'fft_c2r'
    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)

    if in_dygraph_mode():
        if n is not None:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                     'last_dim_size', n)
        else:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if n is not None:
            attrs['last_dim_size'] = n
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _complex_to_real_dtype(dtype))
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fftn_c2c(x, s, axes, norm, forward, name):
    if is_interger(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes).tolist()
        axes = [axes[i] for i in axes_argsoft]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    "Length of s ({}) and length of axes ({}) does not match.".
                    format(len(s), len(axes)))
            s = [s[i] for i in axes_argsoft]

    if s is not None:
        x = _resize_fft_input(x, s, axes)
    op_type = 'fft_c2c'
    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out


def fftn_r2c(x, s, axes, norm, forward, onesided, name):
    if is_interger(x):
        x = paddle.cast(x, paddle.get_default_dtype())
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes[:-1]).tolist()
        axes = [axes[i] for i in axes_argsoft] + [axes[-1]]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    "Length of s ({}) and length of axes ({}) does not match.".
                    format(len(s), len(axes)))
            s = [s[i] for i in axes_argsoft] + [s[-1]]

    if s is not None:
        x = _resize_fft_input(x, s, axes)

    op_type = 'fft_r2c'
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], op_type)

    if in_dygraph_mode():
        attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                 'onesided', onesided)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {
            'axes': axes,
            'normalization': norm,
            'forward': forward,
            'onesided': onesided,
        }
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _real_to_complex_dtype(dtype))
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)

    return out


def fftn_c2r(x, s, axes, norm, forward, name):
    if is_interger(x):
        x = paddle.cast(x, _real_to_complex_dtype(paddle.get_default_dtype()))
    _check_normalization(norm)
    if s is not None:
        _check_fft_shape(x, s)

    rank = x.ndim
    if axes is None:
        if s is None:
            axes = list(range(rank))
        else:
            fft_ndims = len(s)
            axes = list(range(rank - fft_ndims, rank))
    else:
        _check_fft_axes(x, axes)
        axes = _normalize_axes(x, axes)
        axes_argsoft = np.argsort(axes[:-1]).tolist()
        axes = [axes[i] for i in axes_argsoft] + [axes[-1]]
        if s is not None:
            if len(s) != len(axes):
                raise ValueError(
                    "Length of s ({}) and length of axes ({}) does not match.".
                    format(len(s), len(axes)))
            s = [s[i] for i in axes_argsoft] + [s[-1]]

    if s is not None:
        fft_input_shape = list(s)
        fft_input_shape[-1] = fft_input_shape[-1] // 2 + 1
        x = _resize_fft_input(x, fft_input_shape, axes)

    op_type = 'fft_c2r'
    check_variable_and_dtype(x, 'x', ['complex64', 'complex128'], op_type)

    if in_dygraph_mode():
        if s:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward,
                     'last_dim_size', s[-1])
        else:
            attrs = ('axes', axes, 'normalization', norm, 'forward', forward)
        out = getattr(_C_ops, op_type)(x, *attrs)
    else:
        inputs = {'X': [x], }
        attrs = {'axes': axes, 'normalization': norm, 'forward': forward}
        if s:
            attrs["last_dim_size"] = s[-1]
        helper = LayerHelper(op_type, **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(
            _complex_to_real_dtype(dtype))
        outputs = {"Out": [out]}
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)
    return out
