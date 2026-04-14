# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# Modified from librosa(https://github.com/librosa/librosa)
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypeVar

import paddle
from paddle import Tensor
from paddle.base.framework import Variable
from paddle.pir import Value

if TYPE_CHECKING:
    _TensorOrFloat = TypeVar("_TensorOrFloat", Tensor, float)


def hz_to_mel(freq: _TensorOrFloat, htk: bool = False) -> _TensorOrFloat:
    """Convert Hz to Mels.

    Args:
        freq (Union[Tensor, float]): The input tensor with arbitrary shape.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        Union[Tensor, float]: Frequency in mels.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> val = 3.0
            >>> htk_flag = True
            >>> mel_paddle_tensor = paddle.audio.functional.hz_to_mel(paddle.to_tensor(val), htk_flag)
    """

    if htk:
        if isinstance(freq, (Tensor, Variable, Value)):
            return 2595.0 * paddle.log10(1.0 + freq / 700.0)
        else:
            return 2595.0 * math.log10(1.0 + freq / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if isinstance(freq, (Tensor, Variable, Value)):
        target = (
            min_log_mel + paddle.log(freq / min_log_hz + 1e-10) / logstep
        )  # prevent nan with 1e-10
        mask = (freq > min_log_hz).astype(freq.dtype)
        mels = target * mask + mels * (
            1 - mask
        )  # will replace by masked_fill OP in future
    else:
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz + 1e-10) / logstep

    return mels


def mel_to_hz(mel: _TensorOrFloat, htk: bool = False) -> _TensorOrFloat:
    """Convert mel bin numbers to frequencies.

    Args:
        mel (Union[float, Tensor]): The mel frequency represented as a tensor with arbitrary shape.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        Union[float, Tensor]: Frequencies in Hz.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> val = 3.0
            >>> htk_flag = True
            >>> mel_paddle_tensor = paddle.audio.functional.mel_to_hz(paddle.to_tensor(val), htk_flag)
    """
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel
    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region
    if isinstance(mel, (Tensor, Variable, Value)):
        target = min_log_hz * paddle.exp(logstep * (mel - min_log_mel))
        mask = (mel > min_log_mel).astype(mel.dtype)
        freqs = target * mask + freqs * (
            1 - mask
        )  # will replace by masked_fill OP in future
    else:
        if mel >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mel - min_log_mel))
    return freqs


def mel_frequencies(
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: float = 11025.0,
    htk: bool = False,
    dtype: str = 'float32',
) -> Tensor:
    """Compute mel frequencies.

    Args:
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 0.0.
        fmax (float, optional): Maximum frequency in Hz. Defaults to 11025.0.
        htk (bool, optional): Use htk scaling. Defaults to False.
        dtype (str, optional): The data type of the return frequencies. Defaults to 'float32'.

    Returns:
        Tensor: Tensor of n_mels frequencies in Hz with shape `(n_mels,)`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> n_mels = 64
            >>> f_min = 0.5
            >>> f_max = 10000
            >>> htk_flag = True

            >>> paddle_mel_freq = paddle.audio.functional.mel_frequencies(n_mels, f_min, f_max, htk_flag, 'float64')
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(f_min, htk=htk)
    max_mel = hz_to_mel(f_max, htk=htk)
    mels = paddle.linspace(min_mel, max_mel, n_mels, dtype=dtype)
    freqs = mel_to_hz(mels, htk=htk)
    return freqs


def fft_frequencies(sr: int, n_fft: int, dtype: str = 'float32') -> Tensor:
    """Compute fourier frequencies.

    Args:
        sr (int): Sample rate.
        n_fft (int): Number of fft bins.
        dtype (str, optional): The data type of the return frequencies. Defaults to 'float32'.

    Returns:
        Tensor: FFT frequencies in Hz with shape `(n_fft//2 + 1,)`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> sr = 16000
            >>> n_fft = 128
            >>> fft_freq = paddle.audio.functional.fft_frequencies(sr, n_fft)
    """
    return paddle.linspace(0, float(sr) / 2, int(1 + n_fft // 2), dtype=dtype)


def compute_fbank_matrix(
    sr: int,
    n_fft: int,
    n_mels: int = 64,
    f_min: float = 0.0,
    f_max: float | None = None,
    htk: bool = False,
    norm: Literal['slaney'] | float = 'slaney',
    dtype: str = 'float32',
) -> Tensor:
    """Compute fbank matrix.

    Args:
        sr (int): Sample rate.
        n_fft (int): Number of fft bins.
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 0.0.
        f_max (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        htk (bool, optional): Use htk scaling. Defaults to False.
        norm (Union[str, float], optional): Type of normalization. Defaults to 'slaney'.
        dtype (str, optional): The data type of the return matrix. Defaults to 'float32'.

    Returns:
        Tensor: Mel transform matrix with shape `(n_mels, n_fft//2 + 1)`.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> sr = 23
            >>> n_fft = 51
            >>> fbank = paddle.audio.functional.compute_fbank_matrix(sr, n_fft)
    """

    if f_max is None:
        f_max = float(sr) / 2

    # Initialize the weights
    weights = paddle.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft, dtype=dtype)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(
        n_mels + 2, f_min=f_min, f_max=f_max, htk=htk, dtype=dtype
    )

    fdiff = mel_f[1:] - mel_f[:-1]  # np.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    # ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = paddle.maximum(
            paddle.zeros_like(lower), paddle.minimum(lower, upper)
        )

    # Slaney-style mel is scaled to be approx constant energy per channel
    if norm == 'slaney':
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)
    elif isinstance(norm, (int, float)):
        weights = paddle.nn.functional.normalize(weights, p=norm, axis=-1)

    return weights


def power_to_db(
    spect: Tensor,
    ref_value: float = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units. The function computes the scaling `10 * log10(x / ref)` in a numerically stable way.

    Args:
        spect (Tensor): STFT power spectrogram.
        ref_value (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): Minimum threshold. Defaults to 1e-10.
        top_db (Optional[float], optional): Threshold the output at `top_db` below the peak. Defaults to None.

    Returns:
        Tensor: Power spectrogram in db scale.

    Examples:
        .. code-block:: pycon

            >>> import paddle

            >>> val = 3.0
            >>> decibel_paddle = paddle.audio.functional.power_to_db(paddle.to_tensor(val))
    """
    if amin <= 0:
        raise Exception("amin must be strictly positive")

    if ref_value <= 0:
        raise Exception("ref_value must be strictly positive")

    ones = paddle.ones_like(spect)
    log_spec = 10.0 * paddle.log10(paddle.maximum(ones * amin, spect))
    log_spec -= 10.0 * math.log10(max(ref_value, amin))

    if top_db is not None:
        if top_db < 0:
            raise Exception("top_db must be non-negative")
        log_spec = paddle.maximum(log_spec, ones * (log_spec.max() - top_db))

    return log_spec


def create_dct(
    n_mfcc: int,
    n_mels: int,
    norm: Literal['ortho'] | None = 'ortho',
    dtype: str = 'float32',
) -> Tensor:
    """Create a discrete cosine transform(DCT) matrix.

    Args:
        n_mfcc (int): Number of mel frequency cepstral coefficients.
        n_mels (int): Number of mel filterbanks.
        norm (Optional[str], optional): Normalization type. Defaults to 'ortho'.
        dtype (str, optional): The data type of the return matrix. Defaults to 'float32'.

    Returns:
        Tensor: The DCT matrix with shape `(n_mels, n_mfcc)`.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> n_mfcc = 23
            >>> n_mels = 257
            >>> dct = paddle.audio.functional.create_dct(n_mfcc, n_mels)
    """
    n = paddle.arange(n_mels, dtype=dtype)
    k = paddle.arange(n_mfcc, dtype=dtype).unsqueeze(1)
    dct = paddle.cos(
        math.pi / float(n_mels) * (n + 0.5) * k
    )  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.T


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: Literal[
        "sinc_interp_hann", "sinc_interp_kaiser"
    ] = "sinc_interp_hann",
    beta: float | None = None,
    dtype: paddle.dtype | None = None,
):
    """
    Generate the sinc interpolation kernel for resampling.

    This internal function computes the resampling kernel based on the sinc
    interpolation formula with windowing. The kernel is used by
    _apply_sinc_resample_kernel to perform the actual resampling.

    Args:
        orig_freq (int): Original sampling frequency.
        new_freq (int): Target sampling frequency.
        gcd (int): Greatest common divisor of orig_freq and new_freq.
        lowpass_filter_width (int, optional): Controls the sharpness of the filter,
            larger value means sharper but less efficient. Default: 6.
        rolloff (float, optional): Roll-off frequency as a fraction of the Nyquist.
            Lower values reduce anti-aliasing but also attenuate high frequencies.
            Default: 0.99.
        resampling_method (str, optional): Window method for filter design.
            Options: ["sinc_interp_hann", "sinc_interp_kaiser"]. Default: "sinc_interp_hann".
        beta (float, optional): Shape parameter for Kaiser window. Required only
            when resampling_method="sinc_interp_kaiser". Default: None.
        dtype (paddle.dtype, optional): Data type for kernel computation.
            If None, uses float64 for computation and converts to float32 for output.
            Default: None.

    Returns:
        tuple: (kernel, width)
            - kernel (Tensor): Resampling kernel of shape (1, 1, kernel_width)
            - width (int): Half-width of the filter in terms of input samples

    Raises:
        Exception: If frequencies are not integers.
        ValueError: If resampling_method is invalid or lowpass_filter_width <= 0.
    """
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise ValueError(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
        )

    if resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError(f"Invalid resampling method: {resampling_method}")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)

    # Perform antialiasing filtering by removing the highest frequencies.
    base_freq *= rolloff

    # Calculate filter width based on lowpass_filter_width and frequency ratio
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else paddle.float64

    idx = (
        paddle.arange(-width, width + orig_freq, dtype=idx_dtype)[None, None]
        / orig_freq
    )

    t = (
        paddle.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq
        + idx
    )
    t *= base_freq
    t = t.clip_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built-in paddle windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = paddle.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = paddle.to_tensor(float(beta))
        window = paddle.i0(
            beta_tensor * paddle.sqrt(1 - (t / lowpass_filter_width) ** 2),
        ) / paddle.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = paddle.where(
        t == 0, paddle.to_tensor(1.0).cast(t.dtype), t.sin() / t
    )
    kernels *= window * scale

    if dtype is None:  # pragma: no cover
        kernels = kernels.cast(paddle.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    """
    Apply sinc interpolation resampling using precomputed kernel.

    This internal function performs the actual resampling operation using the
    kernel generated by _get_sinc_resample_kernel. It handles batch processing
    and ensures correct output length.

    Args:
        waveform (Tensor): Input waveform of shape (..., time). Must be floating point.
        orig_freq (int): Original sampling frequency.
        new_freq (int): Target sampling frequency.
        gcd (int): Greatest common divisor of orig_freq and new_freq.
        kernel (Tensor): Resampling kernel from _get_sinc_resample_kernel.
        width (int): Half-width of the filter from _get_sinc_resample_kernel.

    Returns:
        Tensor: Resampled waveform of shape (..., new_time).

    """

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.shape
    waveform = waveform.reshape([-1, shape[-1]])

    num_wavs, length = waveform.shape
    waveform = paddle.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = paddle.nn.functional.conv1d(
        waveform[:, None], kernel, stride=orig_freq
    )
    resampled = resampled.transpose([0, 2, 1]).reshape((num_wavs, -1))
    target_length = paddle.ceil(
        paddle.to_tensor(new_freq * length / orig_freq)
    ).astype(paddle.int64)
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.reshape(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: Literal[
        "sinc_interp_hann", "sinc_interp_kaiser"
    ] = "sinc_interp_hann",
    beta: float | None = None,
) -> Tensor:
    """
    Resample the waveform from orig_freq to new_freq using bandlimited interpolation.

    This function implements resampling through sinc interpolation with windowing.
    It first computes a resampling kernel based on the specified parameters, then
    applies it to the input waveform using convolution. The algorithm handles both
    upsampling and downsampling while minimizing aliasing artifacts.

    Args:
        waveform (Tensor): The input signal of dimension (..., time). Must be
            floating point type (float32 or float64).
        orig_freq (int): The original frequency of the signal. Must be positive.
        new_freq (int): The desired target frequency. Must be positive.
        lowpass_filter_width (int, optional): Controls the sharpness of the filter.
            Larger values give sharper filtering but are less efficient.
            Default: 6.
        rolloff (float, optional): The roll-off frequency of the filter as a fraction
            of the Nyquist frequency. Lower values reduce anti-aliasing but also
            attenuate some high frequencies. Default: 0.99.
        resampling_method (str, optional): The windowing method to use for filter
            design. Options: "sinc_interp_hann" (Hann window) or "sinc_interp_kaiser"
            (Kaiser window). Default: "sinc_interp_hann".
        beta (float, optional): Shape parameter for the Kaiser window. Required only
            when resampling_method="sinc_interp_kaiser". If not provided for Kaiser,
            a default value of 14.769656459379492 is used. Default: None.

    Returns:
        Tensor: The waveform resampled to new_freq, with dimension (..., new_time).

    Raises:
        ValueError: If orig_freq or new_freq are not positive.
        Exception: If frequencies are not integers (see note below).
        TypeError: If waveform is not floating point.

    Note:
        - orig_freq and new_freq must be integers. For non-integer frequencies,
          convert them to integers while maintaining the ratio.
        - For repeated resampling with same parameters, use
          :class:`paddle.audio.transforms.Resample` for better efficiency.
        - Uses windowed sinc interpolation for high-quality audio resampling.
        - This function does not support ONNX export now.

    Examples:
        .. code-block:: pycon

            >>> import paddle
            >>> from paddle.audio.functional import resample

            >>> # Create a sample waveform (1 channel, 1000 samples at 16000 Hz)
            >>> waveform = paddle.randn([1, 1000])

            >>> # Downsample from 16000 Hz to 8000 Hz
            >>> resampled = resample(waveform, 16000, 8000)
            >>> print(resampled.shape)
            paddle.Size([1, 500])

            >>> # Upsample from 16000 Hz to 48000 Hz with custom filter width
            >>> resampled = resample(waveform, 16000, 48000, lowpass_filter_width=12)
            >>> print(resampled.shape)
            paddle.Size([1, 3000])

            >>> # Use Kaiser window resampling
            >>> resampled = resample(waveform, 16000, 8000, resampling_method="sinc_interp_kaiser", beta=12.0)
            >>> print(resampled.shape)
            paddle.Size([1, 500])

            >>> # Batch processing: multiple waveforms
            >>> batch_waveforms = paddle.randn([4, 1, 1000])  # [batch, channels, time]
            >>> resampled_batch = resample(batch_waveforms, 16000, 8000)
            >>> print(resampled_batch.shape)
            paddle.Size([4, 1, 500])
    """
    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError(
            "Original frequency and desired frequency should be positive integers"
        )
    if not waveform.is_floating_point():
        raise TypeError(
            f"Expected floating point type for waveform tensor, but received {waveform.dtype}."
        )

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(
        waveform, orig_freq, new_freq, gcd, kernel, width
    )
    return resampled
