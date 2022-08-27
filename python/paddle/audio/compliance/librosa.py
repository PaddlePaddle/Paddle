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
# Modified from librosa(https://github.com/librosa/librosa)

import warnings
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import scipy
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from scipy import fftpack

from ..utils import depth_convert
from ..utils import ParameterError

__all__ = [
    # dsp
    'stft',
    'mfcc',
    'hz_to_mel',
    'mel_to_hz',
    'mel_frequencies',
    'power_to_db',
    'compute_fbank_matrix',
    'melspectrogram',
    'spectrogram',
]


def _pad_center(data: np.ndarray,
                size: int,
                axis: int = -1,
                **kwargs) -> np.ndarray:
    """Pad an array to a target length along a target axis.

    This differs from `np.pad` by centering the data prior to padding,
    analogous to `str.center`
    """

    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(("Target size ({size:d}) must be "
                              "at least input size ({n:d})"))

    return np.pad(data, lengths, **kwargs)


def _split_frames(x: np.ndarray,
                  frame_length: int,
                  hop_length: int,
                  axis: int = -1) -> np.ndarray:
    """Slice a data array into (overlapping) frames.

    This function is aligned with librosa.frame
    """

    if not isinstance(x, np.ndarray):
        raise ParameterError(
            f"Input must be of type numpy.ndarray, given type(x)={type(x)}")

    if x.shape[axis] < frame_length:
        raise ParameterError(f"Input is too short (n={x.shape[axis]:d})"
                             f" for frame_length={frame_length:d}")

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length:d}")

    if axis == -1 and not x.flags["F_CONTIGUOUS"]:
        warnings.warn(f"librosa.util.frame called with axis={axis} "
                      "on a non-contiguous input. This will result in a copy.")
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags["C_CONTIGUOUS"]:
        warnings.warn(f"librosa.util.frame called with axis={axis} "
                      "on a non-contiguous input. This will result in a copy.")
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise ParameterError(f"Frame axis={axis} must be either 0 or -1")

    return as_strided(x, shape=shape, strides=strides)


def _check_audio(y, mono=True) -> bool:
    """Determine whether a variable contains valid audio data.

    The audio y must be a np.ndarray, ether 1-channel or two channel
    """
    if not isinstance(y, np.ndarray):
        raise ParameterError("Audio data must be of type numpy.ndarray")
    if y.ndim > 2:
        raise ParameterError(
            f"Invalid shape for audio ndim={y.ndim:d}, shape={y.shape}")

    if mono and y.ndim == 2:
        raise ParameterError(
            f"Invalid shape for mono audio ndim={y.ndim:d}, shape={y.shape}")

    if (mono and len(y) == 0) or (not mono and y.shape[1] < 0):
        raise ParameterError(f"Audio is empty ndim={y.ndim:d}, shape={y.shape}")

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError("Audio data must be floating-point")

    if not np.isfinite(y).all():
        raise ParameterError("Audio buffer is not finite everywhere")

    return True


def hz_to_mel(frequencies: Union[float, List[float], np.ndarray],
              htk: bool = False) -> np.ndarray:
    """Convert Hz to Mels.

    Args:
        frequencies (Union[float, List[float], np.ndarray]): Frequencies in Hz.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        np.ndarray: Frequency in mels.
    """
    freq = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if freq.ndim:
        # If we have array data, vectorize
        log_t = freq >= min_log_hz
        mels[log_t] = min_log_mel + \
            np.log(freq[log_t] / min_log_hz) / logstep
    elif freq >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(freq / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: Union[float, List[float], np.ndarray],
              htk: int = False) -> np.ndarray:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Union[float, List[float], np.ndarray]): Frequency in mels.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        np.ndarray: Frequencies in Hz.
    """
    mel_array = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mel_array / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel_array

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mel_array.ndim:
        # If we have vector data, vectorize
        log_t = mel_array >= min_log_mel
        freqs[log_t] = min_log_hz * \
            np.exp(logstep * (mel_array[log_t] - min_log_mel))
    elif mel_array >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mel_array - min_log_mel))

    return freqs


def mel_frequencies(n_mels: int = 128,
                    fmin: float = 0.0,
                    fmax: float = 11025.0,
                    htk: bool = False) -> np.ndarray:
    """Compute mel frequencies.

    Args:
        n_mels (int, optional): Number of mel bins. Defaults to 128.
        fmin (float, optional): Minimum frequency in Hz. Defaults to 0.0.
        fmax (float, optional): Maximum frequency in Hz. Defaults to 11025.0.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        np.ndarray: Vector of n_mels frequencies in Hz with shape `(n_mels,)`.
    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr: int, n_fft: int) -> np.ndarray:
    """Compute fourier frequencies.

    Args:
        sr (int): Sample rate.
        n_fft (int): FFT size.

    Returns:
        np.ndarray: FFT frequencies in Hz with shape `(n_fft//2 + 1,)`.
    """
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def compute_fbank_matrix(sr: int,
                         n_fft: int,
                         n_mels: int = 128,
                         fmin: float = 0.0,
                         fmax: Optional[float] = None,
                         htk: bool = False,
                         norm: str = "slaney",
                         dtype: type = np.float32) -> np.ndarray:
    """Compute fbank matrix.

    Args:
        sr (int): Sample rate.
        n_fft (int): FFT size.
        n_mels (int, optional): Number of mel bins. Defaults to 128.
        fmin (float, optional): Minimum frequency in Hz. Defaults to 0.0.
        fmax (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        htk (bool, optional): Use htk scaling. Defaults to False.
        norm (str, optional): Type of normalization. Defaults to "slaney".
        dtype (type, optional): Data type. Defaults to np.float32.


    Returns:
        np.ndarray: Mel transform matrix with shape `(n_mels, n_fft//2 + 1)`.
    """
    if norm != "slaney":
        raise ParameterError('norm must set to slaney')

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn("Empty filters detected in mel frequency basis. "
                      "Some channels will produce empty responses. "
                      "Try increasing your sampling rate (and fmax) or "
                      "reducing n_mels.")

    return weights


def stft(x: np.ndarray,
         n_fft: int = 2048,
         hop_length: Optional[int] = None,
         win_length: Optional[int] = None,
         window: str = "hann",
         center: bool = True,
         dtype: type = np.complex64,
         pad_mode: str = "reflect") -> np.ndarray:
    """Short-time Fourier transform (STFT).

    Args:
        x (np.ndarray): Input waveform in one dimension.
        n_fft (int, optional): FFT size. Defaults to 2048.
        hop_length (Optional[int], optional): Number of steps to advance between adjacent windows. Defaults to None.
        win_length (Optional[int], optional): The size of window. Defaults to None.
        window (str, optional): A string of window specification. Defaults to "hann".
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        dtype (type, optional): Data type of STFT results. Defaults to np.complex64.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to "reflect".

    Returns:
        np.ndarray: The complex STFT output with shape `(n_fft//2 + 1, num_frames)`.
    """
    _check_audio(x)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = signal.get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = _pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        if n_fft > x.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too small for input signal of length={x.shape[-1]}"
            )
        x = np.pad(x, int(n_fft // 2), mode=pad_mode)

    elif n_fft > x.shape[-1]:
        raise ParameterError(
            f"n_fft={n_fft} is too small for input signal of length={x.shape[-1]}"
        )

    # Window the time series.
    x_frames = _split_frames(x, frame_length=n_fft, hop_length=hop_length)
    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), x_frames.shape[1]),
                           dtype=dtype,
                           order="F")
    fft = np.fft  # use numpy fft as default
    # Constrain STFT block sizes to 256 KB
    MAX_MEM_BLOCK = 2**8 * 2**10
    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        stft_matrix[:,
                    bl_s:bl_t] = fft.rfft(fft_window * x_frames[:, bl_s:bl_t],
                                          axis=0)

    return stft_matrix


def power_to_db(spect: np.ndarray,
                ref: float = 1.0,
                amin: float = 1e-10,
                top_db: Optional[float] = 80.0) -> np.ndarray:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units. The function computes the scaling `10 * log10(x / ref)` in a numerically stable way.

    Args:
        spect (np.ndarray): STFT power spectrogram of an input waveform.
        ref (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): Minimum threshold. Defaults to 1e-10.
        top_db (Optional[float], optional): Threshold the output at `top_db` below the peak. Defaults to 80.0.

    Returns:
        np.ndarray: Power spectrogram in db scale.
    """
    spect = np.asarray(spect)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if np.issubdtype(spect.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.")
        magnitude = np.abs(spect)
    else:
        magnitude = spect

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def mfcc(x: np.ndarray,
         sr: int = 16000,
         spect: Optional[np.ndarray] = None,
         n_mfcc: int = 20,
         dct_type: int = 2,
         norm: str = "ortho",
         lifter: int = 0,
         **kwargs) -> np.ndarray:
    """Mel-frequency cepstral coefficients (MFCCs)

    Args:
        x (np.ndarray): Input waveform in one dimension.
        sr (int, optional): Sample rate. Defaults to 16000.
        spect (Optional[np.ndarray], optional): Input log-power Mel spectrogram. Defaults to None.
        n_mfcc (int, optional): Number of cepstra in MFCC. Defaults to 20.
        dct_type (int, optional): Discrete cosine transform (DCT) type. Defaults to 2.
        norm (str, optional): Type of normalization. Defaults to "ortho".
        lifter (int, optional): Cepstral filtering. Defaults to 0.

    Returns:
        np.ndarray: Mel frequency cepstral coefficients array with shape `(n_mfcc, num_frames)`.
    """
    if spect is None:
        spect = melspectrogram(x, sr=sr, **kwargs)

    M = fftpack.dct(spect, axis=0, type=dct_type, norm=norm)[:n_mfcc]

    if lifter > 0:
        factor = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) /
                        lifter)
        return M * factor[:, np.newaxis]
    elif lifter == 0:
        return M
    else:
        raise ParameterError(
            f"MFCC lifter={lifter} must be a non-negative number")


def melspectrogram(x: np.ndarray,
                   sr: int = 16000,
                   window_size: int = 512,
                   hop_length: int = 320,
                   n_mels: int = 64,
                   fmin: float = 50.0,
                   fmax: Optional[float] = None,
                   window: str = 'hann',
                   center: bool = True,
                   pad_mode: str = 'reflect',
                   power: float = 2.0,
                   to_db: bool = True,
                   ref: float = 1.0,
                   amin: float = 1e-10,
                   top_db: Optional[float] = None) -> np.ndarray:
    """Compute mel-spectrogram.

    Args:
        x (np.ndarray): Input waveform in one dimension.
        sr (int, optional): Sample rate. Defaults to 16000.
        window_size (int, optional): Size of FFT and window length. Defaults to 512.
        hop_length (int, optional): Number of steps to advance between adjacent windows. Defaults to 320.
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        fmin (float, optional): Minimum frequency in Hz. Defaults to 50.0.
        fmax (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        window (str, optional): A string of window specification. Defaults to "hann".
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to "reflect".
        power (float, optional): Exponent for the magnitude melspectrogram. Defaults to 2.0.
        to_db (bool, optional): Enable db scale. Defaults to True.
        ref (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): Minimum threshold. Defaults to 1e-10.
        top_db (Optional[float], optional): Threshold the output at `top_db` below the peak. Defaults to None.

    Returns:
        np.ndarray: The mel-spectrogram in power scale or db scale with shape `(n_mels, num_frames)`.
    """
    _check_audio(x, mono=True)
    if len(x) <= 0:
        raise ParameterError('The input waveform is empty')

    if fmax is None:
        fmax = sr // 2
    if fmin < 0 or fmin >= fmax:
        raise ParameterError('fmin and fmax must statisfy 0<fmin<fmax')

    s = stft(x,
             n_fft=window_size,
             hop_length=hop_length,
             win_length=window_size,
             window=window,
             center=center,
             pad_mode=pad_mode)

    spect_power = np.abs(s)**power
    fb_matrix = compute_fbank_matrix(sr=sr,
                                     n_fft=window_size,
                                     n_mels=n_mels,
                                     fmin=fmin,
                                     fmax=fmax)
    mel_spect = np.matmul(fb_matrix, spect_power)
    if to_db:
        return power_to_db(mel_spect, ref=ref, amin=amin, top_db=top_db)
    else:
        return mel_spect


def spectrogram(x: np.ndarray,
                sr: int = 16000,
                window_size: int = 512,
                hop_length: int = 320,
                window: str = 'hann',
                center: bool = True,
                pad_mode: str = 'reflect',
                power: float = 2.0) -> np.ndarray:
    """Compute spectrogram.

    Args:
        x (np.ndarray): Input waveform in one dimension.
        sr (int, optional): Sample rate. Defaults to 16000.
        window_size (int, optional): Size of FFT and window length. Defaults to 512.
        hop_length (int, optional): Number of steps to advance between adjacent windows. Defaults to 320.
        window (str, optional): A string of window specification. Defaults to "hann".
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to "reflect".
        power (float, optional): Exponent for the magnitude melspectrogram. Defaults to 2.0.

    Returns:
        np.ndarray: The STFT spectrogram in power scale `(n_fft//2 + 1, num_frames)`.
    """

    s = stft(x,
             n_fft=window_size,
             hop_length=hop_length,
             win_length=window_size,
             window=window,
             center=center,
             pad_mode=pad_mode)

    return np.abs(s)**power
