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
import math
from typing import Optional
from typing import Union

import paddle
from paddle import Tensor

__all__ = [
    'hz_to_mel',
    'mel_to_hz',
    'mel_frequencies',
    'fft_frequencies',
    'compute_fbank_matrix',
    'power_to_db',
    'create_dct',
]


def hz_to_mel(freq: Union[Tensor, float],
              htk: bool = False) -> Union[Tensor, float]:
    """Convert Hz to Mels.

    Args:
        freq (Union[Tensor, float]): The input tensor with arbitrary shape.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        Union[Tensor, float]: Frequency in mels.
    """

    if htk:
        if isinstance(freq, Tensor):
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

    if isinstance(freq, Tensor):
        target = min_log_mel + paddle.log(
            freq / min_log_hz + 1e-10) / logstep  # prevent nan with 1e-10
        mask = (freq > min_log_hz).astype(freq.dtype)
        mels = target * mask + mels * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz + 1e-10) / logstep

    return mels


def mel_to_hz(mel: Union[float, Tensor],
              htk: bool = False) -> Union[float, Tensor]:
    """Convert mel bin numbers to frequencies.

    Args:
        mel (Union[float, Tensor]): The mel frequency represented as a tensor with arbitrary shape.
        htk (bool, optional): Use htk scaling. Defaults to False.

    Returns:
        Union[float, Tensor]: Frequencies in Hz.
    """
    if htk:
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel
    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region
    if isinstance(mel, Tensor):
        target = min_log_hz * paddle.exp(logstep * (mel - min_log_mel))
        mask = (mel > min_log_mel).astype(mel.dtype)
        freqs = target * mask + freqs * (
            1 - mask)  # will replace by masked_fill OP in future
    else:
        if mel >= min_log_mel:
            freqs = min_log_hz * math.exp(logstep * (mel - min_log_mel))
    return freqs


def mel_frequencies(n_mels: int = 64,
                    f_min: float = 0.0,
                    f_max: float = 11025.0,
                    htk: bool = False,
                    dtype: str = 'float32') -> Tensor:
    """Compute mel frequencies.

    Args:
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 0.0.
        fmax (float, optional): Maximum frequency in Hz. Defaults to 11025.0.
        htk (bool, optional): Use htk scaling. Defaults to False.
        dtype (str, optional): The data type of the return frequencies. Defaults to 'float32'.

    Returns:
        Tensor: Tensor of n_mels frequencies in Hz with shape `(n_mels,)`.
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
    """
    return paddle.linspace(0, float(sr) / 2, int(1 + n_fft // 2), dtype=dtype)


def compute_fbank_matrix(sr: int,
                         n_fft: int,
                         n_mels: int = 64,
                         f_min: float = 0.0,
                         f_max: Optional[float] = None,
                         htk: bool = False,
                         norm: Union[str, float] = 'slaney',
                         dtype: str = 'float32') -> Tensor:
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
    """

    if f_max is None:
        f_max = float(sr) / 2

    # Initialize the weights
    weights = paddle.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft, dtype=dtype)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2,
                            f_min=f_min,
                            f_max=f_max,
                            htk=htk,
                            dtype=dtype)

    fdiff = mel_f[1:] - mel_f[:-1]  #np.diff(mel_f)
    ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
    #ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = paddle.maximum(paddle.zeros_like(lower),
                                    paddle.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    if norm == 'slaney':
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm.unsqueeze(1)
    elif isinstance(norm, int) or isinstance(norm, float):
        weights = paddle.nn.functional.normalize(weights, p=norm, axis=-1)

    return weights


def power_to_db(spect: Tensor,
                ref_value: float = 1.0,
                amin: float = 1e-10,
                top_db: Optional[float] = 80.0) -> Tensor:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units. The function computes the scaling `10 * log10(x / ref)` in a numerically stable way.

    Args:
        spect (Tensor): STFT power spectrogram.
        ref_value (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): Minimum threshold. Defaults to 1e-10.
        top_db (Optional[float], optional): Threshold the output at `top_db` below the peak. Defaults to None.

    Returns:
        Tensor: Power spectrogram in db scale.
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


def create_dct(n_mfcc: int,
               n_mels: int,
               norm: Optional[str] = 'ortho',
               dtype: str = 'float32') -> Tensor:
    """Create a discrete cosine transform(DCT) matrix.

    Args:
        n_mfcc (int): Number of mel frequency cepstral coefficients.
        n_mels (int): Number of mel filterbanks.
        norm (Optional[str], optional): Normalizaiton type. Defaults to 'ortho'.
        dtype (str, optional): The data type of the return matrix. Defaults to 'float32'.

    Returns:
        Tensor: The DCT matrix with shape `(n_mels, n_mfcc)`.
    """
    n = paddle.arange(n_mels, dtype=dtype)
    k = paddle.arange(n_mfcc, dtype=dtype).unsqueeze(1)
    dct = paddle.cos(math.pi / float(n_mels) * (n + 0.5) *
                     k)  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.T
