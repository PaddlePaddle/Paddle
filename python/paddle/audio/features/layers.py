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
from functools import partial
from typing import Optional, Union

import paddle
from paddle import Tensor, nn

from ..functional import compute_fbank_matrix, create_dct, power_to_db
from ..functional.window import get_window


class Spectrogram(nn.Layer):
    """Compute spectrogram of given signals, typically audio waveforms.
    The spectrogram is defined as the complex norm of the short-time Fourier transformation.

    Args:
        n_fft (int, optional): The number of frequency components of the discrete Fourier transform. Defaults to 512.
        hop_length (Optional[int], optional): The hop length of the short time FFT. If `None`, it is set to `win_length//4`. Defaults to None.
        win_length (Optional[int], optional): The window length of the short time FFT. If `None`, it is set to same as `n_fft`. Defaults to None.
        window (str, optional): The window function applied to the signal before the Fourier transform. Supported window functions: 'hamming', 'hann', 'kaiser', 'gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor'. Defaults to 'hann'.
        power (float, optional): Exponent for the magnitude spectrogram. Defaults to 2.0.
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to 'reflect'.
        dtype (str, optional): Data type of input and window. Defaults to 'float32'.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of Spectrogram.



    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.audio.features import Spectrogram

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])

            >>> feature_extractor = Spectrogram(n_fft=512, window = 'hann', power = 1.0)
            >>> feats = feature_extractor(waveform)
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: Optional[int] = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        power: float = 1.0,
        center: bool = True,
        pad_mode: str = 'reflect',
        dtype: str = 'float32',
    ) -> None:
        super().__init__()

        assert power > 0, 'Power of spectrogram must be > 0.'
        self.power = power

        if win_length is None:
            win_length = n_fft

        self.fft_window = get_window(
            window, win_length, fftbins=True, dtype=dtype
        )
        self._stft = partial(
            paddle.signal.stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.fft_window,
            center=center,
            pad_mode=pad_mode,
        )
        self.register_buffer('fft_window', self.fft_window)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of waveforms with shape `(N, T)`

        Returns:
            Tensor: Spectrograms with shape `(N, n_fft//2 + 1, num_frames)`.
        """
        stft = self._stft(x)
        spectrogram = paddle.pow(paddle.abs(stft), self.power)
        return spectrogram


class MelSpectrogram(nn.Layer):
    """Compute the melspectrogram of given signals, typically audio waveforms. It is computed by multiplying spectrogram with Mel filter bank matrix.

    Args:
        sr (int, optional): Sample rate. Defaults to 22050.
        n_fft (int, optional): The number of frequency components of the discrete Fourier transform. Defaults to 512.
        hop_length (Optional[int], optional): The hop length of the short time FFT. If `None`, it is set to `win_length//4`. Defaults to None.
        win_length (Optional[int], optional): The window length of the short time FFT. If `None`, it is set to same as `n_fft`. Defaults to None.
        window (str, optional): The window function applied to the signal before the Fourier transform. Supported window functions: 'hamming', 'hann', 'kaiser', 'gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor'. Defaults to 'hann'.
        power (float, optional): Exponent for the magnitude spectrogram. Defaults to 2.0.
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to 'reflect'.
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 50.0.
        f_max (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        htk (bool, optional): Use HTK formula in computing fbank matrix. Defaults to False.
        norm (Union[str, float], optional): Type of normalization in computing fbank matrix. Slaney-style is used by default. You can specify norm=1.0/2.0 to use customized p-norm normalization. Defaults to 'slaney'.
        dtype (str, optional): Data type of input and window. Defaults to 'float32'.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MelSpectrogram.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.audio.features import MelSpectrogram

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])

            >>> feature_extractor = MelSpectrogram(sr=sample_rate, n_fft=512, window = 'hann', power = 1.0)
            >>> feats = feature_extractor(waveform)
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 2048,
        hop_length: Optional[int] = 512,
        win_length: Optional[int] = None,
        window: str = 'hann',
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = 'reflect',
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: Optional[float] = None,
        htk: bool = False,
        norm: Union[str, float] = 'slaney',
        dtype: str = 'float32',
    ) -> None:
        super().__init__()

        self._spectrogram = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            power=power,
            center=center,
            pad_mode=pad_mode,
            dtype=dtype,
        )
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.htk = htk
        self.norm = norm
        if f_max is None:
            f_max = sr // 2
        self.fbank_matrix = compute_fbank_matrix(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            dtype=dtype,
        )
        self.register_buffer('fbank_matrix', self.fbank_matrix)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of waveforms with shape `(N, T)`

        Returns:
            Tensor: Mel spectrograms with shape `(N, n_mels, num_frames)`.
        """
        spect_feature = self._spectrogram(x)
        mel_feature = paddle.matmul(self.fbank_matrix, spect_feature)
        return mel_feature


class LogMelSpectrogram(nn.Layer):
    """Compute log-mel-spectrogram feature of given signals, typically audio waveforms.

    Args:
        sr (int, optional): Sample rate. Defaults to 22050.
        n_fft (int, optional): The number of frequency components of the discrete Fourier transform. Defaults to 512.
        hop_length (Optional[int], optional): The hop length of the short time FFT. If `None`, it is set to `win_length//4`. Defaults to None.
        win_length (Optional[int], optional): The window length of the short time FFT. If `None`, it is set to same as `n_fft`. Defaults to None.
        window (str, optional): The window function applied to the signal before the Fourier transform. Supported window functions: 'hamming', 'hann', 'kaiser', 'gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor'. Defaults to 'hann'.
        power (float, optional): Exponent for the magnitude spectrogram. Defaults to 2.0.
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to 'reflect'.
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 50.0.
        f_max (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        htk (bool, optional): Use HTK formula in computing fbank matrix. Defaults to False.
        norm (Union[str, float], optional): Type of normalization in computing fbank matrix. Slaney-style is used by default. You can specify norm=1.0/2.0 to use customized p-norm normalization. Defaults to 'slaney'.
        ref_value (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): The minimum value of input magnitude. Defaults to 1e-10.
        top_db (Optional[float], optional): The maximum db value of spectrogram. Defaults to None.
        dtype (str, optional): Data type of input and window. Defaults to 'float32'.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of LogMelSpectrogram.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.audio.features import LogMelSpectrogram

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])

            >>> feature_extractor = LogMelSpectrogram(sr=sample_rate, n_fft=512, window = 'hann', power = 1.0)
            >>> feats = feature_extractor(waveform)
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = 'reflect',
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: Optional[float] = None,
        htk: bool = False,
        norm: Union[str, float] = 'slaney',
        ref_value: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
        dtype: str = 'float32',
    ) -> None:
        super().__init__()

        self._melspectrogram = MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            power=power,
            center=center,
            pad_mode=pad_mode,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            dtype=dtype,
        )

        self.ref_value = ref_value
        self.amin = amin
        self.top_db = top_db

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of waveforms with shape `(N, T)`

        Returns:
            Tensor: Log mel spectrograms with shape `(N, n_mels, num_frames)`.
        """
        mel_feature = self._melspectrogram(x)
        log_mel_feature = power_to_db(
            mel_feature,
            ref_value=self.ref_value,
            amin=self.amin,
            top_db=self.top_db,
        )
        return log_mel_feature


class MFCC(nn.Layer):
    """Compute mel frequency cepstral coefficients(MFCCs) feature of given waveforms.

    Args:
        sr (int, optional): Sample rate. Defaults to 22050.
        n_mfcc (int, optional): [description]. Defaults to 40.
        n_fft (int, optional): The number of frequency components of the discrete Fourier transform. Defaults to 512.
        hop_length (Optional[int], optional): The hop length of the short time FFT. If `None`, it is set to `win_length//4`. Defaults to None.
        win_length (Optional[int], optional): The window length of the short time FFT. If `None`, it is set to same as `n_fft`. Defaults to None.
        window (str, optional): The window function applied to the signal before the Fourier transform. Supported window functions: 'hamming', 'hann', 'kaiser', 'gaussian', 'exponential', 'triang', 'bohman', 'blackman', 'cosine', 'tukey', 'taylor'. Defaults to 'hann'.
        power (float, optional): Exponent for the magnitude spectrogram. Defaults to 2.0.
        center (bool, optional): Whether to pad `x` to make that the :math:`t \times hop\\_length` at the center of `t`-th frame. Defaults to True.
        pad_mode (str, optional): Choose padding pattern when `center` is `True`. Defaults to 'reflect'.
        n_mels (int, optional): Number of mel bins. Defaults to 64.
        f_min (float, optional): Minimum frequency in Hz. Defaults to 50.0.
        f_max (Optional[float], optional): Maximum frequency in Hz. Defaults to None.
        htk (bool, optional): Use HTK formula in computing fbank matrix. Defaults to False.
        norm (Union[str, float], optional): Type of normalization in computing fbank matrix. Slaney-style is used by default. You can specify norm=1.0/2.0 to use customized p-norm normalization. Defaults to 'slaney'.
        ref_value (float, optional): The reference value. If smaller than 1.0, the db level of the signal will be pulled up accordingly. Otherwise, the db level is pushed down. Defaults to 1.0.
        amin (float, optional): The minimum value of input magnitude. Defaults to 1e-10.
        top_db (Optional[float], optional): The maximum db value of spectrogram. Defaults to None.
        dtype (str, optional): Data type of input and window. Defaults to 'float32'.

    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of MFCC.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.audio.features import MFCC

            >>> sample_rate = 16000
            >>> wav_duration = 0.5
            >>> num_channels = 1
            >>> num_frames = sample_rate * wav_duration
            >>> wav_data = paddle.linspace(-1.0, 1.0, num_frames) * 0.1
            >>> waveform = wav_data.tile([num_channels, 1])

            >>> feature_extractor = MFCC(sr=sample_rate, n_fft=512, window = 'hann')
            >>> feats = feature_extractor(waveform)
    """

    def __init__(
        self,
        sr: int = 22050,
        n_mfcc: int = 40,
        n_fft: int = 512,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: str = 'hann',
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = 'reflect',
        n_mels: int = 64,
        f_min: float = 50.0,
        f_max: Optional[float] = None,
        htk: bool = False,
        norm: Union[str, float] = 'slaney',
        ref_value: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[float] = None,
        dtype: str = 'float32',
    ) -> None:
        super().__init__()
        assert (
            n_mfcc <= n_mels
        ), 'n_mfcc cannot be larger than n_mels: %d vs %d' % (n_mfcc, n_mels)
        self._log_melspectrogram = LogMelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            power=power,
            center=center,
            pad_mode=pad_mode,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            htk=htk,
            norm=norm,
            ref_value=ref_value,
            amin=amin,
            top_db=top_db,
            dtype=dtype,
        )
        self.dct_matrix = create_dct(n_mfcc=n_mfcc, n_mels=n_mels, dtype=dtype)
        self.register_buffer('dct_matrix', self.dct_matrix)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor of waveforms with shape `(N, T)`

        Returns:
            Tensor: Mel frequency cepstral coefficients with shape `(N, n_mfcc, num_frames)`.
        """
        log_mel_feature = self._log_melspectrogram(x)
        mfcc = paddle.matmul(
            log_mel_feature.transpose((0, 2, 1)), self.dct_matrix
        ).transpose(
            (0, 2, 1)
        )  # (B, n_mels, L)
        return mfcc
