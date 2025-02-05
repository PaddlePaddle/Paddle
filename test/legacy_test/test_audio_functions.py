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
import itertools
import unittest

import librosa
import numpy as np
from parameterized import parameterized
from scipy import signal

import paddle
import paddle.audio


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)))


class TestAudioFunctions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.initParmas()

    def initParmas(self):
        def get_wav_data(dtype: str, num_channels: int, num_frames: int):
            dtype_ = getattr(paddle, dtype)
            base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_) * 0.1
            data = base.tile([num_channels, 1])
            return data

        self.n_fft = 512
        self.hop_length = 128
        self.n_mels = 40
        self.n_mfcc = 20
        self.fmin = 0.0
        self.window_str = 'hann'
        self.pad_mode = 'reflect'
        self.top_db = 80.0
        self.duration = 0.5
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        self.window_size = 1024
        waveform_tensor = get_wav_data(
            self.dtype,
            self.num_channels,
            num_frames=int(self.duration * self.sr),
        )
        self.waveform = waveform_tensor.numpy()

    @parameterize([1.0, 3.0, 9.0, 25.0], [True, False])
    def test_audio_function(self, val: float, htk_flag: bool):
        mel_paddle = paddle.audio.functional.hz_to_mel(val, htk_flag)
        mel_paddle_tensor = paddle.audio.functional.hz_to_mel(
            paddle.to_tensor([val]), htk_flag
        )
        mel_librosa = librosa.hz_to_mel(val, htk=htk_flag)
        np.testing.assert_almost_equal(mel_paddle, mel_librosa, decimal=5)
        np.testing.assert_almost_equal(
            mel_paddle_tensor.numpy(), mel_librosa, decimal=3
        )

        hz_paddle = paddle.audio.functional.mel_to_hz(val, htk_flag)
        hz_paddle_tensor = paddle.audio.functional.mel_to_hz(
            paddle.to_tensor([val]), htk_flag
        )
        hz_librosa = librosa.mel_to_hz(val, htk=htk_flag)
        np.testing.assert_almost_equal(hz_paddle, hz_librosa, decimal=4)
        np.testing.assert_almost_equal(
            hz_paddle_tensor.numpy(), hz_librosa, decimal=4
        )

        decibel_paddle = paddle.audio.functional.power_to_db(
            paddle.to_tensor([val])
        )
        decibel_librosa = librosa.power_to_db(val)
        np.testing.assert_almost_equal(
            decibel_paddle.numpy(), decibel_paddle, decimal=5
        )

    @parameterize([1.0, 3.0, 9.0, 25.0], [True, False])
    def test_audio_function_static(self, val: float, htk_flag: bool):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            mel_paddle_tensor = paddle.audio.functional.hz_to_mel(
                paddle.to_tensor([val]), htk_flag
            )

            hz_paddle_tensor = paddle.audio.functional.mel_to_hz(
                paddle.to_tensor([val]), htk_flag
            )

            decibel_paddle = paddle.audio.functional.power_to_db(
                paddle.to_tensor([val])
            )

            exe = paddle.static.Executor()
            (
                mel_paddle_tensor_ret,
                hz_paddle_tensor_ret,
                decibel_paddle_ret,
            ) = exe.run(
                main,
                fetch_list=[
                    mel_paddle_tensor,
                    hz_paddle_tensor,
                    decibel_paddle,
                ],
            )

            mel_librosa = librosa.hz_to_mel(val, htk=htk_flag)
            np.testing.assert_almost_equal(
                mel_paddle_tensor_ret, mel_librosa, decimal=3
            )

            hz_librosa = librosa.mel_to_hz(val, htk=htk_flag)
            np.testing.assert_almost_equal(
                hz_paddle_tensor_ret, hz_librosa, decimal=4
            )

            decibel_librosa = librosa.power_to_db(val)
            np.testing.assert_almost_equal(
                decibel_paddle_ret, decibel_librosa, decimal=5
            )

        paddle.disable_static()

    @parameterize(
        [64, 128, 256], [0.0, 0.5, 1.0], [10000, 11025], [False, True]
    )
    def test_audio_function_mel(
        self, n_mels: int, f_min: float, f_max: float, htk_flag: bool
    ):
        librosa_mel_freq = librosa.mel_frequencies(
            n_mels, fmin=f_min, fmax=f_max, htk=htk_flag
        )
        paddle_mel_freq = paddle.audio.functional.mel_frequencies(
            n_mels, f_min, f_max, htk_flag, 'float64'
        )
        np.testing.assert_almost_equal(
            paddle_mel_freq, librosa_mel_freq, decimal=3
        )

    @parameterize(
        [64, 128, 256], [0.0, 0.5, 1.0], [10000, 11025], [False, True]
    )
    # TODO(MarioLulab) May cause precision error. Fix it soon

    def test_audio_function_mel_static(
        self, n_mels: int, f_min: float, f_max: float, htk_flag: bool
    ):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            paddle_mel_freq = paddle.audio.functional.mel_frequencies(
                n_mels, f_min, f_max, htk_flag, 'float64'
            )

        exe = paddle.static.Executor()
        (paddle_mel_freq_ret,) = exe.run(main, fetch_list=[paddle_mel_freq])
        librosa_mel_freq = librosa.mel_frequencies(
            n_mels, fmin=f_min, fmax=f_max, htk=htk_flag
        )
        np.testing.assert_almost_equal(
            paddle_mel_freq_ret, librosa_mel_freq, decimal=3
        )

        paddle.disable_static()

    @parameterize([8000, 16000], [64, 128, 256])
    def test_audio_function_fft(self, sr: int, n_fft: int):
        librosa_fft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        paddle_fft = paddle.audio.functional.fft_frequencies(sr, n_fft)
        np.testing.assert_almost_equal(paddle_fft, librosa_fft, decimal=5)

    @parameterize([1.0, 3.0, 9.0])
    def test_audio_function_exception(self, spect: float):
        try:
            paddle.audio.functional.power_to_db(
                paddle.to_tensor([spect]), amin=0
            )
        except Exception:
            pass

        try:
            paddle.audio.functional.power_to_db(
                paddle.to_tensor([spect]), ref_value=0
            )

        except Exception:
            pass

        try:
            paddle.audio.functional.power_to_db(
                paddle.to_tensor([spect]), top_db=-1
            )
        except Exception:
            pass

    @parameterize(
        [
            "hamming",
            "hann",
            "triang",
            "bohman",
            "blackman",
            "cosine",
            "tukey",
            "taylor",
        ],
        [1, 512],
    )
    def test_window(self, window_type: str, n_fft: int):
        window_scipy = signal.get_window(window_type, n_fft)
        window_paddle = paddle.audio.functional.get_window(window_type, n_fft)
        np.testing.assert_array_almost_equal(
            window_scipy, window_paddle.numpy(), decimal=5
        )

    @parameterize([1, 512])
    def test_gaussian_window_and_exception(self, n_fft: int):
        window_scipy_gaussain = signal.windows.gaussian(n_fft, std=7)
        window_paddle_gaussian = paddle.audio.functional.get_window(
            ('gaussian', 7), n_fft, False
        )
        np.testing.assert_array_almost_equal(
            window_scipy_gaussain, window_paddle_gaussian.numpy(), decimal=5
        )
        window_scipy_general_gaussain = signal.windows.general_gaussian(
            n_fft, 1, 7
        )
        window_paddle_general_gaussian = paddle.audio.functional.get_window(
            ('general_gaussian', 1, 7), n_fft, False
        )
        np.testing.assert_array_almost_equal(
            window_scipy_gaussain, window_paddle_gaussian.numpy(), decimal=5
        )

        window_scipy_exp = signal.windows.exponential(n_fft)
        window_paddle_exp = paddle.audio.functional.get_window(
            ('exponential', None, 1), n_fft, False
        )
        np.testing.assert_array_almost_equal(
            window_scipy_exp, window_paddle_exp.numpy(), decimal=5
        )

        try:
            window_paddle = paddle.audio.functional.get_window("hann", -1)
        except ValueError:
            pass

        try:
            window_paddle = paddle.audio.functional.get_window(
                "fake_window", self.n_fft
            )
        except ValueError:
            pass

        try:
            window_paddle = paddle.audio.functional.get_window(1043, self.n_fft)
        except ValueError:
            pass

    @parameterize([5, 13, 23], [257, 513, 1025])
    def test_create_dct(self, n_mfcc: int, n_mels: int):
        def dct(n_filters, n_input):
            basis = np.empty((n_filters, n_input))
            basis[0, :] = 1.0 / np.sqrt(n_input)
            samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

            for i in range(1, n_filters):
                basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / n_input)
            return basis.T

        librosa_dct = dct(n_mfcc, n_mels)
        paddle_dct = paddle.audio.functional.create_dct(n_mfcc, n_mels)
        np.testing.assert_array_almost_equal(librosa_dct, paddle_dct, decimal=5)

    @parameterize(
        [128, 256, 512],
        [
            "hamming",
            "hann",
            "triang",
            "bohman",
        ],
        [True, False],
    )
    def test_stft_and_spect(
        self, n_fft: int, window_str: str, center_flag: bool
    ):
        hop_length = int(n_fft / 4)
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0
            )  # 1D input for librosa.feature.melspectrogram
        feature_librosa = librosa.core.stft(
            y=self.waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window=window_str,
            center=center_flag,
            dtype=None,
            pad_mode=self.pad_mode,
        )
        x = paddle.to_tensor(self.waveform).unsqueeze(0)
        window = paddle.audio.functional.get_window(
            window_str, n_fft, dtype=x.dtype
        )
        feature_paddle = paddle.signal.stft(
            x=x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window=window,
            center=center_flag,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
        ).squeeze(0)
        np.testing.assert_array_almost_equal(
            feature_librosa, feature_paddle, decimal=5
        )

        feature_bg = np.power(np.abs(feature_librosa), 2.0)
        feature_extractor = paddle.audio.features.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window=window_str,
            power=2.0,
            center=center_flag,
            pad_mode=self.pad_mode,
        )
        feature_layer = feature_extractor(x).squeeze(0)
        np.testing.assert_array_almost_equal(
            feature_layer, feature_bg, decimal=3
        )

    @parameterize(
        [128, 256, 512],
        [64, 82],
        [
            "hamming",
            "hann",
            "triang",
            "bohman",
        ],
    )
    def test_istft(self, n_fft: int, hop_length: int, window_str: str):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0
            )  # 1D input for librosa.feature.melspectrogram
        # librosa
        # Get stft result from librosa.
        stft_matrix = librosa.core.stft(
            y=self.waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window=window_str,
            center=True,
            pad_mode=self.pad_mode,
        )
        feature_librosa = librosa.core.istft(
            stft_matrix=stft_matrix,
            hop_length=hop_length,
            win_length=None,
            window=window_str,
            center=True,
            dtype=None,
            length=None,
        )
        x = paddle.to_tensor(stft_matrix).unsqueeze(0)
        window = paddle.audio.functional.get_window(
            window_str, n_fft, dtype=paddle.to_tensor(self.waveform).dtype
        )
        feature_paddle = paddle.signal.istft(
            x=x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=None,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False,
        ).squeeze(0)

        np.testing.assert_array_almost_equal(
            feature_librosa, feature_paddle, decimal=5
        )


if __name__ == '__main__':
    unittest.main()
