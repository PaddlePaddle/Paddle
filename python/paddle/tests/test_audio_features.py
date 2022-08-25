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
import unittest

import librosa
import numpy as np
import paddle

import paddle.audio
from audio_base import get_wav_data 
from paddle.audio.functional.window import get_window


class TestLibrosa(unittest.TestCase):
    def setUp(self):
        self.initParmas()

    def initParmas(self):
        self.n_fft = 512
        self.hop_length = 128
        self.n_mels = 40
        self.n_mfcc = 20
        self.fmin = 0.0
        self.window_str = 'hann'
        self.pad_mode = 'reflect'
        self.top_db = 80.0
        self.duration = 1
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        waveform_tensor = get_wav_data(self.dtype,
                            self.num_channels,
                            normalize=False,
                            num_frames=self.duration * self.sr)
        self.waveform = waveform_tensor.numpy() * 0.1
        print(self.waveform.shape)
        print(self.waveform)

    def test_stft(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        feature_librosa = librosa.core.stft(
            y=self.waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=self.window_str,
            center=True,
            dtype=None,
            pad_mode=self.pad_mode, )
        x = paddle.to_tensor(self.waveform).unsqueeze(0)
        window = get_window(self.window_str, self.n_fft, dtype=x.dtype)
        feature_paddle = paddle.signal.stft(
            x=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=window,
            center=True,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True, ).squeeze(0)

        np.testing.assert_array_almost_equal(
            feature_librosa, feature_paddle, decimal=5)

    def test_istft(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # Get stft result from librosa.
        stft_matrix = librosa.core.stft(
            y=self.waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=self.window_str,
            center=True,
            pad_mode=self.pad_mode, )

        feature_librosa = librosa.core.istft(
            stft_matrix=stft_matrix,
            hop_length=self.hop_length,
            win_length=None,
            window=self.window_str,
            center=True,
            dtype=None,
            length=None, )

        x = paddle.to_tensor(stft_matrix).unsqueeze(0)
        window = get_window(
            self.window_str,
            self.n_fft,
            dtype=paddle.to_tensor(self.waveform).dtype)
        feature_paddle = paddle.signal.istft(
            x=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False, ).squeeze(0)

        np.testing.assert_array_almost_equal(
            feature_librosa, feature_paddle, decimal=5)

    def test_mel(self):
        feature_librosa = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=None,
            htk=False,
            norm='slaney',
            dtype=self.waveform.dtype, )
        feature_compliance = paddle.audio.compliance.librosa.compute_fbank_matrix(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=None,
            htk=False,
            norm='slaney',
            dtype=self.waveform.dtype, )
        x = paddle.to_tensor(self.waveform)
        feature_functional = paddle.audio.functional.compute_fbank_matrix(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=None,
            htk=False,
            norm='slaney',
            dtype=x.dtype, )

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_compliance)
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_functional)

    def test_melspect(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = librosa.feature.melspectrogram(
            y=self.waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin)

        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.melspectrogram(
            x=self.waveform,
            sr=self.sr,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            to_db=False)

        # paddle.audio.features.layer
        x = paddle.to_tensor(
            self.waveform, dtype=paddle.float64).unsqueeze(0)  # Add batch dim.
        feature_extractor = paddle.audio.features.MelSpectrogram(
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(
            feature_librosa, feature_compliance, decimal=5)
        np.testing.assert_array_almost_equal(
            feature_librosa, feature_layer, decimal=5)

    def test_log_melspect(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = librosa.feature.melspectrogram(
            y=self.waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin)
        feature_librosa = librosa.power_to_db(feature_librosa, top_db=None)

        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.melspectrogram(
            x=self.waveform,
            sr=self.sr,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin)

    def test_mfcc(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = librosa.feature.mfcc(
            y=self.waveform,
            sr=self.sr,
            S=None,
            n_mfcc=self.n_mfcc,
            dct_type=2,
            norm='ortho',
            lifter=0,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin)

        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.mfcc(
            x=self.waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            dct_type=2,
            norm='ortho',
            lifter=0,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            top_db=self.top_db)

if __name__ == '__main__':
    unittest.main()
