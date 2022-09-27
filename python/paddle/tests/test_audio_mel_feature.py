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
import os
import paddle

import paddle.audio
from scipy import signal
import itertools
from parameterized import parameterized


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)))


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.initParmas()

    def initParmas(self):

        def get_wav_data(dtype: str, num_channels: int, num_frames: int):
            dtype_ = getattr(paddle, dtype)
            base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_) * 0.1
            data = base.tile([num_channels, 1])
            return data

        self.hop_length = 128
        self.duration = 0.5
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        waveform_tensor = get_wav_data(self.dtype,
                                       self.num_channels,
                                       num_frames=self.duration * self.sr)
        self.waveform = waveform_tensor.numpy()

    @parameterize([8000], [128, 256], [64, 32], [0.0, 1.0],
                  ['float32', 'float64'])
    def test_mel(self, sr: int, n_fft: int, n_mels: int, fmin: float,
                 dtype: str):
        feature_librosa = librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=None,
            htk=False,
            norm='slaney',
            dtype=np.dtype(dtype),
        )
        paddle_dtype = getattr(paddle, dtype)
        feature_functional = paddle.audio.functional.compute_fbank_matrix(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            f_min=fmin,
            f_max=None,
            htk=False,
            norm='slaney',
            dtype=paddle_dtype,
        )

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_functional)

    @parameterize([8000, 16000], [128, 256], [64, 82], [40, 80], [False, True])
    def test_melspect(self, sr: int, n_fft: int, hop_length: int, n_mels: int,
                      htk: bool):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = librosa.feature.melspectrogram(y=self.waveform,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         n_mels=n_mels,
                                                         htk=htk,
                                                         fmin=50.0)

        # paddle.audio.features.layer
        x = paddle.to_tensor(self.waveform, dtype=paddle.float64).unsqueeze(
            0)  # Add batch dim.
        feature_extractor = paddle.audio.features.MelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            htk=htk,
            dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_layer,
                                             decimal=5)


if __name__ == '__main__':
    unittest.main()
