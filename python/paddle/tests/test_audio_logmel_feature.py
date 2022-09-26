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
import scipy
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

        self.fmin = 0.0
        self.top_db = 80.0
        self.duration = 0.5
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        waveform_tensor = get_wav_data(self.dtype,
                                       self.num_channels,
                                       num_frames=self.duration * self.sr)
        self.waveform = waveform_tensor.numpy()

    @parameterize([16000], ["hamming", "bohman"], [128], [128, 64], [64, 32],
                  [0.0, 50.0])
    def test_log_melspect(self, sr: int, window_str: str, n_fft: int,
                          hop_length: int, n_mels: int, fmin: float):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = librosa.feature.melspectrogram(y=self.waveform,
                                                         sr=sr,
                                                         n_fft=n_fft,
                                                         hop_length=hop_length,
                                                         window=window_str,
                                                         n_mels=n_mels,
                                                         center=True,
                                                         fmin=fmin,
                                                         pad_mode='reflect')
        feature_librosa = librosa.power_to_db(feature_librosa, top_db=None)
        x = paddle.to_tensor(self.waveform, dtype=paddle.float64).unsqueeze(
            0)  # Add batch dim.
        feature_extractor = paddle.audio.features.LogMelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window_str,
            center=True,
            n_mels=n_mels,
            f_min=fmin,
            top_db=None,
            dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_layer,
                                             decimal=2)
        # relative difference
        np.testing.assert_allclose(feature_librosa, feature_layer, rtol=1e-4)

    @parameterize([16000], [256, 128], [40, 64], [64, 128],
                  ['float32', 'float64'])
    def test_mfcc(self, sr: int, n_fft: int, n_mfcc: int, n_mels: int,
                  dtype: str):
        if paddle.version.cuda() != 'False':
            if float(paddle.version.cuda()) >= 11.0:
                return

        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        np_dtype = getattr(np, dtype)
        feature_librosa = librosa.feature.mfcc(y=self.waveform,
                                               sr=sr,
                                               S=None,
                                               n_mfcc=n_mfcc,
                                               dct_type=2,
                                               lifter=0,
                                               n_fft=n_fft,
                                               hop_length=64,
                                               n_mels=n_mels,
                                               fmin=50.0,
                                               dtype=np_dtype)
        # paddlespeech.audio.features.layer
        x = paddle.to_tensor(self.waveform,
                             dtype=dtype).unsqueeze(0)  # Add batch dim.
        feature_extractor = paddle.audio.features.MFCC(sr=sr,
                                                       n_mfcc=n_mfcc,
                                                       n_fft=n_fft,
                                                       hop_length=64,
                                                       n_mels=n_mels,
                                                       top_db=self.top_db,
                                                       dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_layer,
                                             decimal=3)

        np.testing.assert_allclose(feature_librosa, feature_layer, rtol=1e-1)

        # split mffcc: logmel-->dct --> mfcc, which prove the difference.
        # the dct module is correct.
        feature_extractor = paddle.audio.features.LogMelSpectrogram(
            sr=sr,
            n_fft=n_fft,
            hop_length=64,
            n_mels=n_mels,
            center=True,
            pad_mode='reflect',
            top_db=self.top_db,
            dtype=x.dtype)
        feature_layer_logmel = feature_extractor(x).squeeze(0).numpy()

        feature_layer_mfcc = scipy.fftpack.dct(feature_layer_logmel,
                                               axis=0,
                                               type=2,
                                               norm="ortho")[:n_mfcc]
        np.testing.assert_array_almost_equal(feature_layer_mfcc,
                                             feature_librosa,
                                             decimal=3)
        np.testing.assert_allclose(feature_layer_mfcc,
                                   feature_librosa,
                                   rtol=1e-1)


if __name__ == '__main__':
    unittest.main()
